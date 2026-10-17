"""Write the plan file of a causal test (input of dd3d.intervene on Kaggle).

The plan has these conditions for each selected test image:

- base: no change (the SDEdit reconstruction);
- shift:probe_exact: the shortest feature change that turns the probe output
  (sin, cos) by the rotation angle. The probe then reads the azimuth of its
  own prediction + 90 deg.
- shift:probe_r<r>: a larger change with the same probe angle. The mean norm of
  a cell vector is about r times the mean token norm of the block.
- shift:pattern_x<k>: k times the mean feature change between images whose
  light differs by this rotation. A regression of the features on
  (sin az, cos az) over the train split gives this change (the activation
  pattern of Haufe et al., 2014). With k = 1, the size of the change is the
  natural size in the data.
- shift:random_<family>: a random change with the same norm in each cell as
  the largest shift of the family (probe or pattern);
- erase: one LEACE eraser of the azimuth (sin and cos), fitted on the cells of
  all positions. It removes only the code that is the same at every position;
- erase_cell (option --cell-eraser): one LEACE eraser for each grid cell. After
  it, no linear function of the grid features carries the azimuth;
- erase_cell_random (option --cell-eraser-control): the same erasers, fitted on
  labels in a random order. They remove a subspace of the same rank and the
  same whitened variance, but not the light code. This is the control of erase_cell.

The plan report gives the error of a new probe on the erased features
(probe_after_erasure). This shows what each eraser removes before the GPU test.

A probe weight vector shows where the probe reads. It can also contain
directions that only cancel noise. The pattern shows how the features change
with the light, so it is the better candidate for a change that the model uses.

The probe, the pattern and the eraser come from the grid features of one
block and one timestep (train split). In the test mode (--test-job), random
values replace them, so the job can be tested before the features exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.intervene import apply_cell_erasers, cell_erasers, leace_fit, make_shift
from dd3d.metrics import decode_azimuth, encode_targets
from dd3d.probe import ProbeConfig, probe_direction, run
from dd3d.store import FeatureStore


def mean_token_norm(store: FeatureStore, key: str) -> float:
    """Return the mean token norm of the run, block and timestep of a key."""
    run_name, block, t = key.split("/")
    stats = store.stats(run_name)
    sel = stats[(stats["block"] == int(block.split("_")[1])) & (stats["t"] == int(t.split("_")[1]))]
    return float(sel["mean_token_norm"].mean())


def as_cells(v: np.ndarray, grid: int, channels: int) -> np.ndarray:
    """Return the (n, grid * grid, channels) float32 view of pooled features."""
    return v.reshape(len(v), grid * grid, channels).astype(np.float32)


def pattern_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return B (k, D) of the least squares fit x - mean(x) = (y - mean(y)) B."""
    yc = y - y.mean(0)
    xc = x - x.mean(0)
    return np.linalg.solve(yc.T @ yc, yc.T @ xc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Feature root (see dd3d.store).")
    parser.add_argument("--key", default="sana/block_10/t_400", help="Run, block and timestep.")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--split", default="test", help="Split of the images to change.")
    parser.add_argument("--n-images", type=int, default=60)
    parser.add_argument("--relative", default="1.0", help="Probe shift norm / mean token norm.")
    parser.add_argument("--pattern-scales", default="1,3", help="Multiples of the natural pattern shift.")
    parser.add_argument("--token-norm", type=float, default=None, help="Default: from the stats files.")
    parser.add_argument("--rotate", type=float, default=90.0, help="Target azimuth = label + rotate.")
    parser.add_argument("--leace-rows", type=int, default=30000, help="Cell rows for the LEACE fit.")
    parser.add_argument("--channels", type=int, default=2240)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--no-erase", action="store_true")
    parser.add_argument(
        "--cell-eraser", action="store_true", help="Also add erase_cell: one LEACE eraser for each grid cell."
    )
    parser.add_argument(
        "--cell-eraser-control",
        action="store_true",
        help="Also add erase_cell_random: the cell erasers of labels in a random order.",
    )
    parser.add_argument("--only-erase", action="store_true", help="Only base and the erase conditions.")
    parser.add_argument("--test-job", action="store_true", help="Random probe and eraser (test of the job).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", args.split)}
    pick = splits[args.split].sample(min(args.n_images, len(splits[args.split])), random_state=args.seed)
    d = args.grid * args.grid * args.channels
    report = {"key": args.key, "split": args.split, "n_images": len(pick), "rotate": args.rotate}

    if args.test_job:
        w = rng.normal(size=(2, d)) / np.sqrt(d)
        b = rng.normal(size=(2, d))
        cells = rng.normal(size=(5000, args.channels))
        z = cells[:, :2] + 0.1 * rng.normal(size=(5000, 2))
        token_norm = args.token_norm or 1.0
        before = encode_targets(rng.uniform(0, 360, len(pick)), np.zeros(len(pick)))[:, :2]
        report["mode"] = "test of the job (random probe, pattern and eraser)"
        z_img = rng.normal(size=(200, 2))
        x = rng.normal(size=(200, args.grid * args.grid, args.channels))
        cell_parts = cell_erasers(x, z_img) if args.cell_eraser else None
        ctrl_parts = cell_erasers(x, rng.permutation(z_img)) if args.cell_eraser_control else None
    else:
        store = FeatureStore(args.features)
        labels = {
            k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy())
            for k, v in splits.items()
        }
        feats = {k: store.load(args.key, v["image_id"].tolist()) for k, v in splits.items()}
        res = run(feats, labels, ProbeConfig())
        w = probe_direction(res.model)[:2]
        report["probe"] = {k: {"az_mae": v["az_mae"], "el_mae": v["el_mae"]} for k, v in res.splits.items()}
        z_img = encode_targets(*labels["train"])[:, :2]
        b = pattern_of(feats["train"], z_img)
        # Each grid cell of each train image is one LEACE sample with the label of its image.
        x = feats["train"].reshape(len(feats["train"]), args.grid * args.grid, args.channels)
        rows = rng.choice(
            x.shape[0] * x.shape[1], size=min(args.leace_rows, x.shape[0] * x.shape[1]), replace=False
        )
        cells = x.reshape(-1, args.channels)[rows]
        z = np.repeat(z_img, x.shape[1], axis=0)[rows]
        token_norm = args.token_norm or mean_token_norm(store, args.key)
        rows_pick = splits[args.split].index.get_indexer(pick.index)
        before = res.model.predict(feats[args.split][rows_pick])[:, :2]
        cell_parts = cell_erasers(x, z_img) if args.cell_eraser else None
        ctrl_parts = cell_erasers(x, rng.permutation(z_img)) if args.cell_eraser_control else None
    eraser_mean, eraser_matrix = (None, None) if args.no_erase else leace_fit(cells, z)
    erasers = {}
    if not args.no_erase:
        erasers["erase"] = lambda v: v - (v - eraser_mean) @ eraser_matrix.T
    if cell_parts is not None:
        erasers["erase_cell"] = lambda v: apply_cell_erasers(v, *cell_parts)
    if ctrl_parts is not None:
        erasers["erase_cell_random"] = lambda v: apply_cell_erasers(v, *ctrl_parts)
    if not args.test_job and erasers:
        # A new probe on the erased features shows what each eraser removes. The
        # eraser is affine, so the cell means of the erased tokens are the erased cell means.
        # The mean change of a cell vector shows if the control changes the features as much.
        report["probe_after_erasure"], report["erasure_change_norm"] = {}, {}
        for name, fn in erasers.items():
            erased = {
                k: fn(as_cells(v, args.grid, args.channels)).reshape(len(v), -1) for k, v in feats.items()
            }
            res_e = run(erased, labels, ProbeConfig())
            report["probe_after_erasure"][name] = {k: v["az_mae"] for k, v in res_e.splits.items()}
            change = as_cells(erased["test"] - feats["test"], args.grid, args.channels)
            report["erasure_change_norm"][name] = float(np.linalg.norm(change, axis=2).mean())
            del erased
        print(json.dumps(report["probe_after_erasure"], indent=1), flush=True)
    if not args.test_job:
        del feats, x

    az = pick["light_azimuth_deg"].to_numpy()
    el = pick["light_elevation_deg"].to_numpy()
    delta_y = encode_targets(az + args.rotate, el)[:, :2] - encode_targets(az, el)[:, :2]

    def cells_of(v):
        return v.reshape(len(v), args.grid * args.grid, args.channels)

    def cell_norm(v):
        return float(np.linalg.norm(cells_of(v), axis=2).mean())

    def random_like(v):
        r = cells_of(rng.normal(size=v.shape))
        r *= (np.linalg.norm(cells_of(v), axis=2) / np.linalg.norm(r, axis=2))[:, :, None]
        return r.reshape(len(r), -1)

    # Turn the predicted (sin, cos) vector of each image by the rotation angle.
    a = np.radians(args.rotate)
    turned = np.stack(
        [
            before[:, 0] * np.cos(a) + before[:, 1] * np.sin(a),
            before[:, 1] * np.cos(a) - before[:, 0] * np.sin(a),
        ],
        axis=1,
    )
    to_turned, from_before = make_shift(w, turned), make_shift(w, before)
    exact = to_turned - from_before
    # A larger shift multiplies the turned output by a factor s for each image. The
    # angle stays the same, and the mean cell norm is about r times the token norm.
    s_unit = token_norm / np.linalg.norm(cells_of(to_turned), axis=2).mean(axis=1)[:, None]
    natural = delta_y @ b
    conditions, shifts = ["base", "shift:probe_exact"], [np.zeros((len(pick), d)), exact]
    if args.only_erase:
        conditions, shifts, args.relative, args.pattern_scales = ["base"], [np.zeros((len(pick), d))], "", ""
    rel = [float(v) for v in args.relative.split(",") if v]
    for r in rel:
        conditions.append(f"shift:probe_r{r:g}")
        shifts.append(r * s_unit * to_turned - from_before)
    if rel:
        conditions.append("shift:random_probe")
        shifts.append(random_like(max(rel) * s_unit * to_turned - from_before))
    scales = [float(v) for v in args.pattern_scales.split(",") if v]
    for k in scales:
        conditions.append(f"shift:pattern_x{k:g}")
        shifts.append(k * natural)
    if scales:
        conditions.append("shift:random_pattern")
        shifts.append(random_like(max(scales) * natural))
    if not args.no_erase:
        conditions.append("erase")
        shifts.append(np.zeros((len(pick), d)))
    if cell_parts is not None:
        conditions.append("erase_cell")
        shifts.append(np.zeros((len(pick), d)))
    if ctrl_parts is not None:
        conditions.append("erase_cell_random")
        shifts.append(np.zeros((len(pick), d)))
    shifts = np.stack(shifts, axis=1).astype(np.float16)  # (n_img, n_cond, D)
    # The probe is linear, so its output after a shift is known before the test.
    # This rotation of the probe output shows how far each shift moves the readout.
    az_before = decode_azimuth(before[:, 0], before[:, 1])
    report["probe_rotation_deg"] = {}
    for i, c in enumerate(conditions):
        after = before + shifts[:, i].astype(np.float64) @ w.T
        rot = (decode_azimuth(after[:, 0], after[:, 1]) - az_before + 180.0) % 360.0 - 180.0
        report["probe_rotation_deg"][c] = float(rot.mean())
    report["mean_token_norm"] = token_norm
    report["cell_norm"] = {c: cell_norm(shifts[:, i].astype(np.float32)) for i, c in enumerate(conditions)}
    report["relative_shifts"] = rel
    report["pattern_scales"] = scales
    report["conditions"] = conditions

    args.out.mkdir(parents=True, exist_ok=True)
    extra = {} if eraser_mean is None else {"eraser_mean": eraser_mean, "eraser_matrix": eraser_matrix}
    if cell_parts is not None:
        extra |= dict(
            zip(("eraser_cell_mean", "eraser_cell_left", "eraser_cell_right"), cell_parts, strict=True)
        )
    if ctrl_parts is not None:
        extra |= dict(
            zip(
                ("eraser_cell_random_mean", "eraser_cell_random_left", "eraser_cell_random_right"),
                ctrl_parts,
                strict=True,
            )
        )
    np.savez(
        args.out / "plan.npz",
        image_ids=pick["image_id"].to_numpy().astype(str),
        shapes=pick["shape"].to_numpy().astype(str),
        azimuth=az,
        conditions=np.array(conditions),
        shifts=shifts,
        **extra,
    )
    (args.out / "plan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
