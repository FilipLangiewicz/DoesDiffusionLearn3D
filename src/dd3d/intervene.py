"""Causal tests on SANA-1.6B: shift or erase the light direction (RQ4).

The test starts from a render. It encodes the image, adds flow noise up to the
level sigma0 and integrates the flow back to sigma = 0 with Euler steps
(SDEdit). A forward hook on one transformer block changes the tokens at each
step. The model gets all conditions of one image as one batch:

- base: no change;
- shift: add a vector to the tokens of each grid cell. Each image has its own
  vector, made from the weights of a linear probe (see make_shift);
- erase: apply a LEACE eraser to every token.

The output images go to a pixel probe, which measures the light direction.
The probe works on the mean of the tokens in each cell. A vector added to all
tokens of a cell changes this mean by the same vector, so the probe output
changes exactly as planned. The question is whether the image changes too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dd3d.extract import MODEL_ID, encode_prompt, preprocess
from dd3d.noise import image_seed


def leace_fit(x: np.ndarray, z: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and the matrix A of the LEACE eraser r(x) = x - A (x - mean).

    LEACE (Belrose et al., 2023): A = S^(1/2) P S^(-1/2), where S is the covariance
    of x and P is the orthogonal projection on the columns of S^(-1/2) cov(x, z).
    After the eraser, no linear function of x has a covariance with z.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mean = x.mean(0)
    xc, zc = x - mean, z - z.mean(0)
    cov = xc.T @ xc / len(x)
    cross = xc.T @ zc / len(x)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > eps * vals.max()
    root = (vecs[:, keep] * np.sqrt(vals[keep])) @ vecs[:, keep].T
    inv_root = (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T
    u, s, _ = np.linalg.svd(inv_root @ cross, full_matrices=False)
    u = u[:, s > eps * s.max()]
    a = root @ (u @ u.T) @ inv_root
    return mean.astype(np.float32), a.astype(np.float32)


def make_shift(w: np.ndarray, delta_y: np.ndarray) -> np.ndarray:
    """Return the shortest feature change d with w @ d = delta_y.

    ``w`` has shape (k, D) (the probe rows for sin and cos of the azimuth) and
    ``delta_y`` has shape (n, k). The result has shape (n, D).
    """
    gram = w @ w.T
    return delta_y @ np.linalg.solve(gram, w)


def spread_cells(delta: torch.Tensor, grid: int, side: int) -> torch.Tensor:
    """Spread (B, grid * grid * C) cell vectors to (B, side * side, C) tokens (row-major)."""
    b = delta.shape[0]
    k = side // grid
    x = delta.reshape(b, grid, 1, grid, 1, -1).expand(b, grid, k, grid, k, -1)
    return x.reshape(b, side * side, -1)


@dataclass
class InterveneConfig:
    images_dir: Path
    out_dir: Path
    plan: Path  # npz with image_ids, conditions, shifts (n_img, n_cond, D) and the eraser
    block: int = 10
    grid: int = 8
    sigma0: float = 0.7
    steps: int = 20
    seed: int = 1234
    dtype: str = "float16"
    device: str = "cuda"
    text_encoder_device: str = "cpu"
    image_size: int = 1024
    model_id: str = MODEL_ID


def run(cfg: InterveneConfig) -> Path:
    """Write one PNG for each image and condition, and a CSV of the run."""
    from diffusers import AutoencoderDC, SanaTransformer2DModel

    from dd3d.extract import ExtractConfig

    torch.set_grad_enabled(False)
    dtype = getattr(torch, cfg.dtype)
    plan = np.load(cfg.plan, allow_pickle=False)
    image_ids = [str(s) for s in plan["image_ids"]]
    shapes = [str(s) for s in plan["shapes"]]
    conditions = [str(s) for s in plan["conditions"]]  # "base", "shift:<name>", "erase"
    shifts = plan["shifts"]  # (n_img, n_cond, D); zero rows for base and erase
    t0 = time.time()

    ecfg = ExtractConfig(
        images_dir=cfg.images_dir,
        out_dir=cfg.out_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        text_encoder_device=cfg.text_encoder_device,
        model_id=cfg.model_id,
    )
    embeds, mask = encode_prompt(ecfg, dtype)
    vae = AutoencoderDC.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=dtype).to(cfg.device)
    model = SanaTransformer2DModel.from_pretrained(
        cfg.model_id, subfolder="transformer", torch_dtype=dtype, variant="bf16"
    ).to(cfg.device)
    model.eval()

    n_c = len(conditions)
    erase_rows = torch.tensor([c == "erase" for c in conditions], device=cfg.device)
    eraser = None
    if "eraser_mean" in plan.files and erase_rows.any():
        eraser = (
            torch.tensor(plan["eraser_mean"], device=cfg.device, dtype=torch.float32),
            torch.tensor(plan["eraser_matrix"], device=cfg.device, dtype=torch.float32),
        )
    state: dict[str, torch.Tensor | None] = {"delta": None}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h32 = h.float()
        if eraser is not None:
            mean, a = eraser
            erased = h32 - (h32 - mean) @ a.T
            h32 = torch.where(erase_rows[:, None, None], erased, h32)
        if state["delta"] is not None:
            h32 = h32 + state["delta"]
        out = h32.to(h.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    handle = model.transformer_blocks[cfg.block].register_forward_hook(hook)
    enc = embeds.to(cfg.device).expand(n_c, *embeds.shape[1:])
    enc_mask = mask.to(cfg.device).expand(n_c, *mask.shape[1:])
    sigmas = np.linspace(cfg.sigma0, 0.0, cfg.steps + 1)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (image_id, shape) in enumerate(zip(image_ids, shapes, strict=True)):
        x = preprocess(cfg.images_dir / shape / f"{image_id}.png", cfg.image_size).to(cfg.device, dtype)
        x0 = vae.encode(x).latent * vae.config.scaling_factor
        side = x0.shape[-1]
        gen = torch.Generator().manual_seed(image_seed(cfg.seed, image_id))
        eps = torch.randn(x0.shape, generator=gen).to(cfg.device, dtype)
        # The same noise for all conditions, so only the change in the block differs.
        xt = ((1 - cfg.sigma0) * x0 + cfg.sigma0 * eps).float().expand(n_c, -1, -1, -1).clone()
        delta = torch.tensor(shifts[i], device=cfg.device, dtype=torch.float32)
        state["delta"] = spread_cells(delta, cfg.grid, side)
        for s_now, s_next in zip(sigmas[:-1], sigmas[1:], strict=True):
            t = torch.full((n_c,), s_now * 1000.0, device=cfg.device)
            v = model(
                hidden_states=xt.to(dtype),
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask,
                timestep=t,
                return_dict=False,
            )[0]
            xt = xt + (s_next - s_now) * v.float()
        images = vae.decode((xt / vae.config.scaling_factor).to(dtype)).sample
        images = ((images.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for c, name in enumerate(conditions):
            out = cfg.out_dir / name.replace(":", "_") / shape
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[c]).resize((512, 512), Image.BICUBIC).save(out / f"{image_id}.png")
            rows.append({"image_id": image_id, "shape": shape, "condition": name})
        if i < 3 or (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(image_ids)}] {image_id} after {time.time() - t0:.0f} s", flush=True)
    handle.remove()
    meta = {k: str(v) for k, v in cfg.__dict__.items()}
    meta["conditions"] = conditions
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (cfg.out_dir / "images.csv").write_text(
        "image_id,shape,condition\n"
        + "".join(f"{r['image_id']},{r['shape']},{r['condition']}\n" for r in rows)
    )
    return cfg.out_dir
