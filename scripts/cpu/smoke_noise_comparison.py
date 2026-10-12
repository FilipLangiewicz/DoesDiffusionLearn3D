"""CPU smoke test: SANA-1.6B with the ZZSN noise rule and with the flow rule.

For each image and timestep the script runs the transformer twice (old and new
noise). It records the mean token norm of blocks 4, 10 and 18 and the error of
the one-step clean latent estimate ``x0_hat = x_t - s * v``. A wrong input
noise level makes this estimate bad. Optionally it decodes ``x0_hat`` to images.

This test checks the code path and the noise rule. It is too small for probes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from dd3d.extract import MODEL_ID, pool_tokens, preprocess
from dd3d.noise import flow_noise, image_seed, noise_coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, default=Path("data/features/prompt_empty.pt"))
    parser.add_argument("--out", type=Path, default=Path("reports/smoke/noise_comparison"))
    parser.add_argument("--timesteps", nargs="+", type=int, default=[200, 500, 800])
    parser.add_argument("--blocks", nargs="+", type=int, default=[4, 10, 18])
    parser.add_argument("--decode", action="store_true", help="Decode x0_hat at the first timestep to PNG.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    t0 = time.time()

    from diffusers import AutoencoderDC, DPMSolverMultistepScheduler, SanaTransformer2DModel

    cache = torch.load(args.prompt_cache)
    embeds, mask = cache["embeds"].to(dtype), cache["mask"]

    vae = AutoencoderDC.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
    scale = vae.config.scaling_factor
    latents = {p.stem: vae.encode(preprocess(p, 1024).to(dtype)).latent * scale for p in args.images}
    del vae
    gc.collect()
    print(f"latents ready after {time.time() - t0:.0f} s", flush=True)

    model = SanaTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=dtype, variant="bf16", low_cpu_mem_usage=True
    ).eval()
    print(f"transformer loaded after {time.time() - t0:.0f} s", flush=True)
    sched = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    captured: dict[int, torch.Tensor] = {}
    for b in args.blocks:
        model.transformer_blocks[b].register_forward_hook(
            lambda _m, _i, out, b=b: captured.__setitem__(b, out[0] if isinstance(out, tuple) else out)
        )

    rows, estimates = [], {}
    for stem, x0 in latents.items():
        gen = torch.Generator().manual_seed(image_seed(1234, stem))
        eps = torch.randn(x0.shape, generator=gen).to(dtype)
        for t in args.timesteps:
            s = t / 1000.0
            for rule in ("zzsn", "flow"):
                xt = sched.add_noise(x0, eps, torch.tensor([t])) if rule == "zzsn" else flow_noise(x0, eps, t)
                a, bcoef = noise_coefficients(xt.float(), x0.float(), eps.float())
                t1 = time.time()
                v = model(
                    hidden_states=xt.to(dtype),
                    encoder_hidden_states=embeds,
                    encoder_attention_mask=mask,
                    timestep=torch.tensor([t]),
                    return_dict=False,
                )[0]
                x0_hat = xt.float() - s * v.float()
                rel_err = float((x0_hat - x0.float()).norm() / x0.float().norm())
                row = {
                    "image": stem,
                    "t": t,
                    "rule": rule,
                    "coef_x0": round(a, 3),
                    "coef_eps": round(bcoef, 3),
                    "x0_hat_rel_error": round(rel_err, 4),
                    "seconds": round(time.time() - t1, 1),
                }
                for b in args.blocks:
                    h = captured[b].float()
                    row[f"norm_block{b}"] = round(float(h.flatten(0, -2).norm(dim=-1).mean()), 1)
                    np.save(args.out / f"{stem}_{rule}_t{t}_b{b:02d}.npy", pool_tokens(h, 8))
                rows.append(row)
                if t == args.timesteps[0]:
                    estimates[(stem, rule)] = x0_hat
                print(json.dumps(row), flush=True)

    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    del model
    gc.collect()

    if args.decode:
        from PIL import Image

        vae = AutoencoderDC.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
        for (stem, rule), x0_hat in estimates.items():
            img = vae.decode((x0_hat / scale).to(dtype)).sample[0].float().clamp(-1, 1)
            arr = ((img.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            Image.fromarray(arr).resize((512, 512)).save(args.out / f"{stem}_{rule}_t{args.timesteps[0]}_x0hat.png")
        print("decoded", flush=True)
    print(f"done after {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
