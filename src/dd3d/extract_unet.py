"""Extract pooled UNet activations of Stable Diffusion 1.5 and SDXL.

These models are a check of generality: the same probe and controls run on two
more diffusion models with a UNet. The rules follow dd3d.extract:

- The noise is the DDPM rule of the model, from DDPMScheduler.add_noise. This
  scheduler reads alphas_cumprod[t], so it is correct without set_timesteps.
- Each image has a fixed noise seed. The prompt is empty.
- The VAE runs in float32, because the SDXL VAE gives NaN values in float16.
- The UNet runs in float16 and gets all timesteps of one image as one batch.

The blocks are the outputs of down_blocks, mid_block and up_blocks, in this
order. The output layout is the same as in dd3d.extract, and meta.json gives
the name and the grid of each block.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from dd3d.extract import list_images, preprocess, task_slice
from dd3d.noise import image_seed

MODELS = {
    "sd15": {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "size": 512, "xl": False},
    "sdxl": {"id": "stabilityai/stable-diffusion-xl-base-1.0", "size": 1024, "xl": True},
}


@dataclass
class UNetExtractConfig:
    images_dir: Path
    out_dir: Path
    model: str = "sd15"
    timesteps: list[int] = field(default_factory=lambda: [100, 200, 400, 600, 800])
    grid: int = 4
    fine_blocks: list[str] = field(default_factory=lambda: ["mid", "up1"])
    fine_grid: int = 8
    seed: int = 1234
    max_per_shape: int = 1000
    task_id: int = 0
    num_tasks: int = 1
    limit: int | None = None
    device: str = "cuda"


def pool_map(out: torch.Tensor, grid: int) -> np.ndarray:
    """Average a (B, C, H, W) map to (B, grid * grid * C) in (row, column, channel) order."""
    x = torch.nn.functional.adaptive_avg_pool2d(out.float(), grid)
    return x.permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy().astype(np.float16)


def block_modules(unet) -> list[tuple[str, torch.nn.Module]]:
    blocks = [(f"down{i}", b) for i, b in enumerate(unet.down_blocks)]
    blocks.append(("mid", unet.mid_block))
    blocks += [(f"up{i}", b) for i, b in enumerate(unet.up_blocks)]
    return blocks


def run(cfg: UNetExtractConfig) -> Path:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    spec = MODELS[cfg.model]
    torch.set_grad_enabled(False)
    paths = task_slice(list_images(cfg.images_dir, cfg.max_per_shape), cfg.task_id, cfg.num_tasks)
    if cfg.limit is not None:
        paths = paths[: cfg.limit]
    t0 = time.time()

    # Empty prompt: the text encoders run once and leave the GPU.
    if spec["xl"]:
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            spec["id"], unet=None, vae=None, torch_dtype=torch.float16, variant="fp16"
        ).to(cfg.device)
        embeds, _, pooled, _ = pipe.encode_prompt("", device=cfg.device, do_classifier_free_guidance=False)
        size = spec["size"]
        time_ids = torch.tensor([[size, size, 0, 0, size, size]], device=cfg.device, dtype=torch.float16)
        extra = {"text_embeds": pooled, "time_ids": time_ids}
    else:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            spec["id"], unet=None, vae=None, safety_checker=None, torch_dtype=torch.float16
        ).to(cfg.device)
        embeds, _ = pipe.encode_prompt("", cfg.device, 1, False)
        extra = None
    del pipe
    gc.collect()

    vae = AutoencoderKL.from_pretrained(spec["id"], subfolder="vae", torch_dtype=torch.float32).to(cfg.device)
    latents = {}
    for p in paths:
        x = preprocess(p, spec["size"]).to(cfg.device)
        latents[p.stem] = (vae.encode(x).latent_dist.mean * vae.config.scaling_factor).cpu()
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"prompt and latents ready in {time.time() - t0:.0f} s", flush=True)

    unet = UNet2DConditionModel.from_pretrained(
        spec["id"], subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    ).to(cfg.device)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(spec["id"], subfolder="scheduler")
    blocks = block_modules(unet)
    grids = {name: (cfg.fine_grid if name in cfg.fine_blocks else cfg.grid) for name, _ in blocks}
    captured: dict[str, torch.Tensor] = {}
    for name, module in blocks:

        def hook(_m, _i, output, name=name):
            captured[name] = output[0] if isinstance(output, tuple) else output

        module.register_forward_hook(hook)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    meta["model_id"] = spec["id"]
    meta["blocks"] = list(range(len(blocks)))
    meta["block_names"] = [name for name, _ in blocks]
    meta["grid_of_block"] = {str(i): grids[name] for i, (name, _) in enumerate(blocks)}
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    n_t = len(cfg.timesteps)
    t_batch = torch.tensor(cfg.timesteps, device=cfg.device)
    enc = embeds.expand(n_t, *embeds.shape[1:])
    cond = None
    if extra is not None:
        cond = {k: v.expand(n_t, *v.shape[1:]) for k, v in extra.items()}
    feats = {(i, t): [] for i in range(len(blocks)) for t in cfg.timesteps}
    for k, p in enumerate(paths):
        x0 = latents[p.stem].to(cfg.device)
        gen = torch.Generator().manual_seed(image_seed(cfg.seed, p.stem))
        eps = torch.randn(x0.shape, generator=gen).to(cfg.device)
        xt = scheduler.add_noise(x0.expand(n_t, -1, -1, -1), eps.expand(n_t, -1, -1, -1), t_batch.cpu())
        unet(xt.half(), t_batch, encoder_hidden_states=enc, added_cond_kwargs=cond, return_dict=False)
        for i, (name, _) in enumerate(blocks):
            pooled = pool_map(captured[name], grids[name])
            for j, t in enumerate(cfg.timesteps):
                feats[(i, t)].append(pooled[j])
        if k < 3 or (k + 1) % 25 == 0:
            print(f"[{k + 1}/{len(paths)}] {p.stem} done after {time.time() - t0:.0f} s", flush=True)

    ids = "\n".join(p.stem for p in paths) + "\n"
    for (i, t), rows in feats.items():
        d = cfg.out_dir / f"block_{i:02d}" / f"t_{t:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"chunk_{cfg.task_id:04d}.npy", np.stack(rows))
        (d / f"chunk_{cfg.task_id:04d}_ids.txt").write_text(ids)
    print(f"task {cfg.task_id}: {len(paths)} images in {time.time() - t0:.0f} s -> {cfg.out_dir}")
    return cfg.out_dir
