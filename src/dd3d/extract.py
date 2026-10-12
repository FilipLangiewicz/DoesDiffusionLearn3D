"""Extract pooled SANA-1.6B activations for rendered images.

Changes against the ZZSN script src/extract_activations.py:

- The noise follows the flow rule ``(1 - t/1000) * x0 + (t/1000) * eps`` (see dd3d.noise).
- Each image has a fixed noise seed, so a second run gives the same features.
- The transformer gets the attention mask of the prompt, as in SanaPipeline.
- The script averages the 32 x 32 token grid to a small grid and stores float16.
  This is 287 KB for each image, block and timestep instead of 9.2 MB.
- The model parts load one after the other, so a CPU smoke test needs less RAM.

Output for each block ``b`` and timestep ``t``:
``<out>/block_<bb>/t_<ttt>/chunk_<task>.npy`` (images x features) and
``chunk_<task>_ids.txt``. The command ``dd3d merge`` joins the chunks.
"""

from __future__ import annotations

import csv
import gc
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dd3d.noise import flow_noise, image_seed

MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
SHAPES = ("Sphere", "SmoothCylinder", "SmoothCube_v2")


@dataclass
class ExtractConfig:
    images_dir: Path
    out_dir: Path
    blocks: list[int] = field(default_factory=lambda: [4, 10, 18])
    timesteps: list[int] = field(default_factory=lambda: [200, 500, 800])
    grid: int = 8
    image_size: int = 1024
    seed: int = 1234
    max_per_shape: int = 1000
    task_id: int = 0
    num_tasks: int = 1
    limit: int | None = None
    device: str = "cuda"
    model_id: str = MODEL_ID
    transformer_variant: str | None = "bf16"
    prompt_cache: Path | None = None
    legacy_noise: bool = False
    random_init: bool = False
    save_vae_latents: bool = True
    full_blocks: list[int] = field(default_factory=list)
    full_timesteps: list[int] = field(default_factory=list)


def list_images(images_dir: Path, max_per_shape: int) -> list[Path]:
    """Return the image paths in a fixed order (shape, then file name)."""
    paths: list[Path] = []
    for shape in SHAPES:
        paths += sorted((images_dir / shape).glob("*.png"))[:max_per_shape]
    return paths


def task_slice(paths: list[Path], task_id: int, num_tasks: int) -> list[Path]:
    """Return the contiguous part of ``paths`` for one array task."""
    size = math.ceil(len(paths) / num_tasks)
    return paths[task_id * size : (task_id + 1) * size]


def pool_tokens(hidden: torch.Tensor, grid: int) -> np.ndarray:
    """Average a (1, N, C) token map with N = side * side to (grid * grid * C,) float16."""
    if hidden.dim() == 4:  # (B, C, H, W) layout of some diffusers versions
        hidden = hidden.flatten(2).transpose(1, 2)
    _, n, c = hidden.shape
    side = int(round(math.sqrt(n)))
    x = hidden[0].float().reshape(side, side, c)
    k = side // grid
    x = x.reshape(grid, k, grid, k, c).mean(dim=(1, 3))
    return x.reshape(-1).cpu().numpy().astype(np.float16)


def preprocess(path: Path, size: int) -> torch.Tensor:
    """Load an RGB image and scale it to [-1, 1], like the ZZSN script."""
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def encode_prompt(cfg: ExtractConfig, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the embeddings and the mask of the empty prompt (cached on disk)."""
    if cfg.prompt_cache and Path(cfg.prompt_cache).exists():
        cached = torch.load(cfg.prompt_cache)
        return cached["embeds"].to(dtype), cached["mask"]
    from diffusers import SanaPipeline

    # Only the text encoder is necessary here; the large parts load later.
    pipe = SanaPipeline.from_pretrained(cfg.model_id, transformer=None, vae=None, torch_dtype=dtype)
    pipe.text_encoder.to(cfg.device)
    with torch.no_grad():
        embeds, mask, _, _ = pipe.encode_prompt("", do_classifier_free_guidance=False, device=cfg.device)
    embeds, mask = embeds.cpu(), mask.cpu()
    if cfg.prompt_cache:
        Path(cfg.prompt_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeds": embeds, "mask": mask}, cfg.prompt_cache)
    del pipe
    gc.collect()
    return embeds.to(dtype), mask


def encode_latents(cfg: ExtractConfig, paths: list[Path], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Return the scaled DC-AE latents of the images."""
    from diffusers import AutoencoderDC

    vae = AutoencoderDC.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=dtype).to(cfg.device)
    latents = {}
    with torch.no_grad():
        for p in paths:
            x = preprocess(p, cfg.image_size).to(cfg.device, dtype=dtype)
            latents[p.stem] = (vae.encode(x).latent * vae.config.scaling_factor).cpu()
    del vae
    gc.collect()
    return latents


def load_transformer(cfg: ExtractConfig, dtype: torch.dtype):
    from diffusers import SanaTransformer2DModel

    if cfg.random_init:
        config = SanaTransformer2DModel.load_config(cfg.model_id, subfolder="transformer")
        torch.manual_seed(cfg.seed)
        model = SanaTransformer2DModel.from_config(config).to(dtype)
    else:
        kwargs = {"variant": cfg.transformer_variant} if cfg.transformer_variant else {}
        model = SanaTransformer2DModel.from_pretrained(
            cfg.model_id, subfolder="transformer", torch_dtype=dtype, **kwargs
        )
    return model.to(cfg.device).eval()


def run(cfg: ExtractConfig) -> Path:
    """Extract the features of one task and return the output folder."""
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    paths = task_slice(list_images(cfg.images_dir, cfg.max_per_shape), cfg.task_id, cfg.num_tasks)
    if cfg.limit is not None:
        paths = paths[: cfg.limit]
    if not paths:
        print(f"task {cfg.task_id}: no images")
        return cfg.out_dir

    t0 = time.time()
    embeds, mask = encode_prompt(cfg, dtype)
    latents = encode_latents(cfg, paths, dtype)
    print(f"prompt and latents ready in {time.time() - t0:.0f} s")

    if cfg.save_vae_latents:
        vae_dir = cfg.out_dir / "vae_latent"
        vae_dir.mkdir(parents=True, exist_ok=True)
        arr = np.stack([latents[p.stem][0].float().numpy().astype(np.float16).ravel() for p in paths])
        np.save(vae_dir / f"chunk_{cfg.task_id:04d}.npy", arr)
        (vae_dir / f"chunk_{cfg.task_id:04d}_ids.txt").write_text("\n".join(p.stem for p in paths) + "\n")

    model = load_transformer(cfg, dtype)
    scheduler = None
    if cfg.legacy_noise:
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for b in cfg.blocks:

        def hook(_module, _inputs, output, b=b):
            captured[b] = (output[0] if isinstance(output, tuple) else output).detach()

        handles.append(model.transformer_blocks[b].register_forward_hook(hook))

    feats = {(b, t): [] for b in cfg.blocks for t in cfg.timesteps}
    stats_rows = []
    embeds_d, mask_d = embeds.to(cfg.device), mask.to(cfg.device)
    for i, p in enumerate(paths):
        x0 = latents[p.stem].to(cfg.device, dtype=dtype)
        gen = torch.Generator().manual_seed(image_seed(cfg.seed, p.stem))
        eps = torch.randn(x0.shape, generator=gen).to(cfg.device, dtype=dtype)
        for t in cfg.timesteps:
            if cfg.legacy_noise:
                xt = scheduler.add_noise(x0, eps, torch.tensor([t]))
            else:
                xt = flow_noise(x0, eps, t)
            model(
                hidden_states=xt.to(dtype),
                encoder_hidden_states=embeds_d,
                encoder_attention_mask=mask_d,
                timestep=torch.tensor([t], device=cfg.device),
                return_dict=False,
            )
            for b in cfg.blocks:
                hidden = captured[b]
                feats[(b, t)].append(pool_tokens(hidden, cfg.grid))
                norm = hidden.float().flatten(0, -2).norm(dim=-1).mean().item()
                stats_rows.append({"image_id": p.stem, "block": b, "t": t, "mean_token_norm": norm})
                if b in cfg.full_blocks and t in cfg.full_timesteps:
                    full_dir = cfg.out_dir / "full" / f"block_{b:02d}" / f"t_{t:03d}"
                    full_dir.mkdir(parents=True, exist_ok=True)
                    np.save(full_dir / f"{p.stem}.npy", hidden[0].float().cpu().numpy().astype(np.float16))
        print(f"[{i + 1}/{len(paths)}] {p.stem} done after {time.time() - t0:.0f} s", flush=True)

    for h in handles:
        h.remove()
    ids = "\n".join(p.stem for p in paths) + "\n"
    for (b, t), rows in feats.items():
        d = cfg.out_dir / f"block_{b:02d}" / f"t_{t:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"chunk_{cfg.task_id:04d}.npy", np.stack(rows))
        (d / f"chunk_{cfg.task_id:04d}_ids.txt").write_text(ids)
    stats_dir = cfg.out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / f"chunk_{cfg.task_id:04d}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "block", "t", "mean_token_norm"])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"task {cfg.task_id}: {len(paths)} images in {time.time() - t0:.0f} s -> {cfg.out_dir}")
    return cfg.out_dir


def merge(feature_root: Path) -> list[Path]:
    """Join the chunk files of every block and timestep folder into one array."""
    merged = []
    for d in sorted({p.parent for p in feature_root.rglob("chunk_*.npy")}):
        chunks = sorted(d.glob("chunk_*.npy"))
        arrays = [np.load(c) for c in chunks]
        ids = []
        for c in chunks:
            ids += (c.parent / f"{c.stem}_ids.txt").read_text().split()
        np.save(d / "features.npy", np.concatenate(arrays))
        (d / "ids.txt").write_text("\n".join(ids) + "\n")
        merged.append(d)
    return merged


def env_task_id(default: int = 0) -> int:
    """Return the SLURM array task id, if the job is an array job."""
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", default))
