"""Extract pooled features of two vision encoders that do not generate images.

These models are a control for the claim that the light code is a property of
diffusion models. DINOv2 learns from images alone, and CLIP learns from images
and text. Neither model renders an image. Assume that their features give the
same probe result as SANA. Then the result shows a property of all trained
vision networks, and not of generative models.

The rules follow dd3d.extract_unet:

- The image goes to the encoder without noise, so the output layout has one
  timestep, t = 0.
- The image is scaled to 224 x 224 px without a crop, because a crop can cut
  an object at the edge of the image. The encoder then has 16 x 16 patch tokens.
- The class token is not used. The patch tokens are averaged on a grid.
- A block is the output of one transformer layer (block 0 is the patch embedding).

The output layout is the same as in dd3d.extract, and meta.json gives the grid
of each block.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dd3d.extract import list_images, task_slice

MODELS = {
    "dinov2": {"id": "facebook/dinov2-large", "kind": "dinov2"},
    "clip": {"id": "openai/clip-vit-large-patch14", "kind": "clip"},
}


@dataclass
class VisionExtractConfig:
    images_dir: Path
    out_dir: Path
    model: str = "dinov2"
    blocks: list[int] = field(default_factory=lambda: list(range(0, 25, 2)))
    grid: int = 4
    fine_blocks: list[int] = field(default_factory=lambda: [8, 16, 24])
    fine_grid: int = 8
    image_size: int = 224
    batch_size: int = 32
    max_per_shape: int = 1000
    task_id: int = 0
    num_tasks: int = 1
    limit: int | None = None
    device: str = "cuda"


def pool_patch_tokens(hidden: torch.Tensor, grid: int) -> np.ndarray:
    """Average (B, 1 + side * side, C) tokens without the class token to (B, grid * grid * C) float16.

    The order of the result is (row, column, channel), as in dd3d.extract.pool_tokens.
    """
    patches = hidden[:, 1:].float()
    b, n, c = patches.shape
    side = int(round(n**0.5))
    x = patches.reshape(b, side, side, c).permute(0, 3, 1, 2)
    x = torch.nn.functional.adaptive_avg_pool2d(x, grid)
    return x.permute(0, 2, 3, 1).reshape(b, -1).cpu().numpy().astype(np.float16)


def load_batch(paths: list[Path], size: int, mean, std) -> torch.Tensor:
    """Load RGB images, scale them to size x size and normalize them with the encoder statistics."""
    arr = [
        np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
        for p in paths
    ]
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]


def load_encoder(spec: dict, device: str):
    """Return the vision encoder and its normalization (mean, std)."""
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

    proc = AutoImageProcessor.from_pretrained(spec["id"])
    if spec["kind"] == "clip":
        model = CLIPVisionModel.from_pretrained(spec["id"], torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(spec["id"], torch_dtype=torch.float16)
    return model.to(device).eval(), (list(proc.image_mean), list(proc.image_std))


def run(cfg: VisionExtractConfig) -> Path:
    """Extract the features of one task and return the output folder."""
    spec = MODELS[cfg.model]
    torch.set_grad_enabled(False)
    paths = task_slice(list_images(cfg.images_dir, cfg.max_per_shape), cfg.task_id, cfg.num_tasks)
    if cfg.limit is not None:
        paths = paths[: cfg.limit]
    t0 = time.time()
    model, (mean, std) = load_encoder(spec, cfg.device)
    grids = {b: (cfg.fine_grid if b in cfg.fine_blocks else cfg.grid) for b in cfg.blocks}

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    meta["model_id"] = spec["id"]
    meta["timesteps"] = [0]
    meta["grid_of_block"] = {str(b): g for b, g in grids.items()}
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    feats = {b: [] for b in cfg.blocks}
    for start in range(0, len(paths), cfg.batch_size):
        batch = paths[start : start + cfg.batch_size]
        x = load_batch(batch, cfg.image_size, mean, std).to(cfg.device, torch.float16)
        hidden = model(pixel_values=x, output_hidden_states=True).hidden_states
        for b in cfg.blocks:
            feats[b].append(pool_patch_tokens(hidden[b], grids[b]))
        print(f"[{start + len(batch)}/{len(paths)}] after {time.time() - t0:.0f} s", flush=True)

    ids = "\n".join(p.stem for p in paths) + "\n"
    for b, rows in feats.items():
        d = cfg.out_dir / f"block_{b:02d}" / "t_000"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"chunk_{cfg.task_id:04d}.npy", np.concatenate(rows))
        (d / f"chunk_{cfg.task_id:04d}_ids.txt").write_text(ids)
    print(f"task {cfg.task_id}: {len(paths)} images in {time.time() - t0:.0f} s -> {cfg.out_dir}")
    return cfg.out_dir
