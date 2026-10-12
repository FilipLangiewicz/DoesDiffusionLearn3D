"""Compute the empty-prompt embeddings of SANA-1.6B on a CPU with little RAM.

The Gemma-2 text encoder needs about 5 GB in bfloat16. This script loads it with
disk offload (accelerate), so a machine with 4 GB of free RAM can run it. The
result is the file that ``prompt_cache`` in configs/extract*.yaml points to.
On a GPU node, ``dd3d extract`` makes the same file without this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dd3d.extract import MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/features/prompt_empty.pt"))
    parser.add_argument("--offload-dir", type=Path, default=Path("data/features/offload_text_encoder"))
    parser.add_argument("--max-cpu-memory", default="1500MiB")
    args = parser.parse_args()

    from diffusers import SanaPipeline
    from transformers import AutoModel

    t0 = time.time()
    encoder = AutoModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={"cpu": args.max_cpu_memory},
        offload_folder=str(args.offload_dir),
    )
    pipe = SanaPipeline.from_pretrained(
        MODEL_ID, text_encoder=encoder, transformer=None, vae=None, torch_dtype=torch.bfloat16
    )
    with torch.no_grad():
        embeds, mask, _, _ = pipe.encode_prompt("", do_classifier_free_guidance=False, device="cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": embeds.cpu(), "mask": mask.cpu()}, args.out)
    print(
        f"embeds {tuple(embeds.shape)} mask sum {int(mask.sum())} -> {args.out} in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
