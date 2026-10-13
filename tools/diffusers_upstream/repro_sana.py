"""Reproduction: SanaPipeline scheduler add_noise() before set_timesteps().

The script encodes one image with the SANA VAE. It noises the latent with
pipe.scheduler.add_noise and does not call set_timesteps, as a fine-tune or probe script does.
Then the transformer gives a one-step clean estimate. The script repeats this with the
expected flow noise x_t = (1 - t/1000) * x0 + (t/1000) * eps.

Usage: python repro_sana.py [--model ID] [--device cuda] [--dtype bfloat16] [--image PATH]
"""

import argparse

import torch
from diffusers import SanaPipeline
from diffusers.utils import load_image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
parser.add_argument("--size", type=int, default=1024)
parser.add_argument(
    "--image",
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)
args = parser.parse_args()
dtype = getattr(torch, args.dtype)

pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
image = pipe.image_processor.preprocess(load_image(args.image), height=args.size, width=args.size)
with torch.no_grad():
    x0 = pipe.vae.encode(image.to(args.device, dtype)).latent * pipe.vae.config.scaling_factor
    embeds, mask, _, _ = pipe.encode_prompt("", do_classifier_free_guidance=False, device=args.device)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)).to(args.device, dtype)

    print(f"{'t':>4} | {'a (x0)':>8} {'b (eps)':>8} | error with add_noise | error with flow noise")
    for t in (200, 500, 800):
        timestep = torch.tensor([t], device=args.device)
        sigma = t / 1000
        expected = (1 - sigma) * x0 + sigma * eps
        noised = pipe.scheduler.add_noise(x0, eps, timestep)
        # Least squares fit noised = a * x0 + b * eps shows the noise level that add_noise used.
        design = torch.stack([x0.flatten(), eps.flatten()], 1).float()
        a, b = torch.linalg.lstsq(design, noised.flatten().float()[:, None]).solution.flatten().tolist()
        errors = []
        for x_t in (noised, expected):
            v = pipe.transformer(
                x_t, encoder_hidden_states=embeds, encoder_attention_mask=mask, timestep=timestep, return_dict=False
            )[0]
            x0_hat = x_t.float() - sigma * v.float()
            errors.append(((x0_hat - x0.float()).norm() / x0.float().norm()).item())
        print(f"{t:>4} | {a:8.3f} {b:8.3f} | {errors[0]:20.3f} | {errors[1]:21.3f}")
