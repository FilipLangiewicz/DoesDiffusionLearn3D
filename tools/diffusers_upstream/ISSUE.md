# Title

`add_noise()` before `set_timesteps()` uses the wrong noise level (DPMSolverMultistep, DPMSolverSinglestep, UniPC, DEIS; also SanaPipeline)

# Describe the bug

If you call `scheduler.add_noise(x0, noise, t)` before `set_timesteps()`, these schedulers use the noise level of timestep `999 - t`:

- `DPMSolverMultistepScheduler`
- `DPMSolverSinglestepScheduler`
- `UniPCMultistepScheduler`
- `DEISMultistepScheduler`

Fine-tune scripts and probe studies use this call order, and #5981 / #6006 made it a supported call order.

The cause: after `__init__`, `self.timesteps` goes from 999 down to 0, but `self.sigmas` (added in #6006) goes up from step 0 to step 999. `add_noise` finds the index of `t` in `self.timesteps` and reads `self.sigmas` at the same index.

With `use_flow_sigmas=True` (the scheduler of SanaPipeline) the result is worse. The code uses `alpha_t = 1 - sigma` with a VP sigma, so for `t = 200` you get `-24.5 * x0 + 25.5 * eps` instead of `0.8 * x0 + 0.2 * eps`.

After `set_timesteps()` everything is correct.

I found this in a probe study on SANA-1.6B. I can open a pull request with a fix and tests.

## Results

SANA-1.6B with the script below, on a Kaggle T4 (16 GB), float32, with the text encoder on the CPU. `a` and `b` are the weights of `x_t = a * x0 + b * eps` from `add_noise`. The error is the relative error of the one-step clean estimate (lower is better).

| t | `a`, `b` (0.41.0) | error, 0.41.0 | error, with my fix | error, expected flow noise |
|---:|---|---:|---:|---:|
| 200 | -24.53, 25.53 | 36.618 | 0.180 | 0.180 |
| 500 | -2.42, 3.42 | 4.615 | 0.379 | 0.379 |
| 800 | 0.28, 0.72 | 0.601 | 0.624 | 0.624 |

With my fix, `add_noise` gives `a, b = 0.8, 0.2 / 0.5, 0.5 / 0.2, 0.8`.

The default VP config has the same problem: at `t = 200`, `DPMSolverMultistepScheduler` gives `a = 0.0391`. `DDPMScheduler` gives `0.8102` at `t = 200` and `0.0391` at `t = 799`. With a trained VP model (`google/ddpm-cifar10-32`), the one-step error at `t = 100` is 0.590 with the four schedulers and 0.045 with `DDPMScheduler`.

# Reproduction

The real workflow: SanaPipeline, an image through the VAE, `pipe.scheduler.add_noise`, then a one-step clean estimate from the transformer. The script compares this with the expected flow noise `(1 - t/1000) * x0 + (t/1000) * eps`. The output in "Logs" comes from `python repro_sana.py --dtype float32 --text-encoder-device cpu` on a 16 GB GPU (with bf16 on a larger GPU, the defaults work).

```python
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
    "--text-encoder-device", default=None, help="Run the text encoder here, for example cpu on a 16 GB GPU."
)
parser.add_argument(
    "--image",
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)
args = parser.parse_args()
dtype = getattr(torch, args.dtype)

pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype)
te_device = args.text_encoder_device or args.device
pipe.text_encoder.to(te_device)
with torch.no_grad():
    embeds, mask, _, _ = pipe.encode_prompt("", do_classifier_free_guidance=False, device=te_device)
embeds, mask = embeds.to(args.device), mask.to(args.device)
pipe.text_encoder = None
pipe.transformer.to(args.device)
pipe.vae.to(args.device)
image = pipe.image_processor.preprocess(load_image(args.image), height=args.size, width=args.size)
with torch.no_grad():
    x0 = pipe.vae.encode(image.to(args.device, dtype)).latent * pipe.vae.config.scaling_factor
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
                x_t,
                encoder_hidden_states=embeds,
                encoder_attention_mask=mask,
                timestep=timestep,
                return_dict=False,
            )[0]
            x0_hat = x_t.float() - sigma * v.float()
            errors.append(((x0_hat - x0.float()).norm() / x0.float().norm()).item())
        print(f"{t:>4} | {a:8.3f} {b:8.3f} | {errors[0]:20.3f} | {errors[1]:21.3f}")
```

The same problem without model weights (only the scheduler config):

```python
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

sana = DPMSolverMultistepScheduler.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="scheduler"
)
x0, eps = torch.ones(1), torch.zeros(1)
for t in (200, 500, 800):
    t = torch.tensor([t])
    print("sana", t.item(), sana.add_noise(x0, eps, t).item(), sana.add_noise(eps, x0, t).item())

# The default VP config gives the noise level of 999 - t:
t = torch.tensor([200])
print("DDPM t=200", DDPMScheduler().add_noise(x0, eps, t).item())
print("DDPM t=799", DDPMScheduler().add_noise(x0, eps, torch.tensor([799])).item())
print("DPMSolver t=200", DPMSolverMultistepScheduler().add_noise(x0, eps, t).item())
```

# Logs

===== diffusers 0.41.0 (PyPI) =====
$ python repro_sana.py --dtype float32 --text-encoder-device cpu
   t |   a (x0)  b (eps) | error with add_noise | error with flow noise
 200 |  -24.528   25.528 |               36.618 |                 0.180
 500 |   -2.424    3.424 |                4.615 |                 0.379
 800 |    0.281    0.719 |                0.601 |                 0.624

===== diffusers main (916a579) with my fix =====
$ PYTHONPATH=dsrc/src python repro_sana.py --dtype float32 --text-encoder-device cpu
   t |   a (x0)  b (eps) | error with add_noise | error with flow noise
 200 |    0.800    0.200 |                0.180 |                 0.180
 500 |    0.500    0.500 |                0.379 |                 0.379
 800 |    0.200    0.800 |                0.624 |                 0.624

# System Info

Also checked on `main` at commit 916a579 (2026-10-13). The code is the same as in 0.41.0.

# Who can help?

@yiyixuxu @dg845
