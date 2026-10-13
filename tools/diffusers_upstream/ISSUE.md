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

# Reproduction

The real workflow: SanaPipeline, an image through the VAE, `pipe.scheduler.add_noise`, then a one-step clean estimate from the transformer. The script compares this with the expected flow noise `(1 - t/1000) * x0 + (t/1000) * eps`.

```python
import torch
from diffusers import SanaPipeline
from diffusers.utils import load_image

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.bfloat16
).to("cuda")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = pipe.image_processor.preprocess(load_image(url), height=1024, width=1024).to("cuda", torch.bfloat16)

with torch.no_grad():
    x0 = pipe.vae.encode(image).latent * pipe.vae.config.scaling_factor
    embeds, mask, _, _ = pipe.encode_prompt("", do_classifier_free_guidance=False, device="cuda")
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)).to("cuda", torch.bfloat16)
    for t in (200, 500, 800):
        timestep = torch.tensor([t], device="cuda")
        for name, x_t in [
            ("add_noise", pipe.scheduler.add_noise(x0, eps, timestep)),
            ("expected", (1 - t / 1000) * x0 + t / 1000 * eps),
        ]:
            v = pipe.transformer(
                x_t, encoder_hidden_states=embeds, encoder_attention_mask=mask, timestep=timestep, return_dict=False
            )[0]
            x0_hat = x_t.float() - t / 1000 * v.float()
            print(t, name, ((x0_hat - x0.float()).norm() / x0.float().norm()).item())
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

Relative error of the one-step clean estimate (lower is better). Real image, real model, float32 on CPU, `Efficient-Large-Model/Sana_600M_512px_diffusers` (same scheduler config as the 1.6B model):

| t | `add_noise` | expected flow noise |
|---:|---:|---:|
| 200 | 45.5 | 0.228 |
| 500 | 5.76 | 0.383 |
| 800 | 0.556 | 0.590 |

<!-- Add the SANA-1.6B numbers from the GPU run of the script above. -->

Weights of `x_t = a * x0 + b * eps` from `add_noise` with the SANA scheduler:

| t | `a`, `b` from `add_noise` | expected |
|---:|---|---|
| 200 | -24.53, 25.53 | 0.8, 0.2 |
| 500 | -2.42, 3.42 | 0.5, 0.5 |
| 800 | 0.28, 0.72 | 0.2, 0.8 |

Default VP config, `t = 200`: `DPMSolverMultistepScheduler` gives `a = 0.0391`. `DDPMScheduler` gives `0.8102` at `t = 200` and `0.0391` at `t = 799`.

With a trained VP model (`google/ddpm-cifar10-32`), the one-step error at `t = 100` is 0.590 with the four schedulers and 0.045 with `DDPMScheduler`.

# System Info

<!-- Paste the output of `diffusers-cli env`. -->
Also checked on `main` at commit 916a579 (2026-10-13). The code is the same as in 0.41.0.

# Who can help?

@yiyixuxu @dg845
