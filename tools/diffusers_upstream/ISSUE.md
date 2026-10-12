# Title

`add_noise` gives the wrong noise level if `set_timesteps` was not called (DPMSolverMultistep, DPMSolverSinglestep, UniPCMultistep, DEISMultistep)

## Description

Some users call `scheduler.add_noise(x0, noise, t)` before `set_timesteps()`. This is the normal case in training code. A maintainer wrote in the fix for #5981 that this call order is typical in training.

In this case, `add_noise` returns the noise level of a different timestep.

In `__init__`, these schedulers set the attributes in this way:

- `self.timesteps` is descending: `[999, 998, ..., 0]`.
- `self.sigmas` is ascending: the VP sigmas `sqrt((1 - alphas_cumprod) / alphas_cumprod)` for `t = 0 ... 999`.

`add_noise` finds the index of `t` in `self.timesteps`. This index is `999 - t`. Then it reads `self.sigmas[999 - t]`. This value is the sigma of timestep `999 - t`, not of `t`.

With `use_flow_sigmas=True` the result is worse. The code then uses `alpha_t = 1 - sigma` and `sigma_t = sigma`. These formulas are correct only for flow sigmas in [0, 1]. The VP sigmas are in [0.01, 157]. The result is a large negative weight for the clean sample.

After `set_timesteps()`, `self.sigmas` and `self.timesteps` have the same order. The result is correct in that case.

## Reproduction

The script needs no model weights. Only the scheduler config is used.

```python
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

def coeffs(scheduler, t):
    t = torch.tensor([t])
    a = scheduler.add_noise(torch.ones(1, 1), torch.zeros(1, 1), t).item()  # weight of x0
    b = scheduler.add_noise(torch.zeros(1, 1), torch.ones(1, 1), t).item()  # weight of noise
    return round(a, 4), round(b, 4)

# 1. Flow config of Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
flow = dict(
    prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0,
    beta_schedule="linear", beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000,
)
for t in (200, 500, 800):
    print("flow", t, coeffs(DPMSolverMultistepScheduler(**flow), t))

# 2. Default VP config, compared with DDPMScheduler (same betas)
for t in (200, 799):
    print("DDPM", t, coeffs(DDPMScheduler(), t))
print("DPMSolver t=200", coeffs(DPMSolverMultistepScheduler(), 200))
```

A longer script with all schedulers is `repro.py` (attached or in the linked pull request).

## Actual output (diffusers 0.41.0)

Flow config (`x_t = a * x0 + b * eps`):

| t   | actual `a`, `b`   | expected `a`, `b` |
|-----|-------------------|-------------------|
| 200 | -24.5285, 25.5285 | 0.8, 0.2          |
| 500 | -2.4241, 3.4241   | 0.5, 0.5          |
| 800 | 0.2807, 0.7193    | 0.2, 0.8          |

Default VP config, `t = 200`:

| scheduler                         | `a`    | `b`    |
|-----------------------------------|--------|--------|
| `DDPMScheduler` at t=200          | 0.8102 | 0.5862 |
| `DDPMScheduler` at t=799          | 0.0391 | 0.9992 |
| `DPMSolverMultistepScheduler`     | 0.0391 | 0.9992 |

The result of `DPMSolverMultistepScheduler` is the noise level of `t = 799 = 999 - 200`. So the bug is also in the VP case, not only with flow sigmas.

## Expected output

`add_noise(x0, noise, t)` gives the noise level of timestep `t`. The result is the same before and after `set_timesteps()`. For flow sigmas, the result is `(1 - t / 1000) * x0 + (t / 1000) * eps`. This is the convention of `set_timesteps`, where `timesteps = sigmas * num_train_timesteps`.

## Affected schedulers

Tested with the default VP config and with `use_flow_sigmas=True`, without `set_timesteps`:

| scheduler                              | VP      | flow sigmas |
|----------------------------------------|---------|-------------|
| `DPMSolverMultistepScheduler`          | wrong   | wrong       |
| `DPMSolverSinglestepScheduler`         | wrong   | wrong       |
| `UniPCMultistepScheduler`              | wrong   | wrong       |
| `DEISMultistepScheduler`               | wrong   | wrong       |
| `DPMSolverMultistepInverseScheduler`   | correct | wrong (its timesteps are ascending, but its sigmas are the VP sigmas) |
| `SASolverScheduler`                    | correct | wrong (its `add_noise` always uses `alphas_cumprod`, also after `set_timesteps`) |
| `EDMDPMSolverMultistepScheduler`       | not affected (the timesteps and sigmas are both in EDM order) | not applicable |

## Versions

- diffusers 0.41.0 (installed) and `main` at commit 1e004eb (2026-10-12). The code is the same.
- torch 2.14.1 (CPU), numpy 2.5.2, Python 3.14

## Related issues and pull requests

- #5981 (closed): `add_noise` raised `AttributeError` because `self.sigmas` did not exist before `set_timesteps`.
- PR #6006 (merged): fix for #5981. It adds `self.sigmas` to `__init__`, in the VP order. The order of the timesteps was not changed, so the two orders do not match.
- PR #5158 (merged): changed `add_noise` to read the sigmas, and caused the need for #6006.
- #6232 (closed): `add_noise` and duplicate timesteps in `DPMSolverSinglestepScheduler`. This is a different problem.
- #10266 (closed): UniPC with flow sigmas, sigma length. This is a different problem.

I did not find an open issue or pull request for this behavior.

## Proposed fix

In `__init__`, make `self.sigmas` match `self.timesteps` (same order):

- VP: flip the sigmas.
- Flow: use `timesteps / num_train_timesteps`.

A pull request is ready, with tests. `set_timesteps()` is not changed, so the behavior after `set_timesteps()` is the same as before.
