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
