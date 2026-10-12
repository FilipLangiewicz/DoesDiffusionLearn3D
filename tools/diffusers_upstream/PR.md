# Title

Fix `add_noise` before `set_timesteps` in the multistep and singlestep solver schedulers

## What

`self.sigmas` is now in the same order as `self.timesteps` after `__init__`. This applies to:

- `DPMSolverMultistepScheduler`
- `DPMSolverSinglestepScheduler`
- `UniPCMultistepScheduler`
- `DEISMultistepScheduler`

If `use_flow_sigmas=True`, `self.sigmas` is now `timestep / num_train_timesteps`. This is a flow sigma in [0, 1]. Before this change, the VP sigmas were used.

`set_timesteps()` and `add_noise()` are not changed.

## Why

In `__init__`, `self.timesteps` is descending (`[999, ..., 0]`), but `self.sigmas` was ascending. `add_noise` finds the index of `t` in `self.timesteps` and reads `self.sigmas` at this index. So `add_noise(x0, noise, t)` used the sigma of `999 - t` if `set_timesteps()` was not called. This call order is the normal case in training code (see #5981).

With `use_flow_sigmas=True` the VP sigmas also gave invalid weights, because `alpha_t = 1 - sigma` is correct only for a sigma in [0, 1]. Example for the Sana scheduler at `t = 200`: the result was `-24.5 * x0 + 25.5 * eps`. The correct result is `0.8 * x0 + 0.2 * eps`.

PR #6006 added `self.sigmas` to `__init__` for #5981. It did not match the order of the timesteps. This PR completes that fix.

The flow value `timestep / num_train_timesteps` follows `set_timesteps`, where `timesteps = sigmas * num_train_timesteps`. So `flow_shift` is already part of the timestep value, and it is not applied a second time.

## How tested

New tests in `tests/schedulers/test_scheduler_dpm_multi.py`:

- `test_add_noise_without_set_timesteps`: VP config. The result must equal `DDPMScheduler.add_noise` for t in (0, 200, 500, 800, 999). It runs for the four schedulers.
- `test_add_noise_without_set_timesteps_flow_sigmas`: flow config. The result must be `(1 - t/1000, t/1000)`. It runs for the four schedulers.
- `test_add_noise_after_set_timesteps_matches_before`: after `set_timesteps(50)`, the result is correct for each timestep, and it is the same as before `set_timesteps`.

Results (CPU, torch 2.14.1):

- Before the fix: 3 new tests fail.
- After the fix: 3 new tests pass.
- `tests/schedulers/test_scheduler_dpm_multi.py`: 43 passed, 1 skipped, 1 failed. The failed test is `test_compatibles`. It needs `torchsde`, which is not installed. It fails in the same way without this change.
- The files for DPM-Solver singlestep, UniPC, DEIS, DPM-Solver inverse, EDM DPM-Solver and SA-Solver: no new failure. Only `test_compatibles` fails, for the same reason (`torchsde`).
- `ruff check` and `ruff format` pass. `utils/check_copies.py` reports no difference.

## Not in this PR

- `DPMSolverMultistepInverseScheduler` with `use_flow_sigmas=True` has the same problem before `set_timesteps`. Its timesteps are ascending, so only the flow branch needs a change.
- `SASolverScheduler.add_noise` always uses `alphas_cumprod`. It ignores `use_flow_sigmas`, also after `set_timesteps`.

I can add these to this PR if the maintainers want it.

## Links

- Related: #5981, #6006, #5158
