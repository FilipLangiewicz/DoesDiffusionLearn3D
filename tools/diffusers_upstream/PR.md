# Title

[schedulers] Fix the sigma order for `add_noise()` before `set_timesteps()`

# What does this PR do?

Fixes #<ISSUE NUMBER>

After `__init__`, `self.timesteps` goes from 999 down to 0, but `self.sigmas` (added in #6006) went from step 0 up to step 999. So `add_noise(x0, noise, t)` before `set_timesteps()` used the noise level of `999 - t`. With `use_flow_sigmas=True` (the SanaPipeline scheduler), it also used a VP sigma in the flow formula: for `t = 200` the input was `-24.5 * x0 + 25.5 * eps` instead of `0.8 * x0 + 0.2 * eps`.

The change is one block in `__init__` of four schedulers (`DPMSolverMultistep`, `DPMSolverSinglestep`, `UniPCMultistep`, `DEISMultistep`):

- VP sigmas: flip them, so that `sigmas[i]` belongs to `timesteps[i]`.
- Flow sigmas: use `timestep / num_train_timesteps`. This is the same convention as `set_timesteps`, where `timesteps = sigmas * num_train_timesteps`, so `flow_shift` is not applied twice.

`add_noise`, `set_timesteps` and `step` do not change. The values after `set_timesteps()` are the same as before.

## Effect on a real model

One-step clean estimate of a real image after `pipe.scheduler.add_noise` (relative error, lower is better). SanaPipeline with SANA-1.6B, the script of the issue, float32 on a T4:

| t | main | this PR | expected flow noise |
|---:|---:|---:|---:|
| 200 | 36.618 | 0.180 | 0.180 |
| 500 | 4.615 | 0.379 | 0.379 |
| 800 | 0.601 | 0.624 | 0.624 |

`Efficient-Large-Model/Sana_600M_512px_diffusers` (float32, CPU) gives the same picture: 45.5 against 0.228 at `t = 200`.

`google/ddpm-cifar10-32` with the four schedulers at `t = 100`: 0.590 on main, 0.045 with this PR, 0.045 with `DDPMScheduler`.

## Tests

Three new tests in `tests/schedulers/test_scheduler_dpm_multi.py`:

- the VP result must equal `DDPMScheduler.add_noise` (four schedulers, five timesteps);
- the flow result must be `(1 - t/1000) * x0 + (t/1000) * eps` (four schedulers);
- the result after `set_timesteps(50)` must agree with the result before it.

Commands and results (CPU, torch 2.14.1, Python 3.14):

```
python -m pytest tests/schedulers/test_scheduler_dpm_multi.py tests/schedulers/test_scheduler_dpm_single.py \
  tests/schedulers/test_scheduler_unipc.py tests/schedulers/test_scheduler_deis.py \
  tests/schedulers/test_scheduler_dpm_multi_inverse.py tests/schedulers/test_scheduler_edm_dpmsolver_multistep.py \
  tests/schedulers/test_scheduler_sasolver.py
# this PR: 273 passed, 6 skipped. main: the 3 new tests fail, all other tests pass.

python -m pytest tests/pipelines/sana
# this PR: 68 passed, 78 skipped

make style; python utils/check_copies.py; python utils/check_dummies.py
# no changes, no errors
```

## Not in this PR

- `DPMSolverMultistepInverseScheduler` with `use_flow_sigmas=True` has a similar problem before `set_timesteps()`.
- `SASolverScheduler.add_noise` ignores `use_flow_sigmas`, also after `set_timesteps()`.

I can add these here or in a separate PR.

## Checklist of the template

GitHub adds the checklist of the diffusers template to the PR text. Tick these items: AI agent used (yes), AI agent guide read, self-review run and notes shared (below), contributor guideline read, issue link, new tests written. Leave the docs item open.

## Self-review notes

Rubric: `.ai/references/review-rules.md` and `code_style.md`, on the full diff.

- Blocking issues: none.
- Non-blocking, left for the review:
  1. The tests for all four schedulers are in `test_scheduler_dpm_multi.py`. I can move them into the test file of each scheduler.
  2. The flow branch uses `timestep / num_train_timesteps`. An alternative is to apply `flow_shift` to the sigmas in `__init__` too, but then `add_noise(t)` would differ from the value after `set_timesteps()` for the same `t`.
  3. The two schedulers in "Not in this PR" are out of scope.
- Dead code: none. The change adds no parameters or helpers.
- Other users of `self.sigmas`: in these four schedulers only `add_noise` reads it before `set_timesteps()`. `step()` and the pipelines read it after `set_timesteps()`, which overwrites it.
- Docs: no public behavior change except the corrected values, so no docs update.

## Who can review?

@yiyixuxu @dg845
