# Notes: add_noise before set_timesteps in the solver schedulers

## Summary

- The bug is real. It is in the VP case and in the flow case.
- Cause: after `__init__`, `self.timesteps` is descending and `self.sigmas` is ascending. `add_noise` uses the index of `t` in `self.timesteps` to read `self.sigmas`.
- Tested with diffusers 0.41.0 (installed) and the `main` branch (commit 1e004eb, 2026-10-12, version 0.41.0.dev0). The code in `main` is the same.
- The behavior after `set_timesteps()` is correct. The fix does not change it.

## Is the VP case also affected?

Yes. Default `DPMSolverMultistepScheduler()`, `add_noise` at t=200, without `set_timesteps`:

| scheduler                     | a (weight of x0) | b (weight of noise) |
|-------------------------------|------------------|---------------------|
| `DDPMScheduler` t=200         | 0.8102           | 0.5862              |
| `DDPMScheduler` t=799         | 0.0391           | 0.9992              |
| `DPMSolverMultistepScheduler` | 0.0391           | 0.9992              |

The result is the noise level of t=799 (= 999 - 200).

## Schedulers with the same problem

Without `set_timesteps`, t=200:

- Wrong for VP and flow: `DPMSolverMultistepScheduler`, `DPMSolverSinglestepScheduler`, `UniPCMultistepScheduler`, `DEISMultistepScheduler`.
- `DPMSolverMultistepInverseScheduler`: correct for VP (its timesteps are ascending, like its sigmas). Wrong for flow sigmas (a=0.2764, b=0.7236 instead of 0.8, 0.2).
- `SASolverScheduler`: its `add_noise` uses `alphas_cumprod[t]`, so VP is correct. With flow sigmas it ignores the flow formula, also after `set_timesteps` (a=0.0091 at t=963, 0.0579 at t=749, 0.7239 at t=249 for 50 steps). This is a separate problem.
- `EDMDPMSolverMultistepScheduler`: not affected. Its `timesteps` are the preconditioned EDM noise values and its `sigmas` are in the same (descending) order. `add_noise` with a value from `scheduler.timesteps` gives the matching sigma.
- Not checked in detail: the other schedulers with a separate `add_noise` (Euler, Heun, KDPM2, LMS, SDE). They set `sigmas` and `timesteps` in `__init__` in one order. I did not find the same pattern.

After `set_timesteps(50)` with t taken from `scheduler.timesteps`, all six schedulers above give the correct value (VP: same as `DDPMScheduler`; flow: `1 - t/1000`, within the integer rounding of the timestep), except `SASolverScheduler` with flow sigmas.

## Existing issues and pull requests (read-only search)

Searches: "add_noise set_timesteps DPMSolverMultistepScheduler", "add_noise before set_timesteps", "use_flow_sigmas add_noise", "sigmas reversed __init__", and related terms. I did not find an issue or a pull request for the wrong order of the sigmas.

| link | status | relation |
|------|--------|----------|
| https://github.com/huggingface/diffusers/issues/5981 | closed (completed, 2023-12-01) | `add_noise` before `set_timesteps` raised `AttributeError` (no `sigmas`). Same call order as this bug. |
| https://github.com/huggingface/diffusers/pull/6006 | merged (2023-12-01) | Fix for #5981: adds `self.sigmas` to `__init__`. The diff is one line, the VP sigmas in ascending order. This line is the cause of the present bug. The PR text says that `add_noise` without `set_timesteps` is "typical in training". |
| https://github.com/huggingface/diffusers/pull/5158 | merged (2023-09-23) | Changed `add_noise` of DPM-Solver multistep, singlestep, UniPC and DEIS to use sigmas. This caused #5981. |
| https://github.com/huggingface/diffusers/issues/6232 | closed (2024-02-18) | `add_noise` of `DPMSolverSinglestepScheduler` fails with duplicate timesteps. Different problem. |
| https://github.com/huggingface/diffusers/issues/10266 | closed (2024-12-18) | UniPC with flow sigmas, index out of bounds in `step`. Different problem (sigma length). |
| https://github.com/huggingface/diffusers/pull/10267 | merged | Fix for #10266 (final sigma with flow sigmas). Different problem. |
| https://github.com/huggingface/diffusers/discussions/10902 | discussion | Which `add_noise` to use for training. The answer suggests `DDPMScheduler`. |
| https://github.com/huggingface/diffusers/issues/5687, /5685, /7651 | closed | Img2img and duplicate timesteps. Not the same problem. |

I did not read the status of every result in full. The statuses above come from the fetched pages (I did not check for newer comments after the fetch).

## Fix

Files changed in the clone `/home/flang/tools/diffusers-src` (not committed; see `fix.patch`):

- `scheduling_dpmsolver_multistep.py`, `scheduling_dpmsolver_singlestep.py`, `scheduling_unipc_multistep.py`, `scheduling_deis_multistep.py`: in `__init__`, `self.sigmas` is flipped (VP) or set to `timesteps / num_train_timesteps` (flow).
- `tests/schedulers/test_scheduler_dpm_multi.py`: three new tests.

Design choice: the change is in `__init__` and not in `add_noise`. It is 5 lines for each file. `add_noise`, `set_timesteps` and `step` stay as they are. The flow sigma at timestep t is `t / num_train_timesteps`, because `set_timesteps` defines `timesteps = sigmas * num_train_timesteps` (the shift is already in the timestep value). The `flow_shift` value is not used in `__init__`.

Risk: code that reads `scheduler.sigmas` before `set_timesteps` and expects the ascending order. I searched `src/` and `examples/` for `scheduler.sigmas`: all uses are after `set_timesteps` or inside the schedulers. `step()` needs `set_timesteps` anyway.

## Test results (exact)

Environment: project venv python, `PYTHONPATH=/home/flang/tools/diffusers-src/src`, pytest, CPU.

New tests, before the fix (`git stash` of `src`):

```
FAILED ...::test_add_noise_after_set_timesteps_matches_before
FAILED ...::test_add_noise_without_set_timesteps
FAILED ...::test_add_noise_without_set_timesteps_flow_sigmas
3 failed
```

New tests, after the fix:

```
3 passed
```

Full file `tests/schedulers/test_scheduler_dpm_multi.py`:

```
before the fix: 4 failed, 40 passed, 1 skipped   (3 new tests + test_compatibles)
after the fix:  1 failed, 43 passed, 1 skipped   (test_compatibles)
```

`test_compatibles` fails in both cases: `ImportError: DPMSolverSDEScheduler requires the torchsde library but it was not found in your environment.` The same test fails in the files for the single-step, UniPC, DEIS and inverse schedulers, also without the fix.

Other scheduler test files after the fix (dpm_multi, dpm_single, unipc, deis, dpm_multi_inverse, edm_dpmsolver_multistep, sasolver):

```
6 failed, 247 passed, 26 skipped   (all 6 failures are test_compatibles, missing torchsde)
```

Without the fix, for the files dpm_single, unipc, deis and dpm_multi_inverse: `5 failed, 165 passed, 2 skipped` (all `test_compatibles`).

Other checks: `ruff check src/diffusers/schedulers tests/schedulers/test_scheduler_dpm_multi.py` passes. `ruff format` is applied to the test file. `python utils/check_copies.py` prints nothing (no difference).

The standalone copy `test_add_noise_before_set_timesteps.py` gives the same result: 3 failed before the fix, 3 passed after the fix.

## Output of repro.py

Installed diffusers 0.41.0 (before the fix):

```
torch 2.14.1+cpu, diffusers 0.41.0

(a) Sana flow config, add_noise WITHOUT set_timesteps
    expected (flow matching): a = 1 - t/1000, b = t/1000
DPMSolverMultistep (Sana config)             t=200  got a= -24.5285 b=  25.5285 | expected a= 0.8000 b= 0.2000 | WRONG
DPMSolverMultistep (Sana config)             t=500  got a=  -2.4241 b=   3.4241 | expected a= 0.5000 b= 0.5000 | WRONG
DPMSolverMultistep (Sana config)             t=800  got a=   0.2807 b=   0.7193 | expected a= 0.2000 b= 0.8000 | WRONG

(b) Default VP config, add_noise WITHOUT set_timesteps (reference: DDPMScheduler, same betas)
DPMSolverMultistep (defaults)                t=200  got a=   0.0391 b=   0.9992 | expected a= 0.8102 b= 0.5862 | WRONG
DPMSolverMultistep (defaults)                t=500  got a=   0.2803 b=   0.9599 | expected a= 0.2789 b= 0.9603 | WRONG
DPMSolverMultistep (defaults)                t=800  got a=   0.8118 b=   0.5839 | expected a= 0.0388 b= 0.9992 | WRONG
    DDPM at t=200: a=0.8102 b=0.5862
    DDPM at t=799: a=0.0391 b=0.9992
    DPMSolver at t=200 : a=0.0391 b=0.9992  (matches t=799)

(c) Same calls AFTER set_timesteps(50). t is taken from scheduler.timesteps.
Sana config after set_timesteps, i=5         t=963  got a=   0.0361 b=   0.9639 | expected a= 0.0370 b= 0.9630 | OK
Sana config after set_timesteps, i=25        t=749  got a=   0.2504 b=   0.7496 | expected a= 0.2510 b= 0.7490 | OK
Sana config after set_timesteps, i=45        t=249  got a=   0.7502 b=   0.2498 | expected a= 0.7510 b= 0.2490 | OK
VP defaults after set_timesteps, i=5         t=899  got a=   0.0166 b=   0.9999 | expected a= 0.0166 b= 0.9999 | OK
VP defaults after set_timesteps, i=25        t=500  got a=   0.2789 b=   0.9603 | expected a= 0.2789 b= 0.9603 | OK
VP defaults after set_timesteps, i=45        t=100  got a=   0.9461 b=   0.3238 | expected a= 0.9461 b= 0.3238 | OK

(d) Other schedulers, VP defaults, add_noise WITHOUT set_timesteps, t=200
DPMSolverMultistepScheduler                  t=200  got a=   0.0391 b=   0.9992 | expected a= 0.8102 b= 0.5862 | WRONG
DPMSolverSinglestepScheduler                 t=200  got a=   0.0391 b=   0.9992 | expected a= 0.8102 b= 0.5862 | WRONG
UniPCMultistepScheduler                      t=200  got a=   0.0391 b=   0.9992 | expected a= 0.8102 b= 0.5862 | WRONG
DEISMultistepScheduler                       t=200  got a=   0.0391 b=   0.9992 | expected a= 0.8102 b= 0.5862 | WRONG
SASolverScheduler                            t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
DPMSolverMultistepInverseScheduler           t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
    EDMDPMSolverMultistepScheduler uses sigma_data/EDM noise levels, so it has no VP reference.
    EDM: timesteps[:3]=[1.0955066680908203, 1.094139814376831, 1.0927720069885254], sigmas[:3]=[79.99998474121094, 79.56378936767578, 79.12967681884766], len(ti
    EDM add_noise with a timestep value taken from s.timesteps: 79.99998474121094

(e) Flow sigmas in the other schedulers, add_noise WITHOUT set_timesteps, t=200
    expected: a = 0.8, b = 0.2
DPMSolverMultistepScheduler (flow)           t=200  got a= -24.5285 b=  25.5285 | expected a= 0.8000 b= 0.2000 | WRONG
DPMSolverSinglestepScheduler (flow)          t=200  got a= -24.5285 b=  25.5285 | expected a= 0.8000 b= 0.2000 | WRONG
UniPCMultistepScheduler (flow)               t=200  got a= -24.5285 b=  25.5285 | expected a= 0.8000 b= 0.2000 | WRONG
DEISMultistepScheduler (flow)                t=200  got a= -24.5285 b=  25.5285 | expected a= 0.8000 b= 0.2000 | WRONG
SASolverScheduler (flow)                     t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8000 b= 0.2000 | WRONG
DPMSolverMultistepInverseScheduler (flow)    t=200  got a=   0.2764 b=   0.7236 | expected a= 0.8000 b= 0.2000 | WRONG
```

Clone with the fix (version 0.41.0.dev0):

```
torch 2.14.1+cpu, diffusers 0.41.0.dev0

(a) Sana flow config, add_noise WITHOUT set_timesteps
    expected (flow matching): a = 1 - t/1000, b = t/1000
DPMSolverMultistep (Sana config)             t=200  got a=   0.8000 b=   0.2000 | expected a= 0.8000 b= 0.2000 | OK
DPMSolverMultistep (Sana config)             t=500  got a=   0.5000 b=   0.5000 | expected a= 0.5000 b= 0.5000 | OK
DPMSolverMultistep (Sana config)             t=800  got a=   0.2000 b=   0.8000 | expected a= 0.2000 b= 0.8000 | OK

(b) Default VP config, add_noise WITHOUT set_timesteps (reference: DDPMScheduler, same betas)
DPMSolverMultistep (defaults)                t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
DPMSolverMultistep (defaults)                t=500  got a=   0.2789 b=   0.9603 | expected a= 0.2789 b= 0.9603 | OK
DPMSolverMultistep (defaults)                t=800  got a=   0.0388 b=   0.9992 | expected a= 0.0388 b= 0.9992 | OK
    DDPM at t=200: a=0.8102 b=0.5862
    DDPM at t=799: a=0.0391 b=0.9992
    DPMSolver at t=200 : a=0.8102 b=0.5862  (matches t=200)

(c) Same calls AFTER set_timesteps(50). t is taken from scheduler.timesteps.
Sana config after set_timesteps, i=5         t=963  got a=   0.0361 b=   0.9639 | expected a= 0.0370 b= 0.9630 | OK
Sana config after set_timesteps, i=25        t=749  got a=   0.2504 b=   0.7496 | expected a= 0.2510 b= 0.7490 | OK
Sana config after set_timesteps, i=45        t=249  got a=   0.7502 b=   0.2498 | expected a= 0.7510 b= 0.2490 | OK
VP defaults after set_timesteps, i=5         t=899  got a=   0.0166 b=   0.9999 | expected a= 0.0166 b= 0.9999 | OK
VP defaults after set_timesteps, i=25        t=500  got a=   0.2789 b=   0.9603 | expected a= 0.2789 b= 0.9603 | OK
VP defaults after set_timesteps, i=45        t=100  got a=   0.9461 b=   0.3238 | expected a= 0.9461 b= 0.3238 | OK

(d) Other schedulers, VP defaults, add_noise WITHOUT set_timesteps, t=200
DPMSolverMultistepScheduler                  t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
DPMSolverSinglestepScheduler                 t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
UniPCMultistepScheduler                      t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
DEISMultistepScheduler                       t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
SASolverScheduler                            t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
DPMSolverMultistepInverseScheduler           t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8102 b= 0.5862 | OK
    EDMDPMSolverMultistepScheduler uses sigma_data/EDM noise levels, so it has no VP reference.
    EDM: timesteps[:3]=[1.0955066680908203, 1.094139814376831, 1.0927720069885254], sigmas[:3]=[79.99998474121094, 79.56378936767578, 79.12967681884766], len(ti
    EDM add_noise with a timestep value taken from s.timesteps: 79.99998474121094

(e) Flow sigmas in the other schedulers, add_noise WITHOUT set_timesteps, t=200
    expected: a = 0.8, b = 0.2
DPMSolverMultistepScheduler (flow)           t=200  got a=   0.8000 b=   0.2000 | expected a= 0.8000 b= 0.2000 | OK
DPMSolverSinglestepScheduler (flow)          t=200  got a=   0.8000 b=   0.2000 | expected a= 0.8000 b= 0.2000 | OK
UniPCMultistepScheduler (flow)               t=200  got a=   0.8000 b=   0.2000 | expected a= 0.8000 b= 0.2000 | OK
DEISMultistepScheduler (flow)                t=200  got a=   0.8000 b=   0.2000 | expected a= 0.8000 b= 0.2000 | OK
SASolverScheduler (flow)                     t=200  got a=   0.8102 b=   0.5862 | expected a= 0.8000 b= 0.2000 | WRONG
DPMSolverMultistepInverseScheduler (flow)    t=200  got a=   0.2764 b=   0.7236 | expected a= 0.8000 b= 0.2000 | WRONG
```

(The line `WRONG` for t=500 in block (b) of the first output is an off-by-one: the result is the noise level of t=499, and the tolerance is 1e-3. Also, the last two `WRONG` lines after the fix are the two separate problems named above: `SASolverScheduler` and the inverse scheduler with flow sigmas.)

## Honest assessment

Real bug or intended behavior?

- It is a real bug. `add_noise` has a name and a doc string that promise the noise level of the given timestep. The wrong result is a clear mismatch with `DDPMScheduler`. No document says that `set_timesteps` is required first.
- The history supports this view. #5981 was a report about exactly this call order. A maintainer fixed it in PR #6006 and called the call order typical in training. That fix put the sigmas in `__init__` but in the wrong order, and no test checked the values.
- The VP part is clear and I expect a maintainer to accept it.
- The flow part is less clear. The maintainers can say that the flow schedulers are made for inference, and that `FlowMatchEulerDiscreteScheduler.scale_noise` or `DDPMScheduler` is the right tool for training. Also, the definition of "timestep" for flow sigmas (shifted `sigma * 1000`) is a convention that I took from `set_timesteps`. A maintainer may prefer a different convention (for example, an unshifted `t / 1000`). I would explain this convention in the issue.
- Risk for the maintainers: a change of `scheduler.sigmas` before `set_timesteps` is a visible change. It is small, and I found no user of the old order.

Is a pull request useful if an issue exists?

- No open issue or pull request exists for this problem, so the pull request does not duplicate work. The closed #5981 and #6006 are related history, and the pull request should link them.
- A pull request with failing tests and a small fix is usually more helpful than an issue alone. Open the issue first (or both together) and link them. Then a maintainer can decide on the flow convention.
- The repository activity is high and the stale bot closes old issues. A short pull request has a better chance to get a review.

For the research project: the easy workaround is to not use `add_noise` of this scheduler before `set_timesteps`. Use the formula `x_t = (1 - t/1000) * x0 + (t/1000) * eps` for the Sana flow config, where `t` is the value from `scheduler.timesteps` (this is also what `add_noise` gives after `set_timesteps`). Or call `set_timesteps(N)` with a number of steps that contains the needed `t` values.
