"""Reproduction: add_noise() before set_timesteps() in the multistep solver schedulers.

No model weights are needed. The script runs on CPU.
It prints, for each case, the coefficients (a, b) in x_t = a * x0 + b * eps.
"""

import torch

import diffusers
from diffusers import (
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EDMDPMSolverMultistepScheduler,
    SASolverScheduler,
    UniPCMultistepScheduler,
)

# Config of the scheduler in Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers.
SANA_CONFIG = dict(
    algorithm_type="dpmsolver++",
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
    num_train_timesteps=1000,
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    flow_shift=3.0,
    solver_order=2,
    solver_type="midpoint",
    final_sigmas_type="zero",
    timestep_spacing="linspace",
    lower_order_final=True,
    euler_at_final=False,
    thresholding=False,
    lambda_min_clipped=-float("inf"),
)


def coefficients(scheduler, t):
    """Return (a, b) with x_t = a * x0 + b * eps. Uses x0 = 1 and eps = 0, then x0 = 0 and eps = 1."""
    timestep = torch.tensor([t])
    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    a = scheduler.add_noise(one, zero, timestep).item()
    b = scheduler.add_noise(zero, one, timestep).item()
    return a, b


def row(name, t, got, expected):
    ok = abs(got[0] - expected[0]) < 1e-3 and abs(got[1] - expected[1]) < 1e-3
    print(
        f"{name:<44} t={t:<4} got a={got[0]:9.4f} b={got[1]:9.4f} | "
        f"expected a={expected[0]:7.4f} b={expected[1]:7.4f} | {'OK' if ok else 'WRONG'}"
    )


def vp_expected(t, **kw):
    ddpm = DDPMScheduler(**kw)
    return coefficients(ddpm, t)


def main():
    print(f"torch {torch.__version__}, diffusers {diffusers.__version__}")
    print()

    print("(a) Sana flow config, add_noise WITHOUT set_timesteps")
    print("    expected (flow matching): a = 1 - t/1000, b = t/1000")
    for t in (200, 500, 800):
        s = DPMSolverMultistepScheduler(**SANA_CONFIG)
        row("DPMSolverMultistep (Sana config)", t, coefficients(s, t), (1 - t / 1000, t / 1000))
    print()

    print("(b) Default VP config, add_noise WITHOUT set_timesteps (reference: DDPMScheduler, same betas)")
    for t in (200, 500, 800):
        s = DPMSolverMultistepScheduler()
        row("DPMSolverMultistep (defaults)", t, coefficients(s, t), vp_expected(t))
    # Show that the result equals the noise level of t' = 999 - t.
    s = DPMSolverMultistepScheduler()
    got = coefficients(s, 200)
    for tt in (200, 799):
        print(f"    DDPM at t={tt}: a={vp_expected(tt)[0]:.4f} b={vp_expected(tt)[1]:.4f}")
    match = "t=200" if abs(got[0] - vp_expected(200)[0]) < 1e-3 else "t=799"
    print(f"    DPMSolver at t=200 : a={got[0]:.4f} b={got[1]:.4f}  (matches {match})")
    print()

    print("(c) Same calls AFTER set_timesteps(50). t is taken from scheduler.timesteps.")
    s = DPMSolverMultistepScheduler(**SANA_CONFIG)
    s.set_timesteps(50)
    for i in (5, 25, 45):
        t = int(s.timesteps[i])
        row(f"Sana config after set_timesteps, i={i}", t, coefficients(s, t), (1 - t / 1000, t / 1000))
    s = DPMSolverMultistepScheduler()
    s.set_timesteps(50)
    for i in (5, 25, 45):
        t = int(s.timesteps[i])
        row(f"VP defaults after set_timesteps, i={i}", t, coefficients(s, t), vp_expected(t))
    print()

    print("(d) Other schedulers, VP defaults, add_noise WITHOUT set_timesteps, t=200")
    t = 200
    expected = vp_expected(t)
    for cls in (
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        UniPCMultistepScheduler,
        DEISMultistepScheduler,
        SASolverScheduler,
        DPMSolverMultistepInverseScheduler,
    ):
        try:
            s = cls()
            row(cls.__name__, t, coefficients(s, t), expected)
        except Exception as e:  # report, do not stop
            print(f"{cls.__name__:<44} t={t:<4} error: {type(e).__name__}: {e}")
    print("    EDMDPMSolverMultistepScheduler uses sigma_data/EDM noise levels, so it has no VP reference.")
    try:
        s = EDMDPMSolverMultistepScheduler()
        print(
            f"    EDM: timesteps[:3]={s.timesteps[:3].tolist()}, sigmas[:3]={s.sigmas[:3].tolist()}, "
            f"len(timesteps)={len(s.timesteps)}, len(sigmas)={len(s.sigmas)}"
        )
        x = torch.zeros(1, 1)
        print("    EDM add_noise with a timestep value taken from s.timesteps:", s.add_noise(x, torch.ones(1, 1), s.timesteps[:1]).item())
    except Exception as e:
        print(f"    EDM error: {type(e).__name__}: {e}")
    print()

    print("(e) Flow sigmas in the other schedulers, add_noise WITHOUT set_timesteps, t=200")
    print("    expected: a = 0.8, b = 0.2")
    flow_kwargs = dict(use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction")
    for cls in (
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        UniPCMultistepScheduler,
        DEISMultistepScheduler,
        SASolverScheduler,
        DPMSolverMultistepInverseScheduler,
    ):
        try:
            s = cls(**flow_kwargs)
        except Exception as e:
            print(f"{cls.__name__:<44} cannot build with flow kwargs: {type(e).__name__}: {str(e)[:70]}")
            continue
        try:
            row(cls.__name__ + " (flow)", 200, coefficients(s, 200), (0.8, 0.2))
        except Exception as e:
            print(f"{cls.__name__:<44} error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
