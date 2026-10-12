"""Noise rules for the activation extraction.

SANA-1.6B is a flow model: the transformer that gets the timestep ``t`` expects
the input ``x_t = (1 - s) * x0 + s * eps`` with ``s = t / 1000``.

The ZZSN code called ``scheduler.add_noise`` of ``DPMSolverMultistepScheduler``
before ``set_timesteps``. After ``__init__`` that scheduler keeps the sigmas in
the opposite order of the timesteps and in the VP form. For the SANA config the
call gives ``-24.5 x0 + 25.5 eps`` for t = 200 (see tests/test_noise.py).
"""

from __future__ import annotations

import torch

NUM_TRAIN_TIMESTEPS = 1000


def flow_sigma(timestep: int | float, num_train_timesteps: int = NUM_TRAIN_TIMESTEPS) -> float:
    """Return the flow sigma that the transformer expects for ``timestep``."""
    s = float(timestep) / num_train_timesteps
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"timestep {timestep} is outside [0, {num_train_timesteps}]")
    return s


def flow_noise(latents: torch.Tensor, noise: torch.Tensor, timestep: int | float) -> torch.Tensor:
    """Return the noised latents for a flow model at ``timestep``."""
    s = flow_sigma(timestep)
    return (1.0 - s) * latents + s * noise


def legacy_zzsn_noise(scheduler, latents: torch.Tensor, noise: torch.Tensor, timestep: int) -> torch.Tensor:
    """Return the noised latents of the ZZSN code (kept only to reproduce the old data)."""
    return scheduler.add_noise(latents, noise, torch.tensor([timestep]))


def noise_coefficients(
    noised: torch.Tensor, latents: torch.Tensor, noise: torch.Tensor
) -> tuple[float, float]:
    """Return ``(a, b)`` of the least squares fit ``noised = a * latents + b * noise``."""
    design = torch.stack([latents.flatten(), noise.flatten()], dim=1).double()
    coef = torch.linalg.lstsq(design, noised.flatten().double().unsqueeze(1)).solution.flatten()
    return float(coef[0]), float(coef[1])


def image_seed(base_seed: int, image_id: str) -> int:
    """Return a stable seed for one image, so that each run gives the same noise."""
    value = base_seed
    for ch in image_id:
        value = (value * 1_000_003 + ord(ch)) % (2**31 - 1)
    return value
