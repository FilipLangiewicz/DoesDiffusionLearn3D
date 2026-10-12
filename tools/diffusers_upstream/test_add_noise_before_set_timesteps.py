"""Standalone copy of the new tests. The same tests are in tests/schedulers/test_scheduler_dpm_multi.py."""

import unittest

import torch

from diffusers import (
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)


class AddNoiseBeforeSetTimestepsTest(unittest.TestCase):
    def get_common_config(self, **kwargs):
        # Only keys that all four scheduler classes accept.
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "solver_order": 2,
            "prediction_type": "epsilon",
        }
        config.update(**kwargs)
        return config

    def get_add_noise_coefficients(self, scheduler, timestep):
        # Use x0 = 1, eps = 0 for the first coefficient and x0 = 0, eps = 1 for the second.
        timesteps = torch.tensor([timestep])
        one = torch.ones(1, 1)
        zero = torch.zeros(1, 1)
        return scheduler.add_noise(one, zero, timesteps).item(), scheduler.add_noise(zero, one, timesteps).item()

    def test_add_noise_without_set_timesteps(self):
        # `add_noise` must give the noise level of the given timestep when `set_timesteps` was not called.
        reference = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
        for scheduler_class in (
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            UniPCMultistepScheduler,
            DEISMultistepScheduler,
        ):
            scheduler = scheduler_class(**self.get_common_config())
            for timestep in (0, 200, 500, 800, 999):
                expected = self.get_add_noise_coefficients(reference, timestep)
                actual = self.get_add_noise_coefficients(scheduler, timestep)
                assert abs(actual[0] - expected[0]) < 1e-4 and abs(actual[1] - expected[1]) < 1e-4, (
                    f"{scheduler_class.__name__}, t={timestep}: expected {expected}, got {actual}"
                )

    def test_add_noise_without_set_timesteps_flow_sigmas(self):
        # Flow sigmas are in [0, 1], so x_t = (1 - t / 1000) * x0 + (t / 1000) * eps.
        for scheduler_class in (
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            UniPCMultistepScheduler,
            DEISMultistepScheduler,
        ):
            scheduler = scheduler_class(
                **self.get_common_config(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
            )
            for timestep in (0, 200, 500, 800, 999):
                actual = self.get_add_noise_coefficients(scheduler, timestep)
                expected = (1 - timestep / 1000, timestep / 1000)
                assert abs(actual[0] - expected[0]) < 1e-4 and abs(actual[1] - expected[1]) < 1e-4, (
                    f"{scheduler_class.__name__}, t={timestep}: expected {expected}, got {actual}"
                )

    def test_add_noise_after_set_timesteps_matches_before(self):
        # `set_timesteps` must keep the same noise level for a timestep that it keeps.
        for use_flow_sigmas in (False, True):
            config = self.get_common_config(final_sigmas_type="zero")
            if use_flow_sigmas:
                config.update(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
            scheduler = DPMSolverMultistepScheduler(**config)
            before = {t: self.get_add_noise_coefficients(scheduler, t) for t in (0, 200, 500, 800, 999)}
            scheduler.set_timesteps(50)
            for timestep in scheduler.timesteps.tolist():
                expected = (
                    (1 - timestep / 1000, timestep / 1000)
                    if use_flow_sigmas
                    else self.get_add_noise_coefficients(
                        DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear"), timestep
                    )
                )
                actual = self.get_add_noise_coefficients(scheduler, timestep)
                # The flow timesteps are rounded down to integers, so the tolerance is larger.
                tolerance = 2e-3 if use_flow_sigmas else 1e-4
                assert abs(actual[0] - expected[0]) < tolerance and abs(actual[1] - expected[1]) < tolerance
                if timestep in before:
                    assert abs(actual[0] - before[timestep][0]) < 2e-3
