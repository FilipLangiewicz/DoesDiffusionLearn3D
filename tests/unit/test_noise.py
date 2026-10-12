import pytest
import torch

from dd3d.noise import flow_noise, flow_sigma, image_seed, legacy_zzsn_noise, noise_coefficients

SCHEDULER_CONFIG = dict(
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
    sample_max_value=1.0,
    dynamic_thresholding_ratio=0.995,
    lambda_min_clipped=-float("inf"),
    rescale_betas_zero_snr=False,
    steps_offset=0,
    use_karras_sigmas=False,
    use_exponential_sigmas=False,
    use_beta_sigmas=False,
    use_lu_lambdas=False,
    variance_type=None,
)


def make_latents():
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(1, 32, 32, 32, generator=gen)
    eps = torch.randn(1, 32, 32, 32, generator=gen)
    return x0, eps


def test_flow_noise_formula():
    x0, eps = make_latents()
    expected = (1 - 200 / 1000) * x0 + (200 / 1000) * eps
    torch.testing.assert_close(flow_noise(x0, eps, 200), expected)


def test_noise_coefficients_recover_flow_weights():
    x0, eps = make_latents()
    a, b = noise_coefficients(flow_noise(x0, eps, 200), x0, eps)
    assert a == pytest.approx(0.8, abs=1e-5)
    assert b == pytest.approx(0.2, abs=1e-5)


def test_flow_sigma_range():
    assert flow_sigma(0) == 0.0
    assert flow_sigma(1000) == 1.0
    with pytest.raises(ValueError):
        flow_sigma(1001)
    with pytest.raises(ValueError):
        flow_sigma(-1)


def test_image_seed_is_deterministic_and_distinct():
    assert image_seed(1234, "Sphere_new_000000") == image_seed(1234, "Sphere_new_000000")
    assert image_seed(1234, "Sphere_new_000000") != image_seed(1234, "Sphere_new_000001")
    assert image_seed(1, "a") != image_seed(2, "a")


def test_legacy_zzsn_noise_pitfall():
    """The old scheduler gives other weights than the flow rule."""
    diffusers = pytest.importorskip("diffusers")
    sched = diffusers.DPMSolverMultistepScheduler(**SCHEDULER_CONFIG)
    if not sched.sigmas[0] < sched.sigmas[-1]:
        pytest.skip("diffusers keeps the sigmas in descending order: the old behavior is fixed")
    x0, eps = make_latents()
    a200, b200 = noise_coefficients(legacy_zzsn_noise(sched, x0, eps, 200), x0, eps)
    assert a200 == pytest.approx(-24.53, abs=0.05)
    assert b200 == pytest.approx(25.53, abs=0.05)
    a800, b800 = noise_coefficients(legacy_zzsn_noise(sched, x0, eps, 800), x0, eps)
    assert a800 == pytest.approx(0.281, abs=0.05)
    assert b800 == pytest.approx(0.719, abs=0.05)
