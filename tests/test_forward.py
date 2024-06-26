import pytest


def test_sample_shapes(benchmark_robot):
    n_samples = 17
    prior_params, coords = benchmark_robot.sample(n_samples, only_end=False)
    assert prior_params.shape == (n_samples, 4)
    assert coords.shape == (n_samples, 4, 3)
