import numpy as np
import pytest


def test_sample_posterior(benchmark_robot):
    y = [1.7, 0.1]
    n_samples = 10_000
    posterior_params = benchmark_robot.sample_posterior(y, n_samples)

    assert posterior_params.shape == (n_samples, 4)

    coord = benchmark_robot.forward(posterior_params)
    assert np.isclose(
        coord[:, :2], np.tile(y, reps=(n_samples, 1))
    ).all(), "Unexpected posterior predictive discrepancy"
