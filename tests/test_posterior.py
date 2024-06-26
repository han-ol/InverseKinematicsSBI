import time

import metrics.c2st as c2st
import numpy as np
import pytest

y = [1.7, 0.1]
n_samples = 1_000


@pytest.fixture
def posterior_samples(benchmark_robot):
    t1 = time.time()
    posterior_params = benchmark_robot.sample_posterior(y, n_samples)
    t2 = time.time()
    delta_t = t2 - t1
    print("Time elapsed while sampling posterior:", delta_t, "seconds")
    assert delta_t < 15, "Posterior sampling took unexpectedly long."

    return posterior_params


def test_posterior_sample_shape(posterior_samples):
    assert posterior_samples.shape == (n_samples, 4)


def test_posterior_predictive(benchmark_robot, posterior_samples):
    coord = benchmark_robot.forward(posterior_samples)
    assert np.isclose(
        coord[:, :2], np.tile(y, reps=(n_samples, 1))
    ).all(), "Unexpected posterior predictive discrepancy detected."


def test_posterior_c2st(posterior_samples):
    c2st_result = c2st.c2st(posterior_samples[: n_samples // 2], posterior_samples[n_samples // 2 :])
    print("C2ST:", c2st_result)
    assert c2st_result < 0.6, "Sampled posteriors are significantly different to the reference."
