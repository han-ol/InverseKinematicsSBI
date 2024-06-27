import pickle
import time

import metrics.c2st as c2st
import numpy as np
import pytest

y = [1.7, 0.1]
n_samples = 100


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


@pytest.fixture
def reference_posteriors():
    with open("tests/invkinematics_reference_posterior_samples_custom_abc_0.002.pkl", "rb") as file:
        ref_post = pickle.load(file)
    return ref_post


n_test_indeces = 10


@pytest.fixture
def test_indeces():
    exclude_mask = np.ones(100)
    exclude_mask[[6, 25, 59]] = 0  # exclude references of arms where finite ABC precision
    # leads to significant errors rendering the reference itself wrong
    options = np.arange(100)[exclude_mask == 1]
    test_indeces = np.random.choice(options, size=n_test_indeces, replace=False)
    return test_indeces


@pytest.fixture
def prepared_posteriors(benchmark_robot, reference_posteriors: np.ndarray, test_indeces) -> list[dict]:
    posteriors: list[dict] = []

    for idx in test_indeces:
        coord = benchmark_robot.forward(reference_posteriors[idx])[:, :2]
        y = np.mean(coord, axis=0)
        # y_err = np.std(coord, axis=0)

        posteriors.append(dict(y=y, posterior=reference_posteriors[idx]))

    return posteriors


@pytest.mark.parametrize("j", range(n_test_indeces))
def test_posteriors_c2st(benchmark_robot, prepared_posteriors, test_indeces, j):
    reference = prepared_posteriors[j]["posterior"]
    y = prepared_posteriors[j]["y"]
    n_samples = len(reference)
    post_samples = benchmark_robot.sample_posterior(y, n_samples)

    c2st_result = c2st.c2st(post_samples, reference)
    print(f"C2ST for reference posterior at idx={test_indeces[j]}, y={y}:", c2st_result)

    assert c2st_result < 0.53, "Sampled posteriors are significantly different to the reference."
