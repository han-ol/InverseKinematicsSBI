import numpy as np
from matplotlib import pyplot as plt

from inverse_kinematics.benchmark.benchmark_robot import BenchmarkRobot
from inverse_kinematics.plot import plot_arms

arm = BenchmarkRobot()

prior_params, coords = arm.sample(100, only_end=False)

plot_arms(coords, "prior_benchmark")

y = np.array([1.2, 0.3])

final_max_ratios = []
for weights in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]:
    print(weights)
    posterior_params = arm.sample_posterior(y, 1000, weights=weights)
    coords = arm.forward(posterior_params, only_end=False)
    plot_arms(coords, f"posterior_benchmark_weights={weights}")
