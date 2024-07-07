import numpy as np
from matplotlib import pyplot as plt

from src.inverse_kinematics_sbi.benchmark_robot import BenchmarkRobot
from src.inverse_kinematics_sbi.plot import plot_arms

arm = BenchmarkRobot()

prior_params, coords = arm.sample(100, only_end=False)

plot_arms(coords, "prior_benchmark")

y = np.array([1.7, 0.0])

for weights in [[1, 1, 1]]:#[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]:
    print(weights)
    arm.get_maximum_weight(y, weights)
    posterior_params = arm.sample_posterior(y, 1000, weights=weights)
    coords = arm.forward(posterior_params, only_end=False)
    plot_arms(coords, f"posterior_benchmark_weights={weights}")
