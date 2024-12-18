import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from src.inverse_kinematics_sbi.plot import plot_arms
from src.inverse_kinematics_sbi.proba_robot import ProbaRobot
from src.inverse_kinematics_sbi.robots import Joint, Rail, RobotArm
from src.inverse_kinematics_sbi.solve_abc import distance_end_effector_position, draw_abc_samples

lengths = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
sigmas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

robot = RobotArm(components=[Joint(length=length) for length in lengths])
prior = norm(np.zeros_like(sigmas), sigmas)
prob_robot = ProbaRobot(robot, prior)
observations = np.array([[2, 0]])
n_sample = 1000
time_start = time.time()
parameters = prob_robot.sample_posterior(observations, n_sample=n_sample, verbose=True)
print("time", time.time() - time_start)
jacobians = robot.forward_jacobian(parameters)
post_prob = (
    np.prod(prior.pdf(parameters), axis=2)
    * np.linalg.det(np.matmul(jacobians, jacobians.transpose((0, 1, 3, 2)))) ** (-1 / 2)
)[0]
sorted_indices = np.argsort(post_prob)
coords = robot.forward(parameters, return_intermediates=True)[0, sorted_indices, :]
plt.figure()
plot_arms(coords, None, density=post_prob[sorted_indices])
target = observations[0]

parameters = draw_abc_samples(
    n_sample,
    robot.forward,
    lambda n_: prior.rvs((n_, robot.get_n_params())),
    distance_end_effector_position,
    target,
    0.01,
    verbose=True,
)
