import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from src.inverse_kinematics_sbi.plot import plot_arms
from src.inverse_kinematics_sbi.proba_robot import ProbaRobot
from src.inverse_kinematics_sbi.robots import RobotArm, Rail, Joint
from src.inverse_kinematics_sbi.solve_abc import draw_abc_samples, distance_end_effector_position

lengths = [0.5, 0.5, 1.0]
sigmas = [0.25, 0.5, 0.5, 0.5]

robot = RobotArm(components=[Rail(np.pi/2)] + [Joint(length=length) for length in lengths])
prior = norm(np.zeros_like(sigmas), sigmas)
prob_robot = ProbaRobot(robot, prior)
observations = np.array([[1.8, 0.0]])
n_sample = 50
target = observations[0]
parameters = prob_robot.sample_posterior(observations, n_sample=n_sample, verbose=True)
jacobians = robot.forward_jacobian(parameters)
post_prob = (np.prod(prior.pdf(parameters), axis=2)*np.linalg.det(np.matmul(jacobians, jacobians.transpose((0, 1, 3, 2))))**(-1/2))[0]
sorted_indices = np.argsort(post_prob)
coords = robot.forward(parameters, return_intermediates=True)[0, sorted_indices, 1:]
#parameters = draw_abc_samples(n_sample, robot.forward, lambda n_: prior.rvs((n_, robot.get_n_params())), distance_end_effector_position, target, 0.01, verbose=True)
plt.figure()
plt.xlim([0, 2])
plt.ylim([-1, 1])
plot_arms(coords, None, density=post_prob[sorted_indices])