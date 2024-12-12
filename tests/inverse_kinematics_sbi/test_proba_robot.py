import numpy as np
from scipy.stats import norm, uniform

from inverse_kinematics_sbi.proba_robot import ProbaRobot
from inverse_kinematics_sbi.robots import Joint, Rail, RobotArm


def test_proba_robot__get_reachable_params():
    robot = RobotArm(components=[Joint(), Joint(), Joint()])
    prior = None
    prob_robot = ProbaRobot(robot, prior)
    prob_robot.robot.check_components()
    params = np.array([[[0, 0, np.pi / 16], [0, 0, 0], [np.pi, 0, 0]]])
    indices = np.array([[[0, 1], [0, 2], [1, 2]]])
    target = np.array([[2.5, 0]])
    full_params, is_reachable = prob_robot._get_reachable_params(params, indices, target)
    target_obtained = robot.forward(full_params)
    np.testing.assert_array_almost_equal(is_reachable, np.array([[True, True, False]]))
    np.testing.assert_array_almost_equal(target_obtained[:, :2, :], np.tile(target[:, None, :], reps=(1, 2, 1)))


def test_proba_robot__sample_reachable_proposal_distribution():
    robot = RobotArm(components=[Joint(), Joint(), Joint()])
    prior = uniform(loc=-np.pi, scale=2 * np.pi)
    prob_robot = ProbaRobot(robot, prior)
    prob_robot.robot.check_components()
    observations = np.array([[2.5, 0], [1.5, 0]])
    parameters = prob_robot._sample_reachable_proposal_distribution(observations, n_sample=100, n_propose_total=100)
    target_obtained = robot.forward(parameters)
    np.testing.assert_array_almost_equal(target_obtained, np.tile(observations[:, None, :], reps=(1, 100, 1)))


def test_proba_robot___get_weights():
    robot = RobotArm(components=[Joint(), Joint(), Joint(), Joint()])
    prior = uniform(loc=-np.pi, scale=2 * np.pi)
    prob_robot = ProbaRobot(robot, prior)
    prob_robot.robot.check_components()
    arm = BenchmarkRobot()
    observations = np.array([[1.2, 0.3], [1.7, 0.1]])
    indices = np.array([[2, 3], [0, 3], [0, 1]])
    parameters = prob_robot._sample_reachable_proposal_distribution(
        observations, n_sample=1000, n_propose_total=1000, indices=indices
    )
    weights = prob_robot._get_weights(parameters)
    p = weights / np.max(weights, axis=-1, keepdims=True)
    accepted_mask = np.random.binomial(1, p) == 1
    first_obs_parameters = parameters[0, accepted_mask[0, :]]
    second_obs_parameters = parameters[1, accepted_mask[0, :]]


def test_proba_robot_sample_posterior():
    lengths = [0.5, 0.5, 1.0]
    sigmas = [0.25, 0.5, 0.5, 0.5]
    robot = RobotArm(components=[Rail(np.pi / 2)] + [Joint(length=length) for length in lengths])
    prior = norm(np.zeros_like(sigmas), sigmas)
    prob_robot = ProbaRobot(robot, prior, None)
    observations = np.array([[1.2, 0.3]])
    n_sample = 100000
    parameters = prob_robot.sample_posterior(observations, n_sample=n_sample)[0]
    first_mean = np.mean(parameters, axis=0)
    second_mean = np.array([-0.33562082, -0.46447627, 0.48865128, 1.08656594])
    print(second_mean)
    np.testing.assert_array_almost_equal(first_mean, second_mean, decimal=1)
