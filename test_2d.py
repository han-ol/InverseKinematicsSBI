from inverse_kinematics.forward_kinematics import Rail, Joint, KinematicChain
from inverse_kinematics.plot import plot_arms_2d
from inverse_kinematics.solve_abc import draw_abc_samples, distance_end_effector_position

import scipy
import numpy as np

n_arms = 100
start_coord = np.zeros((n_arms, 3))
sigmas = np.array([[0.25, 0.5, 0.5, 0.5]])
def prior(n_sample):
    return scipy.stats.norm(loc=np.zeros_like(sigmas), scale=sigmas).rvs(size=(n_sample, sigmas.shape[1]))


theta = prior(n_arms)

arm = KinematicChain([
    Rail(),
    Joint(0.5),
    Joint(0.5),
    Joint(1.0)
])

coords = arm.forward_2d(theta, return_intermediates=True)
plot_arms_2d(coords, 'prior_2d')

abc_samples = draw_abc_samples(n_arms, arm.forward_2d, prior, distance_end_effector_position, [1.7, 0], tolerance=0.01,
                               verbose=True)
coords = arm.forward_2d(abc_samples, return_intermediates=True)
plot_arms_2d(coords, 'abc_samples_2d')
