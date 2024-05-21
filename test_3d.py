import numpy as np
import scipy

from forward_kinematics import Rail, Joint, KinematicChain
from inverse_kinematics.solve_abc import draw_abc_samples, distance_end_effector_position
from utils.plot import plot_arms_3d

n_arms = 100
start_coord = np.zeros((n_arms, 3))
sigmas = np.array([[0.25, 0.25, 0.5, 0.5, 0.5]])


def prior(n_sample):
    return scipy.stats.norm(loc=np.zeros_like(sigmas), scale=sigmas).rvs(size=(n_sample, sigmas.shape[1]))


theta = prior(n_arms)

arm = KinematicChain([
    Rail(direction_index=1),
    Rail(direction_index=2),
    Joint(rotation_index=2, length=0.5),
    Joint(rotation_index=1, length=0.5),
    Joint(rotation_index=2, length=1.0)
])

coords = arm.forward(theta, return_intermediates=True)
plot_arms_3d(coords, 'prior_3d')

abc_samples = draw_abc_samples(n_arms, arm.forward, prior, distance_end_effector_position, [1.7, 0, 0], tolerance=0.02,
                               verbose=True)
coords = arm.forward(abc_samples, return_intermediates=True)
plot_arms_3d(coords[1:], 'abc_samples_3d')