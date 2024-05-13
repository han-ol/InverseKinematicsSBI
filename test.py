from inverse_kinematics.forward import Rail, Joint, RobotArm
from inverse_kinematics.plot import plot_arms 
from inverse_kinematics.solve_abc import draw_abc_samples, distance_end_effector_position
import scipy
import numpy as np

n_arms = 1000
start_coord = np.zeros((n_arms,3))  #np.random.normal(size=(5,3))

sigmas = [0.25, 0.5, 0.5, 0.5]
prior = dict(
    c1 = scipy.stats.norm(0, sigmas[0]),
    c2 = scipy.stats.norm(0, sigmas[1]),
    c3 = scipy.stats.norm(0, sigmas[2]),
    c4 = scipy.stats.norm(0, sigmas[3]),
)
# theta = sample_from_distdict(prior, n_samples=n_arms)
theta = {name: distribution.rvs(n_arms) for name, distribution in prior.items()}

r1 = Rail()
j1 = Joint(0.5)
j2 = Joint(0.5)
j3 = Joint(1.0)
arm = RobotArm(dict(
    c1 = r1,
    c2 = j1,
    c3 = j2,
    c4 = j3,
))

coords = arm.forward(theta, only_end=False)
print(coords.shape)
plot_arms(coords, 'prior')

abc_samples = draw_abc_samples(1000, arm.forward, prior, distance_end_effector_position, [1.7,0], tolerance=0.05, verbose=True)
plot_arms(arm.forward(abc_samples, only_end=False), 'abc_samples')


