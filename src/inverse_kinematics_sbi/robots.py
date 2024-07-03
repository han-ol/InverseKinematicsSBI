from abc import ABC

import numpy as np


class RobotArm(ABC):

    def __init__(self, components):
        self.components = components

    def get_n_params(self):
        return  # number of non const components

    def forward_kinematics(self, params):
        return  # composed action

    def forward_kinematics_reduced(self, params, start_indices, end_indices):
        return  # composed reduced action

    def forward(self, params, start_positions=None):
        return # end_positions


class Rail(RobotArm):
    pass  # implement rail as components of const and simplerail


class Joint(RobotArm):
    pass  # implement joint as components of const and simplejoint

