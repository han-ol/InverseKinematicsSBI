from abc import ABC

import numpy as np

from src.inverse_kinematics_sbi.base_robots import Component, ConstComponent, SimpleRail, SimpleJoint, Robot
from src.inverse_kinematics_sbi.trigonometry import se2_compose, se2_action


class RobotArm(Robot):

    def __init__(self, components):
        self.components = components

    def check_components(self):
        new_components = []
        for i, component in enumerate(self.components):
            if isinstance(component, RobotArm):
                component.check_components()
                new_components = new_components + component.components
            elif isinstance(component, Component):
                new_components = new_components + [component]
            else:
                raise ValueError(
                    f"All components of `self.component` must be of type `RobotArm` or `Component` but component {i} is"
                    f"of type {type(component)}."
                )
        self.components = new_components

    def get_n_params(self):
        return sum(comp.get_n_params() for comp in self.components)

    def _get_n_params_intermediates(self):
        return list(np.cumsum([0] + [comp.get_n_params() for comp in self.components]))

    def forward_kinematics(self, params, return_intermediates=False):
        self.check_components()
        self.get_n_params()
        current_action = np.zeros(params.shape[:-1] + (3,))
        if return_intermediates:
            intermediate_actions = np.zeros(params.shape[:-1] + (1+self.get_n_params(), 3))

        indices = self._get_n_params_intermediates()

        for i, component in enumerate(self.components):
            if return_intermediates and indices[i+1] > indices[i]:
                intermediate_actions[..., indices[i], :] = current_action
            component_action = component.forward_kinematics(params[..., indices[i]:indices[i+1]])
            current_action = se2_compose(current_action, component_action)

        if return_intermediates:
            intermediate_actions[..., indices[len(self.components)], :] = current_action
            return intermediate_actions
        else:
            return current_action

    def _forward_kinematics_reduced(self, params, start_indices, end_indices):
        # call check components before
        current_action = np.zeros(params.shape[:-1] + (3,))
        indices = self._get_n_params_intermediates()
        for i, component in enumerate(self.components):
            is_turn = (start_indices <= indices[i]) & (indices[i+1] <= end_indices)
            component_action = component.forward_kinematics(params[..., indices[i]:indices[i+1]])
            current_action[is_turn] = se2_compose(current_action[is_turn], component_action[is_turn])
        return current_action


class Rail(RobotArm):

    def __init__(self, rail_angle=0):
        before = ConstComponent(action=np.array([0, 0, rail_angle]))
        rail_simple = SimpleRail()
        after = ConstComponent(action=np.array([0, 0, -rail_angle]))
        components = [before, rail_simple, after]
        super().__init__(components)


class Joint(RobotArm):

    def __init__(self, length=1, initial_angle=0):
        before = ConstComponent(action=np.array([0, 0, initial_angle]))
        joint_simple = SimpleJoint()
        after = ConstComponent(action=np.array([length, 0, 0]))
        components = [before, joint_simple, after]
        super().__init__(components)

