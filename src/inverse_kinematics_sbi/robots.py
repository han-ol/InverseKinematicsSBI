from abc import ABC

import numpy as np

from src.inverse_kinematics_sbi.components import Component, ConstComponent, SimpleRail, SimpleJoint
from src.inverse_kinematics_sbi.trigonometry import se2_compose, se2_action


class RobotArm(ABC):

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

    def get_n_params(self):
        return sum(not isinstance(comp, ConstComponent) for comp in self.components)

    def forward_kinematics(self, params, return_intermediates=False):
        self.check_components()
        current_action = np.zeros(params.shape[:-1] + (3,))
        if return_intermediates:
            intermediate_actions = np.zeros(params.shape[:-1] + (1+self.get_n_params(), 3))
        current_index = 0

        for component in self.components:
            if return_intermediates and not isinstance(component, ConstComponent):
                intermediate_actions[..., current_index, :] = current_action
            component_action = component.forward_kinematics(params[..., current_index])
            current_action = se2_compose(current_action, component_action)
            if not isinstance(component, ConstComponent):
                current_index += 1

        if return_intermediates:
            intermediate_actions[..., current_index, :] = current_action
            return intermediate_actions
        else:
            return current_action

    def forward(self, params, start_positions=None, return_intermediates=False):
        if start_positions is None:
            start_positions = np.zeros(params.shape[:-1] + (2,))
        if return_intermediates:
            return se2_action(start_positions[..., None, :], self.forward_kinematics(params, return_intermediates=True))
        else:
            return se2_action(start_positions[..., :], self.forward_kinematics(params, return_intermediates=False))

    def _forward_kinematics_reduced(self, params, start_indices, end_indices):
        # call check components before
        current_action = np.zeros(params.shape[:-1] + (3,))
        current_index = 0
        for component in self.components:
            is_turn = (start_indices <= current_index) & (current_index < end_indices)
            component_action = component.forward_kinematics(params[..., current_index])
            current_action[is_turn] = se2_compose(current_action[is_turn], component_action[is_turn])
            if not isinstance(component, ConstComponent):
                current_index += 1
        return current_action # composed reduced action


class Rail(RobotArm):
    # implement rail as components of const and simplerail

    def __init__(self, rail_angle):
        before = ConstComponent(action=np.array([0, 0, rail_angle]))
        rail_simple = SimpleRail()
        after = ConstComponent(action=np.array([0, 0, -rail_angle]))
        components = [before, rail_simple, after]
        super().__init__(components)


class Joint(RobotArm):
    # implement joint as components of const and simplejoint

    def __init__(self, length, initial_angle):
        before = ConstComponent(action=np.array([length, 0, initial_angle]))
        joint_simple = SimpleJoint()
        components = [before, joint_simple]
        super().__init__(components)

