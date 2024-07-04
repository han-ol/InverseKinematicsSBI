from abc import ABC, abstractmethod

import numpy as np

from src.inverse_kinematics_sbi.trigonometry import se2_action


class Component(ABC):
    """
    Base Class for all robot components to build a kinematic chain.

    """

    @abstractmethod
    def forward_kinematics(self, params):
        """Calculates the forward kinematics of the component.

        Parameters
        ----------
        params : np.array of shape (...,)
            The parameters for the component.

        Returns
        -------
        action : np.array of shape (..., 3)
            The action that the components performs, i.e. the se2 action that describes the
            transformation between two frames.
        """
        pass

    def check_params(self, params):
        assert isinstance(params, np.ndarray)

    def forward(self, params, start_position=None):
        """Calculates the forward movement of the component for a start_position.

        Parameters
        ----------
        params : np.array of shape (...,)
            The parameters for the component.
        start_position: np.array of shape (..., 2)
            The start position that gets moved forward by the component.

        Returns
        -------
        end_position : np.array of shape (..., 2)
            The position after being moved forward.
        """
        self.check_params(params)
        if start_position is None:
            start_position = np.zeros(params.shape + (2,))
        assert isinstance(start_position, np.ndarray)
        assert params.shape == start_position[:-1]
        return se2_action(start_position, self.forward_kinematics(params))


class SimpleRail(Component):

    def forward_kinematics(self, params):
        self.check_params(params)

        action = np.zeros(params.shape + (3,))
        action[..., 0] = params

        return action


class SimpleJoint(Component):

    def forward_kinematics(self, params):
        self.check_params(params)

        action = np.zeros(params.shape + (3,))
        action[..., 2] = params

        return action


class ConstComponent(Component):

    def __init__(self, action):
        self.action = action

    def forward_kinematics(self, params):
        self.check_params(params)

        action = np.zeros(params.shape + (3,))
        action[..., :] = self.action

        return action

