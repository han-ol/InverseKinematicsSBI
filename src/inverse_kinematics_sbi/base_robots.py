from abc import ABC, abstractmethod

import numpy as np

from src.inverse_kinematics_sbi.trigonometry import se2_action


class Robot(ABC):
    """
    Base Class for all robots.

    """

    @abstractmethod
    def forward_kinematics(self, params, return_intermediates=False):
        """Calculates the forward kinematics of the component.

        Parameters
        ----------
        params : np.array of shape (..., n_params)
            The parameters for the component.
        return_intermediates : True
            Whether to return intermediate actions.

        Returns
        -------
        action : np.array of shape (..., 3) or of shape (..., n_params + 1, 3)
            The action that the components performs, i.e. the se2 action that describes the
            transformation between two frames.
            If return_intermediates returns the n_params + 1 intermediate actions.
        """
        pass

    @abstractmethod
    def get_n_params(self):
        pass

    def check_params(self, params):
        assert isinstance(params, np.ndarray)

    def forward(self, params, start_position=None):
        """Calculates the forward movement of the component for a start_position.

        Parameters
        ----------
        params : np.array of shape (..., n_params)
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
            start_position = np.zeros(params.shape[:-1] + (2,))
        assert isinstance(start_position, np.ndarray)
        assert params.shape[:-1] == start_position.shape[:-1]
        return se2_action(start_position, self.forward_kinematics(params))


class Component(Robot):
    """
    Base Class for all components.

    """
    pass


class SimpleRail(Component):

    def get_n_params(self):
        return 1

    def forward_kinematics(self, params, return_intermediates=False):
        self.check_params(params)

        action = np.zeros(params.shape[:-1] + (3,))
        action[..., [0]] = params

        if return_intermediates:
            return np.stack((np.zeros_like(action), action), axis=-2)
        else:
            return action


class SimpleJoint(Component):

    def get_n_params(self):
        return 1

    def forward_kinematics(self, params, return_intermediates=False):
        self.check_params(params)

        action = np.zeros(params.shape[:-1] + (3,))
        action[..., [2]] = params

        if return_intermediates:
            return np.stack((np.zeros_like(action), action), axis=-2)
        else:
            return action


class ConstComponent(Component):

    def __init__(self, action):
        self.action = action

    def get_n_params(self):
        return 0

    def forward_kinematics(self, params, return_intermediates=False):
        self.check_params(params)

        action = np.zeros(params.shape[:-1] + (3,))
        action[..., :] = self.action

        if return_intermediates:
            return np.stack((action,), axis=-2)
        else:
            return action

