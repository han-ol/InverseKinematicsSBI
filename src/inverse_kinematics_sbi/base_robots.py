from abc import ABC, abstractmethod

import numpy as np

from .trigonometry import se2_action


class Robot(ABC):
    """
    Base Class for all robots.

    """

    @abstractmethod
    def forward_kinematics(self, params, return_intermediates=False):
        """Calculates the forward kinematics of the robot.

        Parameters
        ----------
        params : np.array of shape (..., n_params)
            The parameters for the robot.
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
        assert params.shape[-1] == self.get_n_params()
        return params

    def check_params_start_position(self, params, start_position):
        self.check_params(params)
        if start_position is None:
            start_position = np.zeros(params.shape[:-1] + (2,))
        else:
            assert isinstance(start_position, np.ndarray)
            assert params.shape[:-1] == start_position.shape[:-1]
            assert start_position.shape[-1] == 2
        return params, start_position

    def forward(self, params, start_position=None, return_intermediates=False):
        """Calculates the position of a point in the end effector frame as seen in the base frame.
        Thereby, the origin gets mapped to position of the end effector.

        Parameters
        ----------
        params : np.array of shape (..., n_params)
            The parameters for the robot.
        start_position: np.array of shape (..., 2)
            The start position that gets moved forward by the component.
        return_intermediates : True
            Whether to return intermediate positions.

        Returns
        -------
        end_position : np.array of shape (..., 3) or of shape (..., n_params + 1, 3)
            The final position of the end effector.
            If return_intermediates returns the n_params + 1 intermediate positions.
        """
        params, start_position = self.check_params_start_position(params, start_position)
        if not return_intermediates:
            return se2_action(start_position, self.forward_kinematics(params, return_intermediates=False))
        else:
            return se2_action(start_position[..., None, :], self.forward_kinematics(params, return_intermediates=True))

    @abstractmethod
    def forward_jacobian(self, params, start_position=None):
        """Calculates the jacobian of the forward function with respect to the parameters.

        Parameters
        ----------
        params : np.array of shape (..., n_params)
            The parameters for the robot.
        start_position: np.array of shape (..., 2)
            The start position that gets moved forward by the robot.

        Returns
        -------
        jacobian : np.array of shape (..., 2, n_params)
            The jacobian of the forward function.
        """
        pass


class Component(Robot):
    """
    Base Class for all components.

    """

    pass


class SimpleRail(Component):
    def get_n_params(self):
        return 1

    def forward_kinematics(self, params, return_intermediates=False):
        params = self.check_params(params)

        action = np.zeros(params.shape[:-1] + (3,))
        action[..., [0]] = params

        if return_intermediates:
            return np.stack((np.zeros_like(action), action), axis=-2)
        else:
            return action

    def forward_jacobian(self, params, start_position=None):
        params, start_position = self.check_params_start_position(params, start_position)
        derivative = np.zeros(params.shape[:-1] + (2, self.get_n_params()))
        derivative[..., 0, 0] = 1
        return derivative


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

    def forward_jacobian(self, params, start_position=None):
        params, start_position = self.check_params_start_position(params, start_position)
        derivative = np.zeros(params.shape[:-1] + (2, self.get_n_params()))
        end_position = self.forward(params, start_position)
        derivative[..., 0, 0] = -end_position[..., 1]
        derivative[..., 1, 0] = end_position[..., 0]
        return derivative


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

    def forward_jacobian(self, params, start_position=None):
        self.check_params(params)

        derivative = np.zeros(params.shape[:-1] + (2, self.get_n_params()))

        return derivative
