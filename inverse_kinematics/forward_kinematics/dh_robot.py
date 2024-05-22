import torch

from inverse_kinematics.forward_kinematics.base_robot import Robot
from inverse_kinematics.forward_kinematics.utils import forward_dh


class DHRobot(Robot):
    """
    A robot parametrized by a DH table.
    See https://de.wikipedia.org/wiki/Denavit-Hartenberg-Transformation

    See also the dh_forward function in utils.

    ...

    Attributes
    ----------
    initial_dh_table : torch.Tensor of shape (n_joints, 4)
        The initial dh_table corresponding to the robot.

    parameter_indices : torch.Tensor of shape (n_params, 2), dtype=int
        The indices of the parameters of the dh table. (n_params <= n_joints*4)

    parameter_ranges : torch.Tensor of shape (n_params, 2)
        The parameter ranges that the parameters have.

    """

    def __init__(self, initial_dh_table, parameter_indices, parameter_ranges, rotation_form=None, precision=None):
        self.initial_dh_table = initial_dh_table
        self.parameter_indices = parameter_indices
        self.parameter_ranges = parameter_ranges
        self.rotation_form = rotation_form
        self.precision = precision

    def forward_kinematics(self, parameters, return_intermediates=False):
        positions, orientations = forward_dh(
            self._new_params(parameters),
            return_intermediates=return_intermediates
        )
        return positions, orientations

    def get_n_params(self):
        return len(self.parameter_indices)

    def get_param_ranges(self):
        return self.parameter_ranges

    def _new_params(self, params):
        if not torch.is_tensor(params):
            params = torch.from_numpy(params)
        self._check_params(params)

        indices = self.parameter_indices
        init_dh = self.initial_dh_table
        new_dh = torch.zeros((len(params),) + init_dh.shape, device=params.device)
        new_dh[:, :, :] = self.initial_dh_table[None, :, :]
        new_dh[:, indices[:, 0], indices[:, 1]] += params
        return new_dh

    def _check_params(self, parameters):

        if self.parameter_ranges is not None and self.parameter_ranges.shape[0] != self.parameter_indices.shape[0]:
            raise ValueError("parameter_ranges and parameter_indices must have the same first dimension.")

        if self.parameter_indices.shape[0] != parameters.shape[1]:
            raise ValueError("self.parameter_ranges.shape[0] and parameters.shape[1] must have the same length.")