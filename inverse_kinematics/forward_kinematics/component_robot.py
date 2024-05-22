import roma
import torch

from inverse_kinematics.forward_kinematics.base_robot import Robot


class Rail(Robot):

    def __init__(self, direction_index=1, initial_value=0, parameter_range=None):
        self.direction_index = direction_index
        self.initial_value = initial_value
        self.parameter_range = parameter_range

    def forward_kinematics(self, params, return_intermediates=False):
        assert params.shape[-1] == 1

        positions = torch.zeros((len(params), 3), device=params.device, dtype=params.dtype)
        positions[:, [self.direction_index]] = self.initial_value + params

        # no orientation change
        orientations = torch.zeros((len(params), 4), device=params.device, dtype=params.dtype)
        orientations[:, 3] = 1

        if return_intermediates:
            return positions[:,  None, :], orientations[:, None, :]
        else:
            return positions, orientations

    def get_n_params(self):
        return 1

    def get_param_ranges(self):
        self._check_parameter_ranges()
        return self.parameter_range

    def _check_parameter_ranges(self):
        if self.parameter_range is None:
            self.parameter_range = torch.tensor([[-float('inf'), float('inf')]])
        elif self.parameter_range.ndim == 1:
            self.parameter_range = self.parameter_range.reshape(-1, 1)
        return self


class Joint(Robot):

    def __init__(self, length=1, rotation_index=2, initial_value=0, parameter_range=None):
        self.rotation_index = rotation_index
        self.length = length
        self.initial_value = initial_value
        self.parameter_range = parameter_range

    def forward_kinematics(self, params, return_intermediates=False):
        assert params.shape[-1] == 1

        positions_not_rotated = torch.zeros((len(params), 3), device=params.device, dtype=params.dtype)
        positions_not_rotated[:, 0] = self.length

        rotation_vec = torch.zeros((len(params), 3), device=params.device, dtype=params.dtype)
        rotation_vec[:, [self.rotation_index]] = self.initial_value + params
        orientations = roma.rotvec_to_unitquat(rotation_vec)

        positions_rotated = roma.quat_action(orientations, positions_not_rotated, True)

        if return_intermediates:
            return positions_rotated[:, None, :], orientations[:, None, :]
        else:
            return positions_rotated, orientations

    def get_n_params(self):
        return 1

    def get_param_ranges(self):
        self._check_parameter_ranges()
        return self.parameter_range

    def _check_parameter_ranges(self):
        if self.parameter_range is None:
            self.parameter_range = torch.tensor([[-torch.pi, torch.pi]])
        elif self.parameter_range.ndim == 1:
            self.parameter_range = self.parameter_range.reshape(-1, 1)
        return self


class OrthogonalPrismatic(Rail):

    def __init__(self, initial_value=0, parameter_range=None):
        super(OrthogonalPrismatic, self).__init__(initial_value=0, direction_index=1, parameter_range=parameter_range)


class CollinearPrismatic(Rail):

    def __init__(self, initial_value=0, parameter_range=None):
        super(CollinearPrismatic, self).__init__(initial_value=0, direction_index=0, parameter_range=parameter_range)


class RotationalJoint(Joint):

    def __init__(self, length=1, initial_value=0, parameter_range=None):
        super(RotationalJoint, self).__init__(rotation_index=2, parameter_range=parameter_range, initial_value=0, length=length)


class TwistingJoint(Joint):

    def __init__(self, length=1, initial_value=0, parameter_range=None):
        super(TwistingJoint, self).__init__(rotation_index=0, parameter_range=parameter_range, initial_value=0, length=length)