from abc import ABC, abstractmethod

import roma
import torch

from inverse_kinematics.forward_kinematics.utils import affine_compo


class Robot(ABC):
    """
    Base Class for all robots to describe the forward kinematics.

    """

    @abstractmethod
    def forward_kinematics(self, params, return_intermediates=False):
        """Calculates the forward kinematics for the robot.

        Parameters
        ----------
        params : torch.Tensor of shape (n_samples, n_params)
            The parameters for the robot.
        return_intermediates : bool
            If true the positions and orientations for the intermediate stages are also returned.

        Returns
        -------
        positions : torch.Tensor of shape (n_samples, 3) or (n_samples, n_stages, 3)
            The position of the end effector in the base frame for each parameter.
            If return_intermediates is true also returns intermediate position.
        orientation : torch.Tensor of shape (n_samples, 4) or (n_samples, n_stages, 4)
            The orientation of the end effector in the base frame for each parameter expressed as a quaternion.
            If return_intermediates is true also returns intermediate orientations.
        """
        pass

    def forward(self, params, rotation_form=None, return_intermediates=False):
        """Calculates the forward kinematics for the robot and encodes the final pose into one vector.

        Parameters
        ----------
        params : torch.Tensor or np.array of shape (n_samples, n_params)
            The parameters for the robot. Also accepts numpy arrays. In this case
            a numpy array is returned.
        rotation_form : str (default None)
            The rotation formalism to use for the forward kinematics,
            where `rotation_form=None` means just the position is returned. If rotation_form is not None,
            the pose is the position with the rotation formalism appended as (3 + d_rot) vector.
            "quaternion" uses the quaternion representation as a 4D vector (d_rot=4).
            "vector" uses the rotation vector representation as a 3D vector (d_rot=3).
            "matrix" uses the flatted matrix itself as a 9D vector (d_rot=9).
            None uses no rotation representation (d_rot=0).
        return_intermediates : bool (default False)
            If true the poses for the intermediate stages are also returned.

        Returns
        -------
        poses : torch.Tensor or np.array of shape (n_samples, 3 + d_rot) or (n_samples, n_stages, 3 + d_rot)
            the pose of the end effector from the base frame for each parameter.
            If return_intermediates is true also returns the pose of every joint.
            If a `params` is a numpy array, then a numpy array is returned.
        """
        is_numpy = not torch.is_tensor(params)
        if is_numpy:
            params = torch.from_numpy(params)

        positions, orientations = self.forward_kinematics(params, return_intermediates)

        if rotation_form is None:
            pose = positions
        elif rotation_form == "quaternion":
            pose = torch.concat((positions, orientations), dim=-1)
        elif rotation_form == "vector":
            rot_vecs = roma.unitquat_to_rotvec(orientations)
            pose = torch.concat((positions, rot_vecs), dim=-1)
        elif rotation_form == "matrix":
            rot_mats_flat = roma.unitquat_to_rotmat(orientations).flatten(start_dim=-2)
            pose = torch.concat((positions, rot_mats_flat), dim=-1)
        else:
            raise ValueError(f"rotation_form must take one of {[None, 'quaternion', 'vector', 'matrix']}")

        if is_numpy:
            return pose.detach().numpy()
        else:
            return pose

    def forward_2d(self, params, return_angle=False, return_intermediates=False):
        """Calculates the forward kinematics for the robot and encodes the final pose into one vector.

        Parameters
        ----------
        params : torch.Tensor of shape (n_samples, n_params)
            The parameters indicated by parameter indices.
        return_angle : bool (default False)
            If true the angles around z is appended to the pose.
        return_intermediates : bool (default False)
            If true the poses for the intermediate stages are also returned.

        Returns
        -------
        poses : torch.Tensor of shape (n_samples, 2+d_rot) or (n_samples, n_stages, 2+d_rot)
            the pose of the end effector from the base frame for each parameter.
            If return_intermediates is true also returns the pose of every joint.
        """
        is_numpy = not torch.is_tensor(params)
        if is_numpy:
            params = torch.from_numpy(params)

        positions, orientations = self.forward_kinematics(params, return_intermediates)

        if return_angle:
            rot_vecs = roma.unitquat_to_rotvec(orientations, shortest_arc=True)[..., 2]
            pose = torch.concat((positions[..., :2], rot_vecs), dim=-1)
        else:
            pose = positions[..., :2]

        if is_numpy:
            return pose.detach().numpy()
        else:
            return pose

    @abstractmethod
    def get_n_params(self):
        """Returns the number of parameters that the robot has.

        Returns
        -------
        n_params : int
            The number of parameters.
        """
        pass

    @abstractmethod
    def get_param_ranges(self):
        """Return a tensor containing the minimum and the maximum value a parameter can take.

        Returns
        -------
        param_ranges : (n_params, 2)
            The tensor containing the minimum and the maximum value a parameter can take,
            where `param_ranges[i, 0]` is the minimum and `param_ranges[i, 1]` is the maximum value
            for the i-th parameter.
        """
        pass


class KinematicChain(Robot):
    """
    Wrapper class to encode a kinematic chain of robots.

    Attributes
    ----------
    components : list[Robot]
        A list of robots to define the kinematic chain.

    """

    def __init__(self, components):
        self.components = components

    def forward_kinematics(self, params, return_intermediates=False):
        trans_total = torch.zeros((len(params), 3), device=params.device, dtype=params.dtype)
        rot_total = roma.rotvec_to_unitquat(torch.zeros((len(params), 3), device=params.device, dtype=params.dtype))
        param_index = 0

        if return_intermediates:
            trans_intermediate = torch.zeros((len(params), 0, 3), device=params.device, dtype=params.dtype)
            rot_intermediate = torch.zeros((len(params), 0, 4), device=params.device, dtype=params.dtype)

        for robot in self.components:
            trans_new, rot_new = robot.forward_kinematics(
                params[:, param_index:param_index + robot.get_n_params()],
                return_intermediates
            )
            if return_intermediates:
                trans_new_base, rot_new_base = affine_compo(trans_total[None, ...], rot_total[None, ...], trans_new, rot_new)
                trans_intermediate = torch.cat((trans_intermediate, trans_new_base), dim=1)
                rot_intermediate = torch.cat((rot_intermediate, rot_new_base), dim=1)
                trans_total = trans_new_base[:, -1, :]
                rot_total = rot_new_base[:, -1, :]
            else:
                trans_total, rot_total = affine_compo(trans_total, rot_total, trans_new, rot_new)
            param_index += robot.get_n_params()

        if return_intermediates:
            return trans_intermediate, rot_intermediate
        else:
            return trans_total, rot_total

    def get_n_params(self):
        return sum((robot.get_n_params() for robot in self.components))

    def get_param_ranges(self):
        return torch.stack(tuple(robot.get_n_params() for robot in self.components), dim=0)


class ConstRobot(Robot):
    """
    Class to encode a robot that returns a constant position and orientation.

    Attributes
    ----------
    robot : Robot
        The robot used to compute the constant value.
    param_value : torch.Tensor or float (default 0)
        The values of the parameter, that leads to the constant value.
    """

    def __init__(self, robot, param_value=None):
        self.robot = robot
        self.param_value = param_value

    def forward_kinematics(self, params, return_intermediates=False):

        param_value = self.param_value
        if param_value is None:
            param_value = 0
        if not torch.is_tensor(param_value):
            param_value = torch.full(self.robot.get_n_params(), param_value)
        new_params = param_value.repeat(len(params), 1)

        return self.robot.forward_kinematics(new_params, return_intermediates)

    def get_n_params(self):
        return 0

    def get_param_ranges(self):
        return torch.zeros((0, 2))
