import roma
import torch


def affine_compo(translation_second, rotation_second, translation_first, rotation_first):
    translation_first_second = translation_second + roma.quat_action(rotation_second, translation_first, True)
    rotation_first_second = roma.quat_product(rotation_second, rotation_first)
    return translation_first_second, rotation_first_second


def forward_dh(dh_tables, return_intermediates=False):
    """
    A function implementing the forward process of the Denavit and Hartenberg convention.

    Dh table:
             | trans z | rot z | trans x | rot x
    ---------|---------|-------|---------|-------
    joint 0-1|   d_1   | th_1  |  a_1    |  al_1
    joint 1-2|   ...   | ...   |  ...    |  ...
    ...      |         |       |         |

    trans z: translation along axis z_{i-1}
    rot z: rotation from x_{i-1} to x_{i} along axis z_{i-1}
    trans x: translation along axis x_{i}
    rot x: rotation from z_{i-1] to z_{i} along axis x_{i}

    https://de.wikipedia.org/wiki/Denavit-Hartenberg-Transformation

    Parameters
    ----------
    dh_tables : torch.Tensor of shape (n_samples, n_joints, 4)
        The dh tables of which the forward kinematics are computed, i.e., dh_tables[i] is the i-th dh table of which the
        forward kinematics are computed.
    return_intermediates : bool
        If true the positions and orientations for the intermediate stages are also returned.

    Returns
    -------
    positions : torch.Tensor of shape (n_samples, 3) or (n_joints, n_samples, 3)
        The position of the end effector in the base frame.
    orientations : torch.Tensor of shape (n_samples, 4) or (n_joints, n_samples, 4)
        The orientation of the end effector in the base frame.
    """
    n_samples = dh_tables.shape[0]
    n_joints = dh_tables.shape[1]

    trans_0_to_i = torch.zeros((n_samples, 3), device=dh_tables.device)
    rot_0_to_i = roma.rotvec_to_unitquat(torch.zeros((n_samples, 3), device=dh_tables.device))

    if return_intermediates:
        intermediate_trans = torch.zeros((n_joints, n_samples, 3), device=dh_tables.device)
        intermediate_rot = torch.zeros((n_joints, n_samples, 4), device=dh_tables.device)

    for i in range(n_joints):
        trans_z = dh_tables[:, i, 0].reshape(-1, 1) * torch.tensor([[0, 0, 1]], device=dh_tables.device)
        rot_z = roma.rotvec_to_unitquat(dh_tables[:, i, 1].reshape(-1, 1) * torch.tensor([[0, 0, 1]], device=dh_tables.device))
        trans_x = dh_tables[:, i, 2].reshape(-1, 1) * torch.tensor([[1, 0, 0]], device=dh_tables.device)
        rot_x = roma.rotvec_to_unitquat(dh_tables[:, i, 3].reshape(-1, 1) * torch.tensor([[1, 0, 0]], device=dh_tables.device))
        trans_i_to_i_plus, rot_i_to_i_plus = affine_compo(trans_z, rot_z, trans_x, rot_x)
        trans_0_to_i, rot_0_to_i = affine_compo(trans_0_to_i, rot_0_to_i, trans_i_to_i_plus, rot_i_to_i_plus)

        if return_intermediates:
            intermediate_trans[i] = trans_0_to_i
            intermediate_rot[i] = rot_0_to_i

    if return_intermediates:
        return intermediate_trans, intermediate_rot
    return trans_0_to_i, rot_0_to_i
