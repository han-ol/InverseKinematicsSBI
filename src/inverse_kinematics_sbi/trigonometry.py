import numpy as np


def so2_inverse(angle):
    return -angle


def so2_action(y, angle, axis=-1):
    # y np.array of shape (..., 2, ...), angle np.array of shape (...,)
    if axis < 0:
        axis = axis + y.ndim
    index_0 = (slice(None),) * axis + ([0],)
    index_1 = (slice(None),) * axis + ([1],)
    angle_index = (slice(None),) * axis + (None,)

    return np.concatenate(
        (
            np.cos(angle[angle_index])*y[index_0] - np.sin(angle[angle_index])*y[index_1],
            np.sin(angle[angle_index])*y[index_0] + np.cos(angle[angle_index])*y[index_1]
        ),
        axis=axis
    )


def se2_inverse(action):
    return np.concatenate((-so2_action(action[..., :2], angle=-action[..., 2]), -action[..., [2]]), axis=-1)


def se2_action(y, action):
    # y np.array of shape (..., 2), action np.array of shape (..., 3) trans_x, trans_y, rot
    return so2_action(y, angle=action[..., 2]) + action[..., :2]


def se2_compose(action1, action2):
    return np.stack(
        (np.cos(action1[..., 2])*action2[..., 0] - np.sin(action1[..., 2])*action2[..., 1] + action1[..., 0],
         np.sin(action1[..., 2])*action2[..., 0] + np.cos(action1[..., 2])*action2[..., 1] + action1[..., 1],
         action1[..., 2] + action2[..., 2]
         ), axis=-1)


def check_triangle(a, b, c):
    return (a <= b + c) & (b <= a + c) & (c <= a + b)

# all for simple joints and simple rails

def check_reachable_joint_joint(action_01_base, action_1e_base, target_0):
    length_01 = np.linalg.norm(action_01_base[..., :2], axis=-1)
    length_1e = np.linalg.norm(action_1e_base[..., :2], axis=-1)
    length_0e = np.linalg.norm(target_0[..., :2], axis=-1)
    return check_triangle(length_01, length_1e, length_0e)


def get_missing_params_joint_joint(action_01_base, action_1e_base, target_0, index):
    length_01 = np.linalg.norm(action_01_base[..., :2], axis=-1)
    length_1e = np.linalg.norm(action_1e_base[..., :2], axis=-1)
    length_0e = np.linalg.norm(target_0[..., :2], axis=-1)
    angle_01_0e_desired = (-1)**(index)*np.arccos((length_01**2 + length_0e**2 - length_1e**2)/(2*length_01*length_0e))
    angle_0_0e_desired = np.arctan2(target_0[..., 1], target_0[..., 0])
    angle_0_01_desired = angle_0_0e_desired - angle_01_0e_desired
    angle_0_01_base = np.arctan2(action_01_base[..., 1], action_01_base[..., 0])
    parameter_0 = angle_0_01_desired - angle_0_01_base # angle_0_base_0_desired

    target_1 = se2_action(so2_action(target_0, angle=so2_inverse(parameter_0)), action=se2_inverse(action_01_base))
    angle_1_1e_desired = np.arctan2(target_1[..., 1], target_1[..., 0])
    angle_1_1e_base = np.arctan2(action_1e_base[..., 1], action_1e_base[..., 0])
    parameter_1 = angle_1_1e_desired - angle_1_1e_base # angle_1_base_1_desired

    return parameter_0, parameter_1


def check_reachable_rail_joint(action_01_base, action_1e_base, target_0):
    orthogonal_to_rail_01_base = action_01_base[..., 1]
    length_1e = np.linalg.norm(action_1e_base[..., :2], axis=-1)
    orthogonal_to_rail_target_0 = target_0[..., 1]
    length_orthogonal = np.abs(orthogonal_to_rail_01_base - orthogonal_to_rail_target_0)
    return length_orthogonal <= length_1e


def get_missing_params_rail_joint(action_01_base, action_1e_base, target_0, index):
    orthogonal_to_rail_01_base = action_01_base[..., 1]
    orthogonal_to_rail_target_0 = target_0[..., 1]
    length_orthogonal = np.abs(orthogonal_to_rail_01_base - orthogonal_to_rail_target_0)
    x_trans_01_base = action_01_base[..., 0]
    length_trans_1e = np.linalg.norm(action_1e_base[..., :2], axis=-1)
    x_trans_0_frame_1e_desired = (-1)**(index)*np.sqrt(length_trans_1e**2 - length_orthogonal**2)
    x_trans_01_desired = target_0[..., 0] - x_trans_0_frame_1e_desired
    parameter_0 = x_trans_01_desired - x_trans_01_base

    target_1 = se2_action(np.stack((target_0[..., 0] - parameter_0, target_0[..., 1]), axis=-1), action=se2_inverse(action_01_base))
    angle_1_1e_desired = np.arctan2(target_1[..., 1], target_1[..., 0])
    angle_1_1e_base = np.arctan2(action_1e_base[..., 1], action_1e_base[..., 0])
    parameter_1 = angle_1_1e_desired - angle_1_1e_base  # angle_1_base_1_desired

    return parameter_0, parameter_1


def check_reachable_joint_rail(action_01_base, action_1e_base, target_0):
    y_trans_01_base_1 = so2_action(action_01_base[..., :2], angle=so2_inverse(action_01_base[..., 2]))[..., 1]
    y_trans_1e_base_1 = action_1e_base[..., 1]
    y_trans_0e_base_1 = y_trans_01_base_1 + y_trans_1e_base_1
    length_orthogonal = np.abs(y_trans_0e_base_1)
    length_0e = np.linalg.norm(target_0, axis=-1)
    return length_orthogonal <= length_0e


def get_missing_params_joint_rail(action_01_base, action_1e_base, target_0, index):
    y_trans_01_base_1 = so2_action(action_01_base[..., :2], angle=so2_inverse(action_01_base[..., 2]))[..., 1]
    y_trans_1e_base_1 = action_1e_base[..., 1]
    y_trans_0e_base_1 = y_trans_01_base_1 + y_trans_1e_base_1
    length_orthogonal = np.abs(y_trans_0e_base_1)
    length_0e = np.linalg.norm(target_0, axis=-1)
    angle_0e_1y_desired = (-1)**index*np.arccos(length_orthogonal/length_0e)
    angle_0e_1_desired = angle_0e_1y_desired - np.pi/2
    angle_0_0e = np.arctan2(target_0[..., 1], target_0[..., 0])
    angle_0_1_desired = angle_0_0e + angle_0e_1_desired
    angle_0_1_base = action_01_base[..., 2]
    parameter_0 = angle_0_1_desired - angle_0_1_base

    target_1 = se2_action(so2_action(target_0, angle=so2_inverse(parameter_0)), action=se2_inverse(action_01_base))
    parameter_1 = target_1[..., 0] - action_1e_base[..., 0]
    return parameter_0, parameter_1


def check_reachable_rail_rail(action_01_base, action_1e_base, target_0):
    base_vector_0 = np.zeros_like(action_01_base[..., :2])
    base_vector_0[..., 0] = 1
    base_vector_1 = so2_action(base_vector_0, angle=so2_inverse(action_01_base[..., 2]))
    transformation_mat = np.stack((base_vector_0, base_vector_1), axis=-1)
    return np.abs(np.linalg.det(transformation_mat)) > 0


def get_missing_params_rail_rail(action_01_base, action_1e_base, target_0, index):
    base_vector_0 = np.zeros_like(action_01_base[..., :2])
    base_vector_0[..., 0] = 1
    base_vector_1 = so2_action(base_vector_0, angle=action_01_base[..., 2])
    transformation_mat = np.stack((base_vector_0, base_vector_1), axis=-1)
    trans_1e_frame_0 = so2_action(action_1e_base[..., :2], angle=action_01_base[..., 2])
    trans_01_frame_0 = action_01_base[..., :2]
    lbmdas = np.linalg.solve(transformation_mat, target_0 - trans_01_frame_0 - trans_1e_frame_0)
    parameter_0 = lbmdas[..., 0]
    parameter_1 = lbmdas[..., 1]
    return parameter_0, parameter_1
