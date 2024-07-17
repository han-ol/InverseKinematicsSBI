from functools import reduce

import numpy as np

from src.inverse_kinematics_sbi.trigonometry import check_reachable_joint_joint, get_missing_params_joint_joint, \
    se2_action, check_reachable_rail_joint, get_missing_params_rail_joint, check_reachable_joint_rail, \
    get_missing_params_joint_rail, check_reachable_rail_rail, get_missing_params_rail_rail


def test_joint_joint():
    action_01_base = np.array([[1, 1, np.pi/4]])
    action_1e_base = np.array([[0, 1, np.pi/2]])
    target_0 = np.array([[1, 0.5]])

    is_reachable = check_reachable_joint_joint(action_01_base, action_1e_base, target_0)
    assert is_reachable
    action_pairs = []
    for index in [np.array(0), np.array(1)]:
        param_1, param_2 = get_missing_params_joint_joint(action_01_base, action_1e_base, target_0, index)
        action_0b_0d = np.array([[0, 0, param_1.item()]])
        action_1b_1d = np.array([[0, 0, param_2.item()]])
        action_pairs.append(np.concatenate((action_0b_0d, action_1b_1d), axis=0))
        position = reduce(se2_action, [action_1e_base, action_1b_1d, action_01_base, action_0b_0d], np.zeros(2))
        assert np.allclose(position, target_0)
    assert not np.allclose(action_pairs[0], action_pairs[1])


def test_rail_joint():
    action_01_base = np.array([[1, 1, np.pi/8]])
    action_1e_base = np.array([[0, 1, np.pi/2]])
    target_0 = np.array([[1, 0.5]])

    is_reachable = check_reachable_rail_joint(action_01_base, action_1e_base, target_0)
    assert is_reachable
    action_pairs = []
    for index in [np.array(0), np.array(1)]:
        param_1, param_2 = get_missing_params_rail_joint(action_01_base, action_1e_base, target_0, index)
        action_0b_0d = np.array([[param_1.item(), 0, 0]])
        action_1b_1d = np.array([[0, 0, param_2.item()]])
        action_pairs.append(np.concatenate((action_0b_0d, action_1b_1d), axis=0))
        position = reduce(se2_action, [action_1e_base, action_1b_1d, action_01_base, action_0b_0d], np.zeros(2))
        assert np.allclose(position, target_0)
    assert not np.allclose(action_pairs[0], action_pairs[1])


def test_joint_rail():
    action_01_base = np.array([[1, 1, np.pi/8]])
    action_1e_base = np.array([[0, 1, np.pi/2]])
    target_0 = np.array([[3, 0.5]])

    is_reachable = check_reachable_joint_rail(action_01_base, action_1e_base, target_0)
    assert is_reachable
    action_pairs = []
    for index in [np.array(0), np.array(1)]:
        param_1, param_2 = get_missing_params_joint_rail(action_01_base, action_1e_base, target_0, index)
        action_0b_0d = np.array([[0, 0, param_1.item()]])
        action_1b_1d = np.array([[param_2.item(), 0, 0]])
        action_pairs.append(np.concatenate((action_0b_0d, action_1b_1d), axis=0))
        position = reduce(se2_action, [action_1e_base, action_1b_1d, action_01_base, action_0b_0d], np.zeros(2))
        assert np.allclose(position, target_0)
    assert not np.allclose(action_pairs[0], action_pairs[1])


def test_rail_rail():
    action_01_base = np.array([[1, 1, np.pi/4]])
    action_1e_base = np.array([[0, 1, np.pi/2]])
    target_0 = np.array([[1, 0.5]])

    is_reachable = check_reachable_rail_rail(action_01_base, action_1e_base, target_0)
    assert is_reachable

    param_1, param_2 = get_missing_params_rail_rail(action_01_base, action_1e_base, target_0)
    action_0b_0d = np.array([[param_1.item(), 0, 0]])
    action_1b_1d = np.array([[param_2.item(), 0, 0]])
    position = reduce(se2_action, [action_1e_base, action_1b_1d, action_01_base, action_0b_0d], np.zeros(2))
    assert np.allclose(position, target_0)