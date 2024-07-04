import sys
sys.path.append('C:\\Users\\lukas\\PycharmProjects\\inversekinematicssbi_base\\InverseKinematicsSBI')
from functools import reduce

import numpy as np
import pytest

from src.inverse_kinematics_sbi.base_robots import SimpleRail, SimpleJoint, ConstComponent, Component
from src.inverse_kinematics_sbi.robots import RobotArm, Rail, Joint


def test_robot_arm_check_components():
    rail = SimpleRail()
    joint = SimpleJoint()
    const = ConstComponent(action=np.array([1, 0, 0]))

    inner_components = [joint, const]
    components = [rail, RobotArm(inner_components)]

    robot_arm = RobotArm(components=components)
    robot_arm.check_components()

    for component_1, component_2 in zip(robot_arm.components, [rail, joint, const]):
        assert component_1 == component_2


def test_robot_arm_get_n_params():
    rail = SimpleRail()
    joint = SimpleJoint()
    const = ConstComponent(action=np.array([1, 0, 0]))

    inner_components = [joint, const]
    components = [rail, RobotArm(inner_components)]

    robot_arm = RobotArm(components=components)
    n_params_actual = robot_arm.get_n_params()
    n_params_desired = 2
    assert n_params_actual == n_params_desired


def test_robot_arm_forward_kinematics():
    params = np.array([[0, 2 * np.pi * k / 4] for k in range(4)])

    robot_arm = RobotArm([SimpleRail(), SimpleJoint(), ConstComponent(action=np.array([1, 0, 0]))])
    actual_action = robot_arm.forward_kinematics(params)
    desired_action = np.array([[np.cos(2 * np.pi * k / 4), np.sin(2 * np.pi * k / 4), 2 * np.pi * k / 4] for k in range(4)])
    np.testing.assert_array_equal(actual_action, desired_action)


def test_rail_forward():
    params = np.array([[k] for k in range(4)])
    rail = Rail(np.pi/2)
    actual_position = rail.forward(params)
    desired_position = np.array([[0, k] for k in range(4)])
    np.testing.assert_allclose(actual_position, desired_position, atol=1e-10)


def test_joint_forward():
    params = np.array([[2*np.pi*k/4] for k in range(2)])
    rail = Joint(length=1)
    actual_action = rail.forward_kinematics(params)
    desired_action = np.array([[1, 0, 0], [0, 1, np.pi/2]])
    np.testing.assert_allclose(actual_action, desired_action, atol=1e-10)


def test_robot_arm_forward_reduced():
    robot_arm = RobotArm(components=[Rail(), Joint(), Joint()])
    robot_arm_0 = RobotArm(components=[Rail(), Joint()])
    robot_arm_1 = RobotArm(components=[Joint(), Joint()])
    params = np.array([[1, np.pi/8, np.pi/8], [-1, -np.pi/8, -np.pi/8]])
    start_indices = np.array([0, 1])
    end_indices = np.array([2, 3])
    actual_action = robot_arm._forward_kinematics_reduced(params, start_indices, end_indices)
    desired_action_0 = robot_arm_0.forward_kinematics(params[:, 0:2])
    desired_action_1 = robot_arm_1.forward_kinematics(params[:, 1:3])
    np.testing.assert_allclose(actual_action[0], desired_action_0[0], atol=1e-10)
    np.testing.assert_allclose(actual_action[1], desired_action_1[1], atol=1e-10)



