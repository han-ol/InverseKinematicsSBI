import numpy as np

from inverse_kinematics_sbi.base_robots import ConstComponent, SimpleJoint, SimpleRail


def test_simple_rail_forward():
    params = np.array([[k / 4] for k in range(4)])

    simple_rail = SimpleRail()

    actual_action = simple_rail.forward_kinematics(params)
    desired_action = np.array([[0, 0, 0], [1 / 4, 0, 0], [2 / 4, 0, 0], [3 / 4, 0, 0]])
    np.testing.assert_allclose(actual_action, desired_action)


def test_simple_joint_forward():
    params = np.array([[2 * np.pi * k / 4] for k in range(4)])

    simple_joint = SimpleJoint()

    actual_action = simple_joint.forward_kinematics(params)
    desired_action = np.array([[0, 0, 0], [0, 0, 1 / 2 * np.pi], [0, 0, np.pi], [0, 0, 3 / 2 * np.pi]])
    np.testing.assert_allclose(actual_action, desired_action)


def test_const_component_forward():
    base_action = np.array([2, 1, np.pi / 4])
    params = np.zeros((4, 0))

    const_component = ConstComponent(base_action)

    actual_action = const_component.forward_kinematics(params)
    desired_action = np.array([[2, 1, np.pi / 4], [2, 1, np.pi / 4], [2, 1, np.pi / 4], [2, 1, np.pi / 4]])
    np.testing.assert_allclose(actual_action, desired_action)
