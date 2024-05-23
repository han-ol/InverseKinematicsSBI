from .components import Joint, Rail, RobotArm


def create_benchmark_robot():
    rail, joint_1, joint_2, joint_3 = Rail(0.25), Joint(0.5, 0.5), Joint(0.5, 0.5), Joint(1.0, 0.5)
    arm = RobotArm((rail, joint_1, joint_2, joint_3))
    return arm
