import pytest


@pytest.fixture
def benchmark_robot():
    from inverse_kinematics_sbi.benchmark_robot import BenchmarkRobot

    return BenchmarkRobot()
