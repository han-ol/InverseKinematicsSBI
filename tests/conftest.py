import pytest


@pytest.fixture
def benchmark_robot():
    from inverse_kinematics.benchmark_robot import BenchmarkRobot

    return BenchmarkRobot()
