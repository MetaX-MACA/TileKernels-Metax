import sys

from scripts.run_pytest_with_logs import build_command


def test_build_command_keeps_targets_before_extra_args():
    command = build_command(["tests/moe"], ["-q", "-k", "gate"])
    assert command == [sys.executable, "-m", "pytest", "tests/moe", "-q", "-k", "gate"]
