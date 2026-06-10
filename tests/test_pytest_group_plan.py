import tempfile
import unittest
from pathlib import Path

from tools.pytest_group_plan import plan


class PytestGroupPlanTest(unittest.TestCase):
    def test_groups_tests_by_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tests" / "quant").mkdir(parents=True)
            (root / "tests" / "quant" / "test_cast.py").write_text("", encoding="utf-8")

            report = plan(root)

        self.assertEqual(report["groups"][0]["name"], "quant")
        self.assertIn("pytest tests/quant/test_cast.py", report["groups"][0]["command"])

    def test_handles_missing_tests_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = plan(Path(tmpdir))

        self.assertEqual(report["group_count"], 0)
        self.assertEqual(report["groups"], [])


if __name__ == "__main__":
    unittest.main()
