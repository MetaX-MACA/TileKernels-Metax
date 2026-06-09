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


if __name__ == "__main__":
    unittest.main()
