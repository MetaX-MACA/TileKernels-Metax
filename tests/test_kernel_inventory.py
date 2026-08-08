import tempfile
import unittest
from pathlib import Path

from tools.kernel_inventory import inventory


class KernelInventoryTest(unittest.TestCase):
    def test_collects_kernel_and_area_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tile_kernels" / "moe").mkdir(parents=True)
            (root / "tests" / "moe").mkdir(parents=True)
            (root / "tile_kernels" / "moe" / "gate_kernel.py").write_text("", encoding="utf-8")
            (root / "tests" / "moe" / "test_gate.py").write_text("", encoding="utf-8")

            report = inventory(root)

        self.assertEqual(report["kernel_count"], 1)
        self.assertEqual(report["kernels"][0]["area"], "moe")


if __name__ == "__main__":
    unittest.main()
