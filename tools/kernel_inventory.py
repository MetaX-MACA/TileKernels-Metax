#!/usr/bin/env python3
"""Export TileKernels kernel modules and associated test files as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inventory(root: Path) -> dict[str, object]:
    kernels = []
    for path in sorted(root.joinpath("tile_kernels").rglob("*_kernel.py")):
        area = path.relative_to(root / "tile_kernels").parts[0]
        test_dir = root / "tests" / area
        tests = sorted(item.relative_to(root).as_posix() for item in test_dir.glob("test_*.py")) if test_dir.exists() else []
        kernels.append({"path": path.relative_to(root).as_posix(), "area": area, "tests": tests})
    return {"kernel_count": len(kernels), "kernels": kernels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(inventory(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
