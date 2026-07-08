#!/usr/bin/env python3
"""Create a JSON inventory of TileKernels test files by operator area."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_inventory(root: Path) -> dict[str, object]:
    groups: dict[str, list[str]] = defaultdict(list)
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return {"total": 0, "groups": {}}

    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(tests_dir).parts
        group = parts[0] if len(parts) > 1 else "root"
        groups[group].append(rel)
    return {
        "total": sum(len(items) for items in groups.values()),
        "groups": {name: {"count": len(items), "files": items} for name, items in sorted(groups.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="List TileKernels tests by area.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(build_inventory(args.root.resolve()), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
