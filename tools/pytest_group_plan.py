#!/usr/bin/env python3
"""Generate grouped pytest commands for TileKernels validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def plan(root: Path) -> dict[str, object]:
    groups: dict[str, list[str]] = defaultdict(list)
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return {"group_count": 0, "groups": []}

    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(tests_dir).parts
        group = parts[0] if len(parts) > 1 else "root"
        groups[group].append(rel)
    return {
        "group_count": len(groups),
        "groups": [
            {"name": name, "count": len(files), "command": "pytest " + " ".join(files), "files": files}
            for name, files in sorted(groups.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(plan(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
