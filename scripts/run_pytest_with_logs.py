#!/usr/bin/env python3
"""Run TileKernels pytest targets and capture structured logs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_command(targets: list[str], extra: list[str]) -> list[str]:
    return [sys.executable, "-m", "pytest", *targets, *extra]


def run(root: Path, targets: list[str], log_root: Path, extra: list[str]) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(targets, extra)
    (run_dir / "metadata.json").write_text(
        json.dumps({"command": command, "root": str(root), "targets": targets}, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=root, stdout=out, stderr=err, text=True)
    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    print(f"pytest logs written to: {run_dir}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TileKernels pytest with logs.")
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--log-root", type=Path, default=Path("pytest_logs"))
    args, extra = parser.parse_known_args()
    return run(args.root.resolve(), args.targets, args.log_root.resolve(), extra)


if __name__ == "__main__":
    raise SystemExit(main())
