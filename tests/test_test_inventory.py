from pathlib import Path

from scripts.test_inventory import build_inventory


def test_build_inventory_groups_nested_tests(tmp_path: Path):
    test_dir = tmp_path / "tests" / "moe"
    test_dir.mkdir(parents=True)
    (test_dir / "test_gate.py").write_text("", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    assert inventory["total"] == 1
    assert inventory["groups"]["moe"]["files"] == ["tests/moe/test_gate.py"]
