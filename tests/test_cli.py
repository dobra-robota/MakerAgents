from pathlib import Path

import yaml

from tests.conftest import _invoke_in


def test_run_command_creates_run_folder(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Łodz",
        "--community",
        "senior citizens",
    )

    assert result.exit_code == 0, result.output

    runs_root = tmp_path / "runs"
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("-lodz-senior-citizens")

    assert (run_dir / "final-report.md").is_file()
    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["city"] == "Łodz"
    assert parsed["community"] == "senior citizens"
    assert parsed["max_opportunities"] == 5
    assert "timestamp" in parsed
    assert parsed["run_id"] == run_dir.name


def test_run_command_honors_max_opportunities(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Berlin",
        "--community",
        "cyclists",
        "--max-opportunities",
        "8",
    )

    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").iterdir())
    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["max_opportunities"] == 8


def test_run_command_rejects_invalid_max_opportunities(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Berlin",
        "--community",
        "cyclists",
        "--max-opportunities",
        "0",
    )

    assert result.exit_code != 0
