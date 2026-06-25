from pathlib import Path

import yaml

from makeragents.run import build_run_metadata, create_run_folder, slugify


def test_slugify_handles_non_ascii_and_separators() -> None:
    assert slugify("Łodz senior citizens") == "lodz-senior-citizens"
    assert slugify("  São Paulo / Youth  ") == "sao-paulo-youth"
    assert slugify("Kraków") == "krakow"


def test_slugify_falls_back_when_no_ascii_remains() -> None:
    assert slugify("日本語") == "run"


def test_build_run_metadata_timestamps_and_slugs_run_id() -> None:
    metadata = build_run_metadata(city="Łodz", community="senior citizens")

    assert metadata.city == "Łodz"
    assert metadata.community == "senior citizens"
    assert metadata.max_opportunities == 5
    assert metadata.run_id.endswith("-lodz-senior-citizens")
    # Timestamp prefix is YYYYMMDD-HHMMSS (15 chars) before the slug.
    prefix = metadata.run_id[:15]
    assert prefix.replace("-", "").isdigit()


def test_create_run_folder_writes_artifacts(tmp_path: Path) -> None:
    metadata = build_run_metadata(
        city="Łodz", community="senior citizens", max_opportunities=3
    )
    run_dir = create_run_folder(metadata, base_dir=tmp_path)

    assert run_dir == tmp_path / "runs" / metadata.run_id
    assert run_dir.is_dir()

    report = run_dir / "final-report.md"
    assert report.is_file()
    assert "Final report" in report.read_text(encoding="utf-8")

    run_yaml = run_dir / "run.yaml"
    assert run_yaml.is_file()
    parsed = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    assert parsed["run_id"] == metadata.run_id
    assert parsed["city"] == "Łodz"
    assert parsed["community"] == "senior citizens"
    assert parsed["max_opportunities"] == 3
    assert parsed["timestamp"] == metadata.created_at.isoformat()


def test_create_run_folder_suffixes_on_collision(tmp_path: Path) -> None:
    metadata = build_run_metadata(city="Łodz", community="senior citizens")

    first = create_run_folder(metadata, base_dir=tmp_path)
    second = create_run_folder(metadata, base_dir=tmp_path)

    # A same-second collision gets a numeric suffix instead of a traceback.
    assert first == tmp_path / "runs" / metadata.run_id
    assert second == tmp_path / "runs" / f"{metadata.run_id}-2"
    assert first.is_dir() and second.is_dir()

    # The written run_id matches the actual (suffixed) folder name.
    parsed = yaml.safe_load((second / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["run_id"] == f"{metadata.run_id}-2"
