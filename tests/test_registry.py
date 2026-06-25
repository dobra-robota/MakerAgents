from pathlib import Path

import pytest
import yaml

from makeragents.schemas import SourceType
from makeragents.sources import SourceRegistry, load_registry
from makeragents.sources.registry import RUN_REGISTRY_RELATIVE_PATH


def test_load_packaged_registry_defaults() -> None:
    registry = load_registry()

    assert registry.default_unknown_domain_score == 40
    assert registry.source_type_defaults[SourceType.GOVERNMENT] == 85
    assert registry.source_type_defaults[SourceType.ANONYMOUS_SOCIAL] == 20


def test_score_for_type_resolves_known_and_unknown() -> None:
    registry = load_registry()

    assert registry.score_for_type("government") == 85
    assert registry.score_for_type(SourceType.REDDIT) == 35
    # UNKNOWN is not in the defaults table -> falls back to default score.
    assert registry.score_for_type(SourceType.UNKNOWN) == 40
    # A string that is not a valid source type also falls back.
    assert registry.score_for_type("not-a-real-type") == 40


def test_score_for_unknown_domain_uses_default() -> None:
    registry = load_registry()

    assert registry.score_for_domain("totally-unknown-domain.example") == 40


def test_score_for_domain_prefers_explicit_domain_over_type() -> None:
    registry = SourceRegistry(domains={"example.gov": 99})

    # Explicit domain wins even when a (lower) source type is supplied.
    assert registry.score_for_domain("example.gov", SourceType.FORUM) == 99
    # Domain normalization: scheme and www are stripped, case-insensitive.
    assert registry.score_for_domain("https://WWW.example.gov/path") == 99


def test_score_for_domain_falls_back_to_type_then_default() -> None:
    registry = load_registry()

    assert registry.score_for_domain("unlisted.example", SourceType.ACADEMIC) == 80
    assert registry.score_for_domain("unlisted.example") == 40


def test_domain_keys_are_normalized_on_construction() -> None:
    registry = SourceRegistry(domains={"WWW.Example.GOV": 90})

    # The stored key is normalized so lookups (which also normalize) match.
    assert registry.domains == {"example.gov": 90}
    assert registry.score_for_domain("example.gov") == 90
    assert registry.score_for_domain("https://WWW.Example.GOV/path") == 90


def test_domain_keys_are_normalized_on_load(tmp_path: Path) -> None:
    custom = tmp_path / "source-registry.yaml"
    custom.write_text(
        yaml.safe_dump({"domains": {"HTTPS://WWW.Trusted.Example/path": 88}}),
        encoding="utf-8",
    )

    registry = load_registry(custom)

    assert registry.score_for_domain("trusted.example") == 88


def test_load_registry_from_user_file(tmp_path: Path) -> None:
    custom = tmp_path / "source-registry.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "default_unknown_domain_score": 10,
                "source_type_defaults": {"government": 50},
                "domains": {"trusted.example": 90},
            }
        ),
        encoding="utf-8",
    )

    registry = load_registry(custom)

    assert registry.default_unknown_domain_score == 10
    assert registry.score_for_type("government") == 50
    assert registry.score_for_domain("trusted.example") == 90


def test_persist_to_run_writes_round_trippable_registry(tmp_path: Path) -> None:
    registry = load_registry()
    run_dir = tmp_path / "20260625-test-run"

    written = registry.persist_to_run(run_dir)

    assert written == run_dir / RUN_REGISTRY_RELATIVE_PATH
    assert written.exists()

    reloaded = load_registry(written)
    assert reloaded.model_dump() == registry.model_dump()


def test_registry_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        SourceRegistry.model_validate({"unexpected_field": 1})
