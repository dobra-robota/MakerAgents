"""Source trust registry: loading and trust-score resolution."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import Field, field_validator

from makeragents.schemas import MakerAgentsModel, ScoreValue, SourceType

PACKAGED_REGISTRY_RESOURCE = "source-registry.yaml"
"""Filename of the packaged default registry under ``makeragents.data``."""

RUN_REGISTRY_RELATIVE_PATH = Path("sources") / "source-registry.yaml"
"""Location of the effective registry inside a run folder."""


def _default_source_type_defaults() -> dict[SourceType, float]:
    return {
        SourceType.GOVERNMENT: 85,
        SourceType.ACADEMIC: 80,
        SourceType.MAJOR_NEWS: 70,
        SourceType.LOCAL_NEWS: 60,
        SourceType.NGO: 60,
        SourceType.COMPANY_OFFICIAL: 55,
        SourceType.FORUM: 40,
        SourceType.REDDIT: 35,
        SourceType.ANONYMOUS_SOCIAL: 20,
    }


class SourceRegistry(MakerAgentsModel):
    """Trust-score registry for sources, keyed by source type and domain."""

    default_unknown_domain_score: ScoreValue = 40
    source_type_defaults: dict[SourceType, ScoreValue] = Field(
        default_factory=_default_source_type_defaults
    )
    domains: dict[str, ScoreValue] = Field(default_factory=dict)

    @field_validator("domains", mode="after")
    @classmethod
    def _normalize_domain_keys(
        cls, value: dict[str, ScoreValue]
    ) -> dict[str, ScoreValue]:
        """Normalize ``domains`` keys at load/construction time.

        Lookups normalize the queried domain, so user-edited keys like
        ``WWW.Example.gov`` or a pasted URL must be normalized here too,
        otherwise they would never match and silently fall back.
        """

        return {_normalize_domain(key): score for key, score in value.items()}

    def score_for_type(self, source_type: SourceType | str) -> float:
        """Resolve a trust score from a source type alone.

        Unknown or unmapped source types fall back to the unknown-domain
        default score.
        """

        try:
            resolved = SourceType(source_type)
        except ValueError:
            return float(self.default_unknown_domain_score)
        score = self.source_type_defaults.get(resolved)
        if score is None:
            return float(self.default_unknown_domain_score)
        return float(score)

    def score_for_domain(
        self,
        domain: str,
        source_type: SourceType | str | None = None,
    ) -> float:
        """Resolve a trust score for a domain, optionally typed.

        Resolution order:

        1. An explicit per-domain score in ``domains``.
        2. The ``source_type_defaults`` score for ``source_type`` if given.
        3. ``default_unknown_domain_score``.
        """

        normalized = _normalize_domain(domain)
        if normalized in self.domains:
            return float(self.domains[normalized])
        if source_type is not None:
            return self.score_for_type(source_type)
        return float(self.default_unknown_domain_score)

    def persist_to_run(self, run_dir: Path | str) -> Path:
        """Write the effective registry into ``<run_dir>/sources/...``.

        Returns the path the registry was written to. Parent directories are
        created as needed.
        """

        destination = Path(run_dir) / RUN_REGISTRY_RELATIVE_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        return destination


def _normalize_domain(domain: str) -> str:
    """Lowercase a domain and strip scheme, path, and a leading ``www.``."""

    value = domain.strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0]
    if value.startswith("www."):
        value = value[len("www.") :]
    return value


def load_registry(path: Path | str | None = None) -> SourceRegistry:
    """Load a source registry from ``path`` or the packaged default.

    When ``path`` is provided, it is read from disk. Otherwise the registry
    packaged under ``makeragents.data`` is used.
    """

    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        resource = resources.files("makeragents.data").joinpath(
            PACKAGED_REGISTRY_RESOURCE
        )
        raw = resource.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return SourceRegistry.model_validate(data)
