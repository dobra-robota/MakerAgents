"""Run-folder creation: slugging, timestamps, and stub artifact writers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from makeragents.schemas import RunMetadata

_RUN_YAML_FILENAME = "run.yaml"
_FINAL_REPORT_FILENAME = "final-report.md"
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
_SLUG_FALLBACK = "run"


def slugify(value: str) -> str:
    """Slugify text to ASCII lowercase words joined by single hyphens.

    Non-ASCII characters (e.g. "Łodz") are transliterated where possible and
    otherwise dropped, so the result is always filesystem-safe.
    """

    normalized = unicodedata.normalize("NFKD", value)
    # "Ł" has no NFKD decomposition, so strip the stroke explicitly first.
    normalized = normalized.replace("Ł", "L").replace("ł", "l")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-") or _SLUG_FALLBACK


def build_run_metadata(
    *,
    city: str,
    community: str,
    max_opportunities: int = 5,
    output_dir: str = "runs",
) -> RunMetadata:
    """Build a :class:`RunMetadata` with a timestamped, slugged ``run_id``."""

    metadata = RunMetadata(
        run_id="placeholder",
        city=city,
        community=community,
        max_opportunities=max_opportunities,
        output_dir=output_dir,
    )
    timestamp = metadata.created_at.strftime(_TIMESTAMP_FORMAT)
    slug = slugify(f"{metadata.city} {metadata.community}")
    # model_copy(update=...) skips validation; round-trip through model_validate
    # so the derived run_id is still checked against the schema constraints.
    return RunMetadata.model_validate(
        metadata.model_dump() | {"run_id": f"{timestamp}-{slug}"}
    )


def create_run_folder(metadata: RunMetadata, *, base_dir: Path | None = None) -> Path:
    """Create the run folder and write the stub artifacts; return its path.

    ``base_dir`` overrides the working directory the ``output_dir`` is resolved
    against (primarily for tests). Defaults to the current working directory.
    """

    root = Path.cwd() if base_dir is None else base_dir
    runs_root = root / metadata.output_dir

    # Run IDs are only second-precision, so repeated invocations for the same
    # city/community within one second would collide. Claim a unique folder
    # atomically (mkdir exist_ok=False), appending -2, -3, ... on collision, so
    # repeated runs always succeed instead of raising FileExistsError while
    # still never overwriting a previous run's artifacts.
    attempt = 1
    while True:
        run_id = metadata.run_id if attempt == 1 else f"{metadata.run_id}-{attempt}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            attempt += 1

    if run_id != metadata.run_id:
        # Keep the written run_id consistent with the actual folder name.
        metadata = RunMetadata.model_validate(
            metadata.model_dump() | {"run_id": run_id}
        )

    _write_run_yaml(run_dir, metadata)
    _write_final_report(run_dir, metadata)
    return run_dir


def _write_run_yaml(run_dir: Path, metadata: RunMetadata) -> None:
    payload = {
        "run_id": metadata.run_id,
        "city": metadata.city,
        "community": metadata.community,
        "timestamp": metadata.created_at.isoformat(),
        "max_opportunities": metadata.max_opportunities,
    }
    (run_dir / _RUN_YAML_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_final_report(run_dir: Path, metadata: RunMetadata) -> None:
    report = (
        f"# Final report: {metadata.city} / {metadata.community}\n\n"
        f"- Run ID: `{metadata.run_id}`\n"
        f"- Created at: {metadata.created_at.isoformat()}\n"
        f"- Max opportunities: {metadata.max_opportunities}\n\n"
        "_This is a stub report produced by the walking skeleton. "
        "No research has been performed yet._\n"
    )
    (run_dir / _FINAL_REPORT_FILENAME).write_text(report, encoding="utf-8")
