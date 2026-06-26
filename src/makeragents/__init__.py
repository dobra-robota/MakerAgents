"""MakerAgents package foundation."""

from makeragents.agents.report import ReportAgent
from makeragents.cli import app
from makeragents.config import AppConfig, load_config
from makeragents.overrides import (
    UserOverride,
    apply_score_overrides,
    apply_trust_overrides,
    apply_verdict_override,
    format_override_markdown,
    has_overrides,
    parse_overrides,
    strip_overrides,
)
from makeragents.retry import (
    PIPELINE_STEPS,
    get_incomplete_steps,
    mark_steps_complete,
    read_opportunity_state,
    read_status,
    write_status,
)
from makeragents.run import build_run_metadata, create_run_folder, slugify
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    POCType,
    RunMetadata,
    ScoreSet,
    SourceType,
    Verdict,
)
from makeragents.scoring import compute_low_harm_score, compute_rank_score
from makeragents.search import ProviderResponse, SearchClient, SearchResult
from makeragents.sources import SourceRegistry, load_registry

__all__ = [
    "AppConfig",
    "ClaimClassification",
    "Confidence",
    "EvidenceItem",
    "EvidenceType",
    "Opportunity",
    "OpportunityType",
    "POCType",
    "ProviderResponse",
    "ReportAgent",
    "RunMetadata",
    "ScoreSet",
    "SearchClient",
    "SearchResult",
    "SourceRegistry",
    "SourceType",
    "UserOverride",
    "Verdict",
    "app",
    "apply_score_overrides",
    "apply_trust_overrides",
    "apply_verdict_override",
    "build_run_metadata",
    "compute_low_harm_score",
    "compute_rank_score",
    "create_run_folder",
    "format_override_markdown",
    "get_incomplete_steps",
    "has_overrides",
    "load_config",
    "load_registry",
    "mark_steps_complete",
    "parse_overrides",
    "read_opportunity_state",
    "read_status",
    "slugify",
    "strip_overrides",
    "write_status",
]

__version__ = "0.1.0"
