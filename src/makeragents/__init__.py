"""MakerAgents package foundation."""

from makeragents.cli import app
from makeragents.config import AppConfig, load_config
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
    "RunMetadata",
    "ScoreSet",
    "SearchClient",
    "SearchResult",
    "SourceRegistry",
    "SourceType",
    "Verdict",
    "app",
    "build_run_metadata",
    "compute_low_harm_score",
    "compute_rank_score",
    "create_run_folder",
    "load_config",
    "load_registry",
    "slugify",
]

__version__ = "0.1.0"
