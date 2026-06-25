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

__all__ = [
    "AppConfig",
    "ClaimClassification",
    "Confidence",
    "EvidenceItem",
    "EvidenceType",
    "Opportunity",
    "OpportunityType",
    "POCType",
    "RunMetadata",
    "ScoreSet",
    "SourceType",
    "Verdict",
    "app",
    "build_run_metadata",
    "create_run_folder",
    "load_config",
    "slugify",
]

__version__ = "0.1.0"
