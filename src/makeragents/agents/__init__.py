"""Agent implementations for the MakerAgents research pipeline."""

from makeragents.agents.evidence import ConflictResult, EvidenceAgent
from makeragents.agents.opportunity import OpportunityAgent
from makeragents.agents.maker import MakerAgent, MakerResult
from makeragents.agents.taker import TakerAgent, TakerOutput

__all__ = [ConflictResult, EvidenceAgent, OpportunityAgent, MakerAgent, MakerResult, TakerAgent, TakerOutput]
