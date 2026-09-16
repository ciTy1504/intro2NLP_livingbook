"""Source-specific research agents plus the synthesizer."""

from .sources import (
    ArxivAgent,
    BenchmarkAgent,
    CommunityAgent,
    ConferenceAgent,
    GithubResearchAgent,
    HuggingFaceAgent,
    ResearchBlogAgent,
    ScholarlyAgent,
    SourceAgent,
)
from .synthesizer import ResearchSynthesizer

#: The eight source agents, in the order the orchestrator fans them out.
SOURCE_AGENTS = [
    ArxivAgent, ScholarlyAgent, ConferenceAgent, GithubResearchAgent,
    HuggingFaceAgent, BenchmarkAgent, ResearchBlogAgent, CommunityAgent,
]

__all__ = [
    "SourceAgent", "ArxivAgent", "ScholarlyAgent", "GithubResearchAgent",
    "HuggingFaceAgent", "ResearchBlogAgent", "ConferenceAgent", "BenchmarkAgent",
    "CommunityAgent", "ResearchSynthesizer", "SOURCE_AGENTS",
]
