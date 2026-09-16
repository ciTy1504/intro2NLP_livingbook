"""Agent layer.

Every agent's contract — capabilities, tools, skills, constraints, failure policy —
lives in config/agents.yaml. The classes here implement the behaviour; the config
decides what they are permitted to do.
"""

from .base import AgentDegraded, AgentFailure, BaseAgent, EscalateToHuman, validate_agent_registry
from .citation import BibtexValidator, CitationAuditor, CitationFinder, CitationVerifier
from .delivery import EmailAgent, GitAgent
from .research import (
    SOURCE_AGENTS,
    ArxivAgent,
    BenchmarkAgent,
    CommunityAgent,
    ConferenceAgent,
    GithubResearchAgent,
    HuggingFaceAgent,
    ResearchBlogAgent,
    ResearchSynthesizer,
    ScholarlyAgent,
)
from .verdict import BookVerdictAgent
from .verification import (
    BookQAAgent,
    CrossChapterConsistencyAgent,
    EditorialVerifier,
    TechnicalVerifier,
)
from .visual import (
    AssetManager,
    DiagramGeneratorAgent,
    FigureQAAgent,
    ImageSearchAgent,
    LicenseChecker,
    SemanticImageVerifier,
    VisualNeedDetector,
)
from .writer import WriterAgent

#: config name -> implementation. Validated against config/agents.yaml by `doctor`.
AGENTS: dict[str, type[BaseAgent]] = {
    "arxiv_agent": ArxivAgent,
    "scholarly_agent": ScholarlyAgent,
    "github_research_agent": GithubResearchAgent,
    "huggingface_agent": HuggingFaceAgent,
    "research_blog_agent": ResearchBlogAgent,
    "conference_agent": ConferenceAgent,
    "benchmark_agent": BenchmarkAgent,
    "community_agent": CommunityAgent,
    "research_synthesizer": ResearchSynthesizer,
    "book_verdict_agent": BookVerdictAgent,
    "writer_agent": WriterAgent,
    "citation_auditor": CitationAuditor,
    "citation_finder": CitationFinder,
    "citation_verifier": CitationVerifier,
    "bibtex_validator": BibtexValidator,
    "technical_verifier": TechnicalVerifier,
    "editorial_verifier": EditorialVerifier,
    "cross_chapter_consistency_agent": CrossChapterConsistencyAgent,
    "book_qa_agent": BookQAAgent,
    "visual_need_detector": VisualNeedDetector,
    "image_search_agent": ImageSearchAgent,
    "semantic_image_verifier": SemanticImageVerifier,
    "license_checker": LicenseChecker,
    "diagram_generator_agent": DiagramGeneratorAgent,
    "figure_qa_agent": FigureQAAgent,
    "asset_manager": AssetManager,
    "git_agent": GitAgent,
    "email_agent": EmailAgent,
}


def get_agent(name: str, **kwargs) -> BaseAgent:
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)


def validate_all() -> list[str]:
    return validate_agent_registry(AGENTS)


__all__ = [
    "BaseAgent", "AgentFailure", "AgentDegraded", "EscalateToHuman",
    "AGENTS", "get_agent", "validate_all", "SOURCE_AGENTS",
]
