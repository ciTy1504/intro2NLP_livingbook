"""Skill layer.

A skill is a reusable capability shared across agents. Skills are stateless and run
under the calling agent's permissions, which is what lets one implementation serve
agents with very different access.
"""

from .base import Skill
from .book import (
    BookPlacementSkill,
    BookRetrievalSkill,
    ChapterEditingSkill,
    CrossChapterConsistencySkill,
)
from .citation import (
    BibtexVerificationSkill,
    CitationAuditSkill,
    CitationSearchSkill,
    CitationVerificationSkill,
)
from .publishing import EmailReportingSkill, GitPublishingSkill
from .research import (
    ClaimExtractionSkill,
    CommunitySignalSkill,
    PaperAnalysisSkill,
    RepoAnalysisSkill,
    ResearchDiscoverySkill,
    ResearchSynthesisSkill,
)
from .verification import (
    BookQASkill,
    EditorialVerificationSkill,
    TechnicalVerificationSkill,
)
from .visual import (
    FigureGenerationSkill,
    FigureQASkill,
    VisualNeedDetectionSkill,
    VisualSearchSkill,
    VisualVerificationSkill,
)

#: name -> class, so agents resolve the skills named in config/agents.yaml.
SKILLS: dict[str, type[Skill]] = {
    "research_discovery": ResearchDiscoverySkill,
    "paper_analysis": PaperAnalysisSkill,
    "repo_analysis": RepoAnalysisSkill,
    "community_signal_analysis": CommunitySignalSkill,
    "claim_extraction": ClaimExtractionSkill,
    "research_synthesis": ResearchSynthesisSkill,
    "book_retrieval": BookRetrievalSkill,
    "book_placement": BookPlacementSkill,
    "chapter_editing": ChapterEditingSkill,
    "cross_chapter_consistency": CrossChapterConsistencySkill,
    "citation_audit": CitationAuditSkill,
    "citation_search": CitationSearchSkill,
    "citation_verification": CitationVerificationSkill,
    "bibtex_verification": BibtexVerificationSkill,
    "technical_verification": TechnicalVerificationSkill,
    "editorial_verification": EditorialVerificationSkill,
    "book_qa": BookQASkill,
    "visual_need_detection": VisualNeedDetectionSkill,
    "visual_search": VisualSearchSkill,
    "visual_verification": VisualVerificationSkill,
    "figure_generation": FigureGenerationSkill,
    "figure_qa": FigureQASkill,
    "git_publishing": GitPublishingSkill,
    "email_reporting": EmailReportingSkill,
}


def get_skill(name: str) -> Skill:
    if name not in SKILLS:
        raise KeyError(f"unknown skill {name!r}; known: {sorted(SKILLS)}")
    return SKILLS[name]()


__all__ = ["Skill", "SKILLS", "get_skill"] + sorted(
    {c.__name__ for c in SKILLS.values()}
)
