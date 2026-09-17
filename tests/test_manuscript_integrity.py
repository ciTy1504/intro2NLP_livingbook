"""Manuscript integrity checks.

These run in CI on every pull request the agents open, so a change that breaks a
cross-reference, orphans a citation or removes a figure fails before a human reviews
it. They deliberately do not require LaTeX to be installed — the build itself is a
separate, slower CI job.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.config import get_config  # noqa: E402
from livingbook.knowledge.bib import Bibliography  # noqa: E402
from livingbook.tools.book import bib_validate, check_figures, latex_lint  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def test_latex_is_structurally_sound():
    result = asyncio.run(latex_lint())
    errors = [p for p in result["problems"] if p["severity"] == "error"]
    assert not errors, "LaTeX structure errors:\n  " + "\n  ".join(
        f"{p['file']}: {p['kind']} — {p['detail']}" for p in errors)


def test_no_orphan_citations():
    result = asyncio.run(bib_validate())
    assert not result["orphan_citations"], (
        f"cited but absent from references.bib: {result['orphan_citations']}")


def test_bibliography_has_no_blocking_errors():
    result = asyncio.run(bib_validate())
    errors = [p for p in result["problems"] if p["severity"] == "error"]
    assert not errors, "bibliography errors:\n  " + "\n  ".join(
        f"{p.get('key')}: {p['kind']} — {p['detail']}" for p in errors)


def test_every_referenced_figure_exists():
    result = asyncio.run(check_figures())
    assert not result["missing"], "figures referenced but not present:\n  " + "\n  ".join(
        f"{m['key']} in {m['file']}:{m['line']}" for m in result["missing"])


def test_all_cross_references_resolve(cfg):
    """A \\ref pointing at a label that does not exist renders as '??' in the PDF."""
    from livingbook.knowledge.latex import LatexParser, collect_labels

    parser = LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex"))
    book = parser.parse()

    labels: set[str] = set()
    for rel in book.files:
        path = cfg.manuscript_dir / rel
        if path.exists():
            # collect_labels understands the tcolorbox theorem environments defined in
            # style.tex, whose labels are `prefix:key` rather than a bare \label.
            labels |= collect_labels(
                path.read_text(encoding="utf-8", errors="replace"))

    dangling = {label for n in book.nodes for _, label in n.refs} - labels
    assert not dangling, f"cross-references to non-existent labels: {sorted(dangling)}"


def test_bibliography_entries_are_well_formed(cfg):
    bib = Bibliography(cfg.bib_path)
    problems = [p for p in bib.validate() if p["severity"] == "error"]
    assert not problems, "malformed bibliography entries:\n  " + "\n  ".join(
        f"{p['key']}: {p['kind']}" for p in problems)


def test_main_tex_inputs_all_exist(cfg):
    from livingbook.knowledge.latex import LatexParser

    parser = LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex"))
    missing = [rel for rel in parser.input_order()
               if not (cfg.manuscript_dir / rel).exists()]
    assert not missing, f"\\input targets that do not exist: {missing}"


def test_no_placeholder_citations_committed(cfg):
    """`\\cite{NEEDS_CITATION}` is the Writer's placeholder; it must never ship."""
    import re

    offenders: list[str] = []
    for path in cfg.manuscript_dir.rglob("*.tex"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"\\cite\{([^}]*NEEDS_CITATION[^}]*)\}", text):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert not offenders, "unresolved citation placeholders:\n  " + "\n  ".join(offenders)


def test_no_secrets_in_manuscript(cfg):
    """Cheap insurance: nothing credential-shaped should ever reach the book."""
    import re

    patterns = [re.compile(p) for p in (
        r"AIzaSy[A-Za-z0-9_\-]{20,}", r"AQ\.Ab8[A-Za-z0-9_\-]{20,}",
        r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}")]
    offenders: list[str] = []
    for path in list(cfg.manuscript_dir.rglob("*.tex")) + [cfg.bib_path]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern.search(text):
                offenders.append(path.name)
    assert not offenders, f"credential-shaped strings found in: {offenders}"


# -- link checking -----------------------------------------------------------
#
# The link report is only useful if a line in it means something is wrong. Two
# classes of false positive made it useless in practice, and both are regressions
# worth guarding: example URLs printed in code listings, and hosts that refuse an
# automated probe on a URL that opens fine in a browser.

def test_example_urls_are_not_treated_as_links():
    from livingbook.tools.book import _is_illustrative

    # The vLLM server a reader starts in chapter 2.6.
    assert _is_illustrative("http://localhost:8000/v1")
    assert _is_illustrative("http://127.0.0.1:8080/generate")
    assert _is_illustrative("https://api.example.com/v1/chat")
    assert _is_illustrative("http://192.168.1.10:11434/api")

    assert not _is_illustrative("https://aclanthology.org/D19-1410")
    assert not _is_illustrative("https://doi.org/10.18653/v1/2023.findings-emnlp.655")


def test_bot_refusal_is_not_reported_as_a_broken_link():
    from livingbook.tools.book import _is_bot_refusal
    from livingbook.tools.registry import ToolUnavailable

    # Wikipedia and openai.com 403 an automated HEAD; aclanthology.org and doi.org
    # drop the connection. None of these mean the citation is dead.
    assert _is_bot_refusal(ToolUnavailable("en.wikipedia.org: HTTP 403 (blocked)"))
    assert _is_bot_refusal(ToolUnavailable(
        "doi.org: RemoteProtocolError: Server disconnected without sending a response."))

    # A genuinely missing page still has to be reported.
    assert not _is_bot_refusal(ToolUnavailable("example.org: HTTP 404"))
    assert not _is_bot_refusal(None)


# -- the book's own bibliography ---------------------------------------------
#
# The citation finder searched five web sources and never looked in references.bib.
# For a claim about material the book already covers, the primary source is often
# already cited — and going to the web instead returns a third-party restatement,
# which the verifier then rejects as needs_primary, which loops the pipeline.

def test_the_bibliography_is_searched_for_a_claim_about_what_the_book_covers():
    import asyncio

    from livingbook.tools.book import search_bibliography

    # The exact claim that sent the finder to the web and looped the pipeline.
    hits = asyncio.run(search_bibliography(
        query="DeepSeek-V3 reports 85-90% acceptance for the extra predicted token"))
    assert hits, "the bibliography search returned nothing"
    assert hits[0]["bib_key"] == "deepseekai2024v3", (
        f"the DeepSeek-V3 Technical Report should rank first, got {hits[0]['bib_key']}")
    assert hits[0]["already_in_bibliography"] is True

    # And it finds the foundational papers for a topic, not merely word matches.
    spec = asyncio.run(search_bibliography(
        query="speculative decoding draft model acceptance rate"))
    assert {"leviathan2023speculative", "chen2023accelerating"} & {
        h["bib_key"] for h in spec[:3]}


def test_a_common_word_does_not_match_a_longer_one():
    """Substring matching ranked the right entry joint-fourth: "extra" hit
    "extracting", "extracted" and "extraction" across unrelated papers."""
    import asyncio

    from livingbook.tools.book import search_bibliography

    hits = asyncio.run(search_bibliography(query="extra predicted token acceptance"))
    keys = {h["bib_key"] for h in hits}
    assert "vincent2008extracting" not in keys, (
        "'extra' must not match 'extracting' — word boundaries, not substrings")


def test_the_citation_finder_is_allowed_and_asked_to_use_it():
    """Wiring: granting the tool is not the same as calling it."""
    import inspect

    import yaml

    from livingbook.skills.citation.audit import CitationSearchSkill

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "agents.yaml").read_text(
            encoding="utf-8"))
    tools = cfg["agents"]["citation_finder"]["tools"]
    assert "search_bibliography" in tools
    assert tools.index("search_bibliography") == 0, "the local bibliography comes first"

    src = inspect.getsource(CitationSearchSkill.run)
    assert "search_bibliography" in src, "the skill must actually call it"
    assert src.index("search_bibliography") < src.index("search_openalex"), (
        "the bibliography must be consulted before the web")


def test_an_existing_key_survives_onto_the_candidate():
    """Finding the entry is not enough — the key has to reach the validator.

    Without it, a source the book already cites goes back through entry creation and
    produces a near-duplicate of an entry a few lines away in the same file.
    """
    import inspect

    from livingbook.skills.citation.audit import CitationSearchSkill

    src = inspect.getsource(CitationSearchSkill._rank)
    assert "bib_key=" in src, "CitationCandidate must be given the existing key"
    assert "ALREADY CITED IN THIS BOOK" in src, (
        "the ranker cannot prefer an already-cited source it cannot see")


def test_arxiv_ids_are_recovered_from_wherever_the_entry_keeps_them():
    """One of 228 entries has a DOI. Without the arXiv id there is no identifier."""
    import asyncio

    from livingbook.tools.book import search_bibliography

    hits = asyncio.run(search_bibliography(
        query="speculative decoding T5-XXL 2-3x speedup"))
    by_key = {h["bib_key"]: h for h in hits}
    # leviathan2023speculative keeps it in note={arXiv:2211.17192}, not a doi field.
    assert by_key["leviathan2023speculative"]["arxiv_id"] == "2211.17192"


def test_a_source_already_in_the_book_is_not_re_identified_externally():
    """Provenance settles identity; external resolution actively got this wrong.

    The verifier resolved Leviathan et al. (ICML 2023) — an entry sitting in
    references.bib — to an unrelated 2026 paper via a Crossref title search, set
    identity_confirmed=False, and rejected a correct primary citation.
    """
    import inspect

    from livingbook.skills.citation.audit import CitationVerificationSkill

    src = inspect.getsource(CitationVerificationSkill)
    assert "from_book" in src
    assert 'if not d.get("identity_confirmed") and not from_book:' in src, (
        "an entry the author already vetted must not be rejected on identity")
    # But the support check must still run: being in the bibliography does not mean
    # it establishes this particular claim.
    assert 'if d.get("supports_claim") in ("does_not_support", "unverifiable"):' in src
    assert "laundering_detected" in src
