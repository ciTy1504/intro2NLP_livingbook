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
