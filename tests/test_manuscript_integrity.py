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
