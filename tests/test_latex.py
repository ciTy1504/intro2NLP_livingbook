"""LaTeX parsing tests.

These run against the real manuscript, not fixtures. The parser's correctness is not
a property of toy input — it has to hold for `\chapter*`, nested braces in titles,
`minted` blocks containing unbalanced braces, and the book's own `\bookimage` macro.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.config import get_config  # noqa: E402
from livingbook.knowledge.latex import (  # noqa: E402
    LatexParser,
    extract_bib_entries,
    match_brace,
    parse_bib_fields,
    read_braced,
    strip_latex,
)


@pytest.fixture(scope="module")
def book():
    cfg = get_config()
    return LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex")).parse()


def test_brace_matching_handles_nesting():
    text = "{a {b {c} d} e} tail"
    assert match_brace(text, 0) == 14
    assert read_braced(text, 0)[0] == "a {b {c} d} e"


def test_brace_matching_ignores_escaped():
    assert match_brace(r"{a \{ b}", 0) == 7


def test_parses_the_real_manuscript(book):
    stats = book.stats()
    assert stats["files"] > 50
    assert stats["chapter"] >= 15
    assert stats["section"] >= 70
    assert stats["words"] > 100_000


def test_every_node_has_a_valid_parent(book):
    ids = {n.id for n in book.nodes}
    for node in book.nodes:
        if node.parent_id:
            assert node.parent_id in ids, f"{node.id} has a dangling parent"


def test_start_line_points_at_the_heading(book):
    root = book.root
    for node in book.nodes:
        if node.kind == "part":
            continue
        lines = (root / node.file).read_text(encoding="utf-8", errors="replace").splitlines()
        assert node.start_line - 1 < len(lines)
        assert f"\\{node.kind}" in lines[node.start_line - 1], (
            f"{node.file}:{node.start_line} does not start with \\{node.kind}")


def test_starred_headings_are_unnumbered(book):
    """`\\chapter*` must not advance the counter, or every later number is wrong."""
    starred = [n for n in book.nodes if n.starred]
    assert starred, "the manuscript uses \\chapter* and \\section*"
    for node in starred:
        assert node.number == "", f"{node.title} is starred but was numbered {node.number}"


def test_chapter_numbering_continues_across_parts(book):
    """The `book` class does not reset the chapter counter at `\\part`."""
    numbers = [int(n.number) for n in book.nodes
               if n.kind == "chapter" and n.number.isdigit()]
    assert numbers == sorted(numbers), "chapter numbers must increase monotonically"
    assert numbers == list(range(1, len(numbers) + 1)), "chapter numbers must be contiguous"


def test_section_numbers_nest_under_their_chapter(book):
    by_id = book.by_id()
    for node in book.nodes:
        if node.kind != "section" or not node.number:
            continue
        chapter = by_id.get(node.parent_id)
        if chapter and chapter.number:
            assert node.number.startswith(chapter.number + "."), (
                f"section {node.number} is not under chapter {chapter.number}")


def test_citations_all_resolve(book):
    cited = {c for n in book.nodes for c in n.cites}
    orphans = cited - book.bib_keys
    assert not orphans, f"citations with no bibliography entry: {sorted(orphans)}"


def test_bookimage_requirements_are_captured(book):
    """The \\bookimage description is the Visual Engine's input; it must survive parsing."""
    bookimages = [f for f in book.figures if f.kind == "bookimage"]
    assert bookimages, "the manuscript uses \\bookimage"
    for fig in bookimages:
        assert fig.key, "every \\bookimage must have a key"
        assert len(fig.requirement) > 40, (
            f"{fig.key}: description was truncated to {len(fig.requirement)} chars — "
            "likely a brace-matching failure")


def test_strip_latex_keeps_citations_and_drops_code():
    latex = r"""
\section{Test}
Đây là một câu có trích dẫn \cite{smith2020}.
\begin{minted}{python}
x = {"a": 1}
\end{minted}
Tham chiếu \ref{sec:foo}.
"""
    plain = strip_latex(latex)
    assert "[smith2020]" in plain, "citations must survive for the verifier to see them"
    assert "(sec:foo)" in plain
    assert "CODE BLOCK" in plain
    assert 'x = {"a": 1}' not in plain


def test_bib_fields_survive_nested_braces():
    raw = '@article{k, title={The {BERT} Model}, author={A and B}, year={2019}}'
    fields = parse_bib_fields(raw)
    assert fields["title"] == "The {BERT} Model"
    assert fields["author"] == "A and B"
    assert fields["year"] == "2019"


def test_bib_entries_extracted_from_real_file():
    cfg = get_config()
    entries = extract_bib_entries(cfg.bib_path.read_text(encoding="utf-8"))
    assert len(entries) > 200
    keys = [e["key"] for e in entries]
    assert len(keys) == len(set(keys)), "duplicate bib keys"
