"""Language-purity regression tests.

These exist because of a specific failure: the Writer produced
"cơ chế sinh học của brain humano" — a Spanish word inside Vietnamese prose — and the
LLM editorial verifier read the passage and passed it. The deterministic check must
catch that case and must not fire on the author's own writing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.tools.language import check_language  # noqa: E402


def run(text: str, **kw):
    return asyncio.run(check_language(text, **kw))


def test_catches_the_actual_defect():
    """The exact sentence that shipped past the LLM verifier."""
    text = ("Cần lưu ý rằng việc gán nhãn nhận thức này trong phần mềm tác tử chỉ "
            "mang tính tượng trưng kỹ thuật và khác biệt bản chất so với cơ chế "
            "sinh học của brain humano.")
    result = run(text)
    assert not result["ok"], "the foreign word must fail the check"
    assert result["blocker_count"] >= 1
    flagged = {f["word"].lower() for f in result["findings"]
               if f["severity"] == "blocker"}
    assert "humano" in flagged


@pytest.mark.parametrize("word", ["humano", "cerebro", "modelo", "aprendizaje",
                                  "donnees", "linguaggio"])
def test_catches_other_foreign_leakage(word):
    result = run(f"Đây là một câu tiếng Việt có chứa từ {word} không đúng chỗ.")
    assert result["blocker_count"] >= 1, f"{word} should have been flagged"


def test_does_not_flag_the_books_own_prose():
    """Zero false positives on real manuscript text, or the check is unusable."""
    from livingbook.knowledge.retrieval import BookRetriever
    from livingbook.state.store import get_store

    store = get_store()
    retriever = BookRetriever(store)
    rows = store.query(
        "SELECT id FROM book_nodes WHERE kind IN ('subsection','section') "
        "AND word_count > 200 LIMIT 12")
    if not rows:
        pytest.skip("knowledge base not indexed")

    offenders = []
    for row in rows:
        node = retriever._load(row["id"], with_body=True)
        if not node or not node.body:
            continue
        result = run(node.body)
        if result["blocker_count"]:
            offenders.append(
                (node.ref, [f["word"] for f in result["findings"]
                            if f["severity"] == "blocker"]))
    assert not offenders, f"false positives on the author's own text: {offenders}"


def test_established_english_terms_pass():
    text = ("Kiến trúc Transformer sử dụng cơ chế self-attention và các khối "
            "encoder, decoder để xử lý token.")
    result = run(text)
    assert result["ok"], (
        f"established technical vocabulary was flagged: {result['findings']}")


def test_new_technical_terms_are_minor_not_blocking():
    """A genuinely new term deserves a human glance, not a rejected patch."""
    result = run("Kỹ thuật quantization theo kiểu hypothetical-newtechnique rất mới.")
    assert result["ok"], "an unknown term must not block on its own"
    assert result["unknown_count"] >= 1


def test_latex_commands_are_ignored():
    text = (r"\textbf{Bộ nhớ ngữ nghĩa} lưu trữ tri thức \cite{smith2020} "
            r"và \emph{quy trình} xử lý.")
    result = run(text)
    flagged = {f["word"].lower() for f in result["findings"]}
    for command in ("textbf", "emph", "cite"):
        assert command not in flagged, f"LaTeX command {command} was treated as a word"


def test_vocabulary_is_built_from_the_manuscript():
    result = run("test")
    assert result["vocabulary_size"] > 3000, (
        "the baseline vocabulary should come from the real manuscript")
