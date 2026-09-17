"""Deterministic language checks for new manuscript text.

A real defect motivated this. The Writer produced otherwise-good Vietnamese containing
the phrase "cơ chế sinh học của brain humano" — a Spanish word fragment sitting inside
a Vietnamese sentence. The LLM editorial verifier read the passage and passed it. That
is not a surprising failure: a model asked "is this good Vietnamese prose?" weighs the
paragraph as a whole, and one foreign word inside a fluent paragraph is easy to miss.

The fix is not a better prompt. It is a check that cannot miss, because it does not
read for meaning at all: build a vocabulary from the 143k words the author has already
written, and flag any Latin-script word in new text that appears nowhere in the book,
is not a known technical concept, and is not a LaTeX command. "humano" fails all three.
"transformer" passes the first. A genuinely new technical term is flagged once, which
is the correct outcome — a new term in a textbook deserves a human glance.

Cheap enough to run on every patch, and it produces a location rather than an opinion.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from ..config import get_config
from .registry import Capability, tool

#: Latin letters plus the Vietnamese diacritic range.
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\-]{2,}")
_VIETNAMESE_DIACRITIC = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                                   r"ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự"
                                   r"ỳýỷỹỵđ]", re.I)

#: Function words from languages that most often leak into an LLM's Vietnamese.
#: These are flagged outright rather than run through the vocabulary check, because
#: a few of them do appear in the manuscript inside English quotations.
_FOREIGN_MARKERS = {
    # Spanish / Portuguese
    "humano", "humana", "cerebro", "memoria", "modelo", "datos", "ejemplo",
    "sistema", "informacion", "conocimiento", "aprendizaje", "lenguaje",
    "palabras", "entrada", "salida", "trabajo", "nuevo", "nueva", "grande",
    "pequeno", "tambien", "porque", "cuando", "donde", "desde", "hasta",
    "entre", "sobre", "todos", "todas", "otros", "mismo", "cada", "muy",
    # French
    "modele", "donnees", "apprentissage", "langage", "connaissance",
    "entrainement", "reseau", "couche", "exemple", "utilisateur",
    # Italian
    "modello", "dati", "apprendimento", "linguaggio", "rete", "esempio",
}

_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z@]+")


@lru_cache(maxsize=1)
def manuscript_vocabulary() -> frozenset[str]:
    """Every word the author has already used, lower-cased.

    Built from the manuscript itself rather than a dictionary, so the baseline is the
    book's own established vocabulary — including its Vietnamese, its English
    technical terms, and its author's particular phrasing.
    """
    cfg = get_config()
    words: set[str] = set()
    for path in cfg.manuscript_dir.rglob("*.tex"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _LATEX_CMD_RE.sub(" ", text)
        words.update(m.group(0).lower() for m in _WORD_RE.finditer(text))
    return frozenset(words)


@lru_cache(maxsize=1)
def known_concepts() -> frozenset[str]:
    """Concept names from the knowledge base, split into words.

    The index holds 1,385 concepts extracted from the book, so a technical term the
    system itself recognises is never flagged as foreign.
    """
    try:
        from ..state.store import get_store
        rows = get_store().query("SELECT canonical_name FROM concepts")
    except Exception:
        return frozenset()
    words: set[str] = set()
    for r in rows:
        words.update(m.group(0).lower() for m in _WORD_RE.finditer(r["canonical_name"]))
    return frozenset(words)


@tool("check_language", [Capability.FS_READ],
      description="Flag foreign-language contamination and unknown vocabulary in new text.")
async def check_language(
    text: str, *, expect: str = "vi", max_findings: int = 30,
) -> dict[str, Any]:
    """Check new manuscript text for words that do not belong.

    Returns findings rather than a verdict, so the caller decides severity. A
    ``foreign_word`` finding is a blocker in practice; ``unknown_word`` is a prompt for
    a human to confirm a genuinely new term.
    """
    vocabulary = manuscript_vocabulary()
    concepts = known_concepts()

    stripped = _LATEX_CMD_RE.sub(" ", text)
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line_no, line in enumerate(stripped.splitlines(), 1):
        for match in _WORD_RE.finditer(line):
            word = match.group(0)
            lower = word.lower()
            if lower in seen:
                continue

            if lower in _FOREIGN_MARKERS:
                seen.add(lower)
                findings.append({
                    "word": word, "line": line_no, "kind": "foreign_word",
                    "severity": "blocker",
                    "detail": f"{word!r} is a Spanish/Portuguese/French word in "
                              f"{expect} prose",
                    "context": line.strip()[:180],
                })
                continue

            if lower in vocabulary or lower in concepts:
                continue
            # A word carrying Vietnamese diacritics is Vietnamese by construction.
            if _VIETNAMESE_DIACRITIC.search(word):
                continue
            if word.isupper() or any(ch.isdigit() for ch in word):
                continue  # acronyms and identifiers

            seen.add(lower)
            findings.append({
                "word": word, "line": line_no, "kind": "unknown_word",
                "severity": "minor",
                "detail": f"{word!r} appears nowhere in the existing manuscript and is "
                          "not a known concept — confirm it is intended",
                "context": line.strip()[:180],
            })

    blockers = [f for f in findings if f["severity"] == "blocker"]
    return {
        "ok": not blockers,
        "expect": expect,
        "vocabulary_size": len(vocabulary),
        "known_concepts": len(concepts),
        "findings": findings[:max_findings],
        "blocker_count": len(blockers),
        "unknown_count": len(findings) - len(blockers),
    }
