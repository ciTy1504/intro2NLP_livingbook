"""BibTeX reading, validation and safe writing.

`references.bib` has exactly one writer in the whole system (`BibtexValidator`), and
this module is how it writes. Entries are appended or replaced surgically — the file
is never regenerated from parsed data, because round-tripping BibTeX loses formatting
and comment structure that the author maintains by hand (the file is organised with
`%% CHƯƠNG n` section banners).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .latex import extract_bib_entries, match_brace, parse_bib_fields

_ARXIV_IN_TEXT = re.compile(r"arxiv[:/\s]*(\d{4}\.\d{4,5})", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")


@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def title(self) -> str:
        return _clean_braces(self.fields.get("title", ""))

    @property
    def authors(self) -> str:
        return _clean_braces(self.fields.get("author", ""))

    @property
    def year(self) -> int | None:
        raw = self.fields.get("year", "")
        m = re.search(r"\d{4}", raw)
        return int(m.group()) if m else None

    @property
    def venue(self) -> str:
        for key in ("journal", "booktitle", "publisher", "school", "institution"):
            if self.fields.get(key):
                return _clean_braces(self.fields[key])
        return ""

    @property
    def doi(self) -> str:
        doi = self.fields.get("doi", "").strip()
        if doi:
            return doi.replace("https://doi.org/", "").strip()
        m = _DOI_RE.search(self.fields.get("url", ""))
        return m.group(1) if m else ""

    @property
    def arxiv_id(self) -> str:
        for key in ("eprint", "archiveprefix", "url", "journal", "note"):
            m = _ARXIV_IN_TEXT.search(self.fields.get(key, ""))
            if m:
                return m.group(1)
        eprint = self.fields.get("eprint", "").strip()
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", eprint):
            return eprint.split("v")[0]
        return ""

    @property
    def url(self) -> str:
        return self.fields.get("url", "").strip()

    def author_list(self) -> list[str]:
        raw = self.authors
        if not raw:
            return []
        return [a.strip() for a in re.split(r"\s+and\s+", raw) if a.strip()]

    def to_row(self) -> dict[str, Any]:
        return {
            "bib_key": self.key, "entry_type": self.entry_type, "title": self.title,
            "authors": self.authors, "year": self.year, "venue": self.venue,
            "doi": self.doi, "arxiv_id": self.arxiv_id, "url": self.url, "raw": self.raw,
        }


def _clean_braces(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


class Bibliography:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, BibEntry] = {}
        self.order: list[str] = []
        self.reload()

    def reload(self) -> None:
        self.entries.clear()
        self.order.clear()
        if not self.path.exists():
            return
        text = self.path.read_text(encoding="utf-8", errors="replace")
        for raw_entry in extract_bib_entries(text):
            key = raw_entry["key"]
            entry = BibEntry(
                key=key, entry_type=raw_entry["type"],
                fields=parse_bib_fields(raw_entry["raw"]), raw=raw_entry["raw"],
            )
            self.entries[key] = entry
            self.order.append(key)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str) -> BibEntry | None:
        return self.entries.get(key)

    def keys(self) -> set[str]:
        return set(self.entries)

    # -- validation --------------------------------------------------------
    def validate(self) -> list[dict[str, Any]]:
        """Deterministic structural checks. Semantic checks live in the agent."""
        problems: list[dict[str, Any]] = []
        seen_titles: dict[str, str] = {}
        seen_dois: dict[str, str] = {}
        seen_arxiv: dict[str, str] = {}

        # Duplicate keys can only be detected on the raw text: the dict already
        # collapsed them.
        if self.path.exists():
            raw_keys = [e["key"] for e in extract_bib_entries(
                self.path.read_text(encoding="utf-8", errors="replace"))]
            for key in {k for k in raw_keys if raw_keys.count(k) > 1}:
                problems.append({
                    "key": key, "severity": "error", "kind": "duplicate_key",
                    "detail": f"{key} is defined {raw_keys.count(key)} times",
                })

        for key, entry in self.entries.items():
            if not entry.title:
                problems.append({"key": key, "severity": "error", "kind": "missing_title",
                                 "detail": "entry has no title field"})
            if not entry.authors:
                problems.append({"key": key, "severity": "warning", "kind": "missing_author",
                                 "detail": "entry has no author field"})
            if entry.year is None:
                problems.append({"key": key, "severity": "error", "kind": "missing_year",
                                 "detail": "entry has no parseable year"})
            elif not (1900 <= entry.year <= 2100):
                problems.append({"key": key, "severity": "error", "kind": "implausible_year",
                                 "detail": f"year {entry.year} is out of range"})
            if not entry.venue and entry.entry_type not in ("misc", "unpublished", "techreport"):
                problems.append({"key": key, "severity": "warning", "kind": "missing_venue",
                                 "detail": f"{entry.entry_type} entry has no venue"})
            if entry.doi and not _DOI_RE.fullmatch(entry.doi):
                problems.append({"key": key, "severity": "warning", "kind": "malformed_doi",
                                 "detail": f"doi {entry.doi!r} is not well formed"})
            if entry.raw.count("{") != entry.raw.count("}"):
                problems.append({"key": key, "severity": "error", "kind": "unbalanced_braces",
                                 "detail": "entry has unbalanced braces"})

            norm = _normalise_title(entry.title)
            if norm and norm in seen_titles and seen_titles[norm] != key:
                problems.append({
                    "key": key, "severity": "warning", "kind": "duplicate_title",
                    "detail": f"same title as {seen_titles[norm]}",
                })
            elif norm:
                seen_titles[norm] = key

            if entry.doi:
                if entry.doi in seen_dois:
                    problems.append({"key": key, "severity": "error", "kind": "duplicate_doi",
                                     "detail": f"same DOI as {seen_dois[entry.doi]}"})
                else:
                    seen_dois[entry.doi] = key
            if entry.arxiv_id:
                if entry.arxiv_id in seen_arxiv:
                    problems.append({"key": key, "severity": "warning", "kind": "duplicate_arxiv",
                                     "detail": f"same arXiv id as {seen_arxiv[entry.arxiv_id]}"})
                else:
                    seen_arxiv[entry.arxiv_id] = key
        return problems

    # -- writing -----------------------------------------------------------
    def suggest_key(self, authors: str, year: int | None, title: str) -> str:
        """Key in the manuscript's existing style: surnameYYYYfirstword."""
        surname = "anon"
        if authors:
            first = re.split(r"\s+and\s+", authors)[0]
            if "," in first:
                surname = first.split(",")[0]
            else:
                surname = first.split()[-1] if first.split() else "anon"
        surname = _ascii_slug(surname).lower() or "anon"
        word = ""
        for token in re.findall(r"[A-Za-z]{3,}", title or ""):
            if token.lower() not in _STOPWORDS:
                word = token.lower()
                break
        base = f"{surname}{year or ''}{_ascii_slug(word)}"
        candidate, n = base, 1
        while candidate in self.entries:
            n += 1
            candidate = f"{base}{n}"
        return candidate

    def render_entry(self, key: str, entry_type: str, fields: dict[str, str]) -> str:
        lines = [f"@{entry_type}{{{key},"]
        ordered = [f for f in ("title", "author", "journal", "booktitle", "volume",
                               "number", "pages", "year", "publisher", "doi",
                               "eprint", "archivePrefix", "primaryClass", "url", "note")
                   if f in fields]
        ordered += [f for f in fields if f not in ordered]
        for name in ordered:
            value = str(fields[name]).strip()
            if not value:
                continue
            lines.append(f"  {name}={{{value}}},")
        if len(lines) > 1:
            lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        return "\n".join(lines)

    def append(self, key: str, entry_type: str, fields: dict[str, str], *,
               comment: str = "") -> str:
        """Append a new entry. Raises if the key already exists."""
        if key in self.entries:
            raise ValueError(f"bib key {key!r} already exists")
        rendered = self.render_entry(key, entry_type, fields)
        block = "\n\n" + (f"% {comment}\n" if comment else "") + rendered + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(block)
        self.reload()
        return rendered

    def replace(self, key: str, entry_type: str, fields: dict[str, str]) -> str:
        """Replace one entry in place, leaving the rest of the file byte-identical."""
        if key not in self.entries:
            raise KeyError(f"bib key {key!r} not found")
        text = self.path.read_text(encoding="utf-8", errors="replace")
        old = self.entries[key].raw
        start = text.find(old)
        if start == -1:
            raise RuntimeError(f"could not locate entry {key!r} in {self.path}")
        rendered = self.render_entry(key, entry_type, fields)
        self.path.write_text(text[:start] + rendered + text[start + len(old):], encoding="utf-8")
        self.reload()
        return rendered

    def dedupe_report(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for key, entry in self.entries.items():
            norm = _normalise_title(entry.title)
            if norm:
                groups.setdefault(norm, []).append(key)
        return {k: v for k, v in groups.items() if len(v) > 1}


_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "from", "into", "via", "using",
    "towards", "toward", "are", "is", "of", "on", "in", "to", "by", "at", "as",
}


def _normalise_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", t)


def _ascii_slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9]+", "", t)


def bibtex_from_metadata(
    *, title: str, authors: Iterable[str], year: int | None, venue: str = "",
    doi: str = "", arxiv_id: str = "", url: str = "", entry_type: str = "",
) -> tuple[str, dict[str, str]]:
    """Build BibTeX fields from verified metadata.

    Used by the citation pipeline after a source has been verified — never from a
    model's recollection of what a paper's metadata probably is.
    """
    author_str = " and ".join(a.strip() for a in authors if a and a.strip())
    fields: dict[str, str] = {"title": title.strip(), "author": author_str}
    if year:
        fields["year"] = str(year)

    if not entry_type:
        if arxiv_id and not venue:
            entry_type = "article"
        elif venue and re.search(r"proceedings|conference|workshop|symposium|meeting",
                                 venue, re.I):
            entry_type = "inproceedings"
        elif venue:
            entry_type = "article"
        else:
            entry_type = "misc"

    if entry_type == "inproceedings" and venue:
        fields["booktitle"] = venue
    elif venue:
        fields["journal"] = venue

    if doi:
        fields["doi"] = doi
    if arxiv_id:
        fields["eprint"] = arxiv_id
        fields["archivePrefix"] = "arXiv"
        if not venue:
            fields["journal"] = f"arXiv preprint arXiv:{arxiv_id}"
    if url:
        fields["url"] = url
    return entry_type, fields
