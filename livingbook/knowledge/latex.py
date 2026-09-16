"""LaTeX-aware parsing of the manuscript.

This is what lets a 153k-word book be edited inside a bounded context: it turns the
`\input` manifest into an addressable tree of nodes with line ranges, so an agent can
ask for "section 5.2" and receive exactly those lines instead of the whole book.

Written against the actual conventions of this manuscript (see AUDIT.md §1): chapter
headings live in each `chapters_*/intro.tex`, figures use the custom
`\bookimage{key}{description}` macro, and labels are namespaced `sec:`/`ssec:`/etc.
A general-purpose LaTeX parser would be both harder and less useful here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Sectioning commands in depth order.
_LEVELS = ["part", "chapter", "section", "subsection", "subsubsection"]
_LEVEL_DEPTH = {name: i for i, name in enumerate(_LEVELS)}

_INPUT_RE = re.compile(r"^[^%]*?\\(?:input|include)\{([^}]+)\}", re.M)
_HEADING_RE = re.compile(
    r"^[ \t]*\\(part|chapter|section|subsection|subsubsection)(\*?)\s*(?:\[[^\]]*\])?\s*\{",
    re.M,
)
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_REF_RE = re.compile(r"\\(ref|eqref|autoref|nameref)\{([^}]+)\}")
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption(?:of\{figure\})?\s*\{")
_EQUATION_ENV_RE = re.compile(r"\\begin\{(equation|align|gather|multline)\*?\}")
_MINTED_RE = re.compile(r"\\begin\{minted\}(?:\[[^\]]*\])?\{([^}]+)\}")
_BIBKEY_RE = re.compile(r"^\s*@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)
_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.M)

#: tcolorbox theorem environments defined in style.tex, mapped to the label PREFIX
#: each one prepends. `\newtcbtheorem{example}{Ví dụ}{...}{ex}` means that
#: `\begin{example}{title}{ex:bow_example}` actually defines the label
#: `ex:ex:bow_example`. Without this, every reference to an example, definition or
#: recipe looks dangling to the cross-reference checker.
_THEOREM_PREFIXES = {
    "definition": "def",
    "example": "ex",
    "recipe": "rcp",
}
_THEOREM_RE = re.compile(
    r"\\begin\{(" + "|".join(_THEOREM_PREFIXES) + r")\}\s*(?:\[[^\]]*\])?\s*(?=\{)"
)


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, honouring the escaped ``\\%``."""
    return _COMMENT_RE.sub("", text)


def match_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_index``, or -1.

    Needed because titles and `\\bookimage` descriptions routinely contain nested
    braces, which a non-greedy regex silently truncates.
    """
    if open_index >= len(text) or text[open_index] != "{":
        return -1
    depth = 0
    i = open_index
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def read_braced(text: str, open_index: int) -> tuple[str, int]:
    """Return (contents, index_after_closing_brace) for a ``{...}`` group."""
    close = match_brace(text, open_index)
    if close == -1:
        return "", open_index + 1
    return text[open_index + 1:close], close + 1


def find_macro_calls(text: str, macro: str, n_args: int = 1) -> Iterator[tuple[int, list[str]]]:
    """Yield (position, args) for every ``\\macro[opt]{a}{b}...`` occurrence."""
    pattern = re.compile(r"\\" + re.escape(macro) + r"\s*(?:\[[^\]]*\])?\s*(?=\{)")
    for m in pattern.finditer(text):
        i = m.end()
        args: list[str] = []
        for _ in range(n_args):
            if i >= len(text) or text[i] != "{":
                break
            arg, i = read_braced(text, i)
            args.append(arg)
        if len(args) == n_args:
            yield m.start(), args


@dataclass
class Figure:
    key: str
    kind: str                 # bookimage | includegraphics
    requirement: str = ""     # the \bookimage description — a ready-made visual spec
    caption: str = ""
    label: str = ""
    line: int = 0
    path: str = ""


@dataclass
class TexNode:
    kind: str                 # part|chapter|section|subsection|subsubsection
    title: str
    label: str | None
    file: str                 # relative to the manuscript root
    start_line: int           # 1-based, inclusive
    end_line: int             # 1-based, inclusive
    depth: int
    starred: bool = False     # \section* etc. — unnumbered, does not advance counters
    order_idx: int = 0
    number: str = ""
    parent_id: str | None = None
    id: str = ""
    body: str = ""
    cites: list[str] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    equations: int = 0
    code_blocks: int = 0
    word_count: int = 0
    content_sha256: str = ""

    def make_id(self) -> str:
        base = self.label or f"{self.file}:{self.start_line}"
        digest = hashlib.sha1(f"{self.file}|{base}|{self.kind}".encode()).hexdigest()[:10]
        slug = re.sub(r"[^a-z0-9]+", "_", (self.label or self.title).lower())[:40].strip("_")
        return f"{self.kind[:4]}_{slug}_{digest}" if slug else f"{self.kind[:4]}_{digest}"


@dataclass
class ParsedBook:
    root: Path
    main_tex: Path
    files: list[str]                      # in reading order
    nodes: list[TexNode]
    bib_keys: set[str]
    figures: list[Figure]

    def by_id(self) -> dict[str, TexNode]:
        return {n.id: n for n in self.nodes}

    def by_label(self) -> dict[str, TexNode]:
        return {n.label: n for n in self.nodes if n.label}

    def chapters(self) -> list[TexNode]:
        return [n for n in self.nodes if n.kind == "chapter"]

    def stats(self) -> dict[str, int]:
        out = {k: 0 for k in _LEVELS}
        for n in self.nodes:
            out[n.kind] = out.get(n.kind, 0) + 1
        out["files"] = len(self.files)
        out["words"] = sum(n.word_count for n in self.nodes)
        out["figures"] = len(self.figures)
        out["cites"] = sum(len(n.cites) for n in self.nodes)
        return out


class LatexParser:
    def __init__(self, manuscript_root: Path, main_tex: str = "main.tex") -> None:
        self.root = Path(manuscript_root)
        self.main_tex = self.root / main_tex

    # -- file order --------------------------------------------------------
    def input_order(self) -> list[str]:
        """Files reachable from main.tex via ``\\input``, in reading order.

        Reading order matters: it is what makes ``order_idx`` meaningful, and
        `order_idx` is how "the section before this one" is answered without an LLM.
        """
        seen: set[str] = set()
        ordered: list[str] = []

        def walk(rel: str) -> None:
            if rel in seen:
                return
            seen.add(rel)
            path = self._resolve(rel)
            if not path or not path.exists():
                return
            ordered.append(rel)
            text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
            for m in _INPUT_RE.finditer(text):
                walk(self._normalise(m.group(1)))

        main_rel = self.main_tex.name
        main_text = strip_comments(self.main_tex.read_text(encoding="utf-8", errors="replace"))
        ordered.append(main_rel)
        seen.add(main_rel)
        for m in _INPUT_RE.finditer(main_text):
            walk(self._normalise(m.group(1)))
        return ordered

    @staticmethod
    def _normalise(rel: str) -> str:
        rel = rel.strip().replace("\\", "/")
        return rel if rel.endswith(".tex") else rel + ".tex"

    def _resolve(self, rel: str) -> Path | None:
        candidate = self.root / rel
        return candidate if candidate.exists() else None

    # -- parsing -----------------------------------------------------------
    def parse(self) -> ParsedBook:
        files = self.input_order()
        nodes: list[TexNode] = []

        for rel in files:
            path = self._resolve(rel)
            if not path:
                continue
            nodes.extend(self._parse_file(rel, path))

        self._assign_structure(nodes)

        figures = [f for n in nodes for f in n.figures]
        bib_keys = self._bib_keys()
        return ParsedBook(
            root=self.root, main_tex=self.main_tex, files=files,
            nodes=nodes, bib_keys=bib_keys, figures=figures,
        )

    def _parse_file(self, rel: str, path: Path) -> list[TexNode]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_comments(raw)
        lines = raw.splitlines()

        headings: list[tuple[int, str, str, bool]] = []   # (offset, kind, title, starred)
        for m in _HEADING_RE.finditer(text):
            kind = m.group(1)
            starred = m.group(2) == "*"
            brace = text.find("{", m.end() - 1)
            title, _ = read_braced(text, brace)
            headings.append((m.start(), kind, _clean_title(title), starred))

        if not headings:
            return []

        # Map char offsets to 1-based line numbers.
        line_starts = _line_starts(text)
        out: list[TexNode] = []
        for i, (offset, kind, title, starred) in enumerate(headings):
            start_line = _line_of(line_starts, offset)
            end_offset = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            end_line = _line_of(line_starts, max(offset, end_offset - 1))
            body = text[offset:end_offset]

            node = TexNode(
                kind=kind, title=title, label=_first_label(body), file=rel,
                start_line=start_line, end_line=min(end_line, max(1, len(lines))),
                depth=_LEVEL_DEPTH[kind], starred=starred,
                body=body,
            )
            node.id = node.make_id()
            self._enrich(node, body, line_starts, offset)
            out.append(node)
        return out

    def _enrich(self, node: TexNode, body: str, line_starts: list[int], offset: int) -> None:
        for m in _CITE_RE.finditer(body):
            node.cites.extend(k.strip() for k in m.group(1).split(",") if k.strip())
        node.refs = [(m.group(1), m.group(2)) for m in _REF_RE.finditer(body)]
        node.equations = len(_EQUATION_ENV_RE.findall(body))
        node.code_blocks = len(_MINTED_RE.findall(body))

        # \bookimage[width]{key}{description} — the manuscript's own figure macro.
        # The third argument is already a specification of what the figure must show,
        # which is exactly what the Visual Engine needs as input.
        for pos, args in find_macro_calls(body, "bookimage", n_args=2):
            key, requirement = args[0].strip(), args[1].strip()
            node.figures.append(Figure(
                key=key, kind="bookimage", requirement=_clean_title(requirement),
                caption=_nearby_caption(body, pos),
                label=_nearby_label(body, pos),
                line=_line_of(line_starts, offset + pos),
            ))

        for m in _INCLUDEGRAPHICS_RE.finditer(body):
            raw_path = m.group(1).strip()
            key = Path(raw_path).stem
            node.figures.append(Figure(
                key=key, kind="includegraphics", path=raw_path,
                caption=_nearby_caption(body, m.start()),
                label=_nearby_label(body, m.start()),
                line=_line_of(line_starts, offset + m.start()),
            ))

        plain = strip_latex(body)
        node.word_count = len(plain.split())
        node.content_sha256 = hashlib.sha256(body.encode()).hexdigest()

    def _assign_structure(self, nodes: list[TexNode]) -> None:
        """Assign reading order, parents and rendered numbering (5.2.1 style)."""
        counters = [0] * len(_LEVELS)
        stack: list[TexNode] = []

        for idx, node in enumerate(nodes):
            node.order_idx = idx
            d = node.depth

            if node.starred:
                # \chapter*/\section* are unnumbered and leave every counter alone,
                # exactly as LaTeX does. The manuscript uses them for the preface,
                # the appendix, the closing chapter and a few interstitial sections;
                # advancing counters here would shift every following number.
                node.number = ""
            else:
                counters[d] += 1
                # In the `book` class `\part` does not reset the chapter counter, so
                # Part II's first chapter continues the sequence rather than
                # restarting at 1.
                if node.kind != "part":
                    for deeper in range(d + 1, len(counters)):
                        counters[deeper] = 0

                if node.kind == "part":
                    node.number = _roman(counters[0])
                else:
                    parts = [str(counters[i]) for i in range(1, d + 1) if counters[i] > 0]
                    node.number = ".".join(parts)

            while stack and stack[-1].depth >= d:
                stack.pop()
            node.parent_id = stack[-1].id if stack else None
            stack.append(node)

    def _bib_keys(self) -> set[str]:
        bib = self.root / "references.bib"
        if not bib.exists():
            return set()
        text = bib.read_text(encoding="utf-8", errors="replace")
        return {m.group(2).strip() for m in _BIBKEY_RE.finditer(text)}


# -- helpers ---------------------------------------------------------------


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _line_of(line_starts: list[int], offset: int) -> int:
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _first_label(body: str) -> str | None:
    m = _LABEL_RE.search(body[:1200])
    return m.group(1).strip() if m else None


def _nearby_label(body: str, pos: int, window: int = 700) -> str:
    window_text = body[pos:pos + window]
    m = _LABEL_RE.search(window_text)
    return m.group(1).strip() if m else ""


def _nearby_caption(body: str, pos: int, window: int = 900) -> str:
    window_text = body[pos:pos + window]
    m = _CAPTION_RE.search(window_text)
    if not m:
        return ""
    brace = window_text.find("{", m.end() - 1)
    caption, _ = read_braced(window_text, brace)
    return _clean_title(caption)


_TITLE_CLEAN = [
    (re.compile(r"\\newtag\b\s*"), ""),
    (re.compile(r"\\reftag\b\s*"), ""),
    (re.compile(r"\\texorpdfstring\{([^}]*)\}\{[^}]*\}"), r"\1"),
    (re.compile(r"\\(?:textbf|textit|emph|texttt|textsc|mbox)\{([^}]*)\}"), r"\1"),
    (re.compile(r"~"), " "),
    (re.compile(r"\\&"), "&"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\\"), " "),
    (re.compile(r"\s+"), " "),
]


def _clean_title(title: str) -> str:
    out = title
    for pattern, repl in _TITLE_CLEAN:
        out = pattern.sub(repl, out)
    return out.strip()


def _roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


_STRIP_ENVS = re.compile(
    r"\\begin\{(minted|verbatim|lstlisting|tikzpicture)\}.*?\\end\{\1\}", re.S
)
_STRIP_CMDS = [
    (re.compile(r"\\(?:label|index|vspace|hspace|newpage|clearpage|noindent|centering)"
                r"\s*(?:\{[^}]*\}|\[[^\]]*\])?"), ""),
    (re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}"), r"[\1]"),
    (re.compile(r"\\(?:ref|eqref|autoref|nameref)\{([^}]+)\}"), r"(\1)"),
    (re.compile(r"\\bookimage\s*(?:\[[^\]]*\])?"), ""),
    (re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\{[^}]*\}"), ""),
    (re.compile(r"\\(?:textbf|textit|emph|texttt|textsc|underline|mbox)\{"), "{"),
    (re.compile(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^}]*\})*"), "\n"),
    (re.compile(r"\\end\{[^}]*\}"), "\n"),
    (re.compile(r"\\item\b"), "\n- "),
    (re.compile(r"\\[a-zA-Z]+\*?"), " "),
    (re.compile(r"[{}]"), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def strip_latex(text: str) -> str:
    """Reduce LaTeX to readable plain text for LLM input.

    Not a renderer — it only needs to preserve meaning. Code blocks are dropped
    entirely (they are noise for claim extraction and expensive in tokens), while
    citations and cross-references are kept in bracket form because a verifier needs
    to see that a statement carries one.
    """
    out = strip_comments(text)
    out = _STRIP_ENVS.sub("\n[CODE BLOCK]\n", out)
    for pattern, repl in _STRIP_CMDS:
        out = pattern.sub(repl, out)
    return out.strip()


def collect_labels(text: str) -> set[str]:
    """Every label a file actually defines, including tcolorbox theorem labels.

    A plain ``\\label{...}`` scan misses theorem environments entirely, because their
    label is assembled from the environment's configured prefix plus the key given at
    the call site.
    """
    labels = set(_LABEL_RE.findall(text))
    for m in _THEOREM_RE.finditer(text):
        env = m.group(1)
        i = m.end()
        # {title}{key}
        _title, i = read_braced(text, i)
        if i < len(text) and text[i] == "{":
            key, _ = read_braced(text, i)
            key = key.strip()
            if key:
                labels.add(f"{_THEOREM_PREFIXES[env]}:{key}")
    return labels


def extract_bib_entries(bib_text: str) -> list[dict[str, str]]:
    """Split a .bib file into raw entries with their keys and types."""
    entries: list[dict[str, str]] = []
    for m in _BIBKEY_RE.finditer(bib_text):
        start = m.start()
        brace = bib_text.find("{", start)
        close = match_brace(bib_text, brace)
        raw = bib_text[start:close + 1] if close != -1 else bib_text[start:start + 2000]
        entries.append({"key": m.group(2).strip(), "type": m.group(1).lower(), "raw": raw})
    return entries


def parse_bib_fields(raw: str) -> dict[str, str]:
    """Pull the fields out of one BibTeX entry.

    Hand-rolled rather than delegated because BibTeX values nest braces freely
    (``title={The {BERT} Model}``) and a regex on ``{[^}]*}`` truncates them.
    """
    fields: dict[str, str] = {}
    body_start = raw.find(",")
    if body_start == -1:
        return fields
    i = body_start + 1
    n = len(raw)
    while i < n:
        eq = raw.find("=", i)
        if eq == -1:
            break
        name = raw[i:eq].strip().strip(",").strip().lower()
        j = eq + 1
        while j < n and raw[j] in " \t\n":
            j += 1
        if j >= n:
            break
        if raw[j] == "{":
            close = match_brace(raw, j)
            if close == -1:
                break
            value = raw[j + 1:close]
            i = close + 1
        elif raw[j] == '"':
            close = raw.find('"', j + 1)
            if close == -1:
                break
            value = raw[j + 1:close]
            i = close + 1
        else:
            close = j
            while close < n and raw[close] not in ",\n}":
                close += 1
            value = raw[j:close]
            i = close
        if name:
            fields[name] = re.sub(r"\s+", " ", value).strip()
        while i < n and raw[i] in " \t\n,":
            i += 1
        if i < n and raw[i] == "}":
            break
    return fields
