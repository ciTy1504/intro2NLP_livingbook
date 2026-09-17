"""Book build and deterministic QA tools.

``build_book`` compiles the manuscript in a **disposable copied workspace**, never in
``manuscript/`` itself. Two reasons, both measured (AUDIT.md §7):

  * `minted` v3 fails under `-output-directory` because it cannot resolve
    TEXMF_OUTPUT_DIRECTORY, so an out-of-tree build is not an option;
  * an in-place build in `manuscript/` leaves `.aux`, `.toc` and `_minted-*` files in
    the tree that the Git Agent would then try to commit.

Everything else here is deterministic QA: no LLM, fast enough to run on every patch,
and able to fail a pipeline on facts rather than opinions.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import get_config
from ..knowledge.bib import Bibliography
from ..knowledge.latex import LatexParser, match_brace
from ..obs import get_logger
from .http import request
from .registry import Capability, ToolError, ToolUnavailable, tool

#: Environments whose contents are literal text rather than LaTeX. They must be
#: removed before any structural counting — see latex_lint.
_CODE_ENV_RE = re.compile(
    r"\\begin\{(minted|verbatim|lstlisting|Verbatim|alltt)\}"
    r"(?:\[[^\]]*\])?(?:\{[^}]*\})?.*?\\end\{\1\}",
    re.S,
)
_MACRO_DEF_RE = re.compile(r"\\(?:new|renew|provide)command|\\def\s*\\")

_LATEX_ERROR_RE = re.compile(r"^! (.+)$", re.M)
_UNDEFINED_REF_RE = re.compile(r"Reference `([^']+)' on page \d+ undefined")
_UNDEFINED_CITE_RE = re.compile(r"Citation `([^']+)' on page \d+ undefined")
_MISSING_FILE_RE = re.compile(r"File `([^']+)' not found")


def _find_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    # winget installs MiKTeX per-user and the PATH entry is not picked up by an
    # already-running shell, so look in the known location too.
    import os
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / f"{name}.exe",
        Path("C:/Program Files/MiKTeX/miktex/bin/x64") / f"{name}.exe",
        Path("/usr/bin") / name,
        Path("/usr/local/bin") / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _build_env(*binaries: str | None) -> dict[str, str]:
    """PATH the LaTeX toolchain can actually call through.

    Two hops have to work, and neither uses the absolute path _find_binary() resolves.
    xelatex spawns `latexminted` by name, and `latexminted.exe` is a MiKTeX shim that
    in turn spawns bare `python` — which on Windows hits the Microsoft Store alias stub
    ("Python was not found") unless the real interpreter's directory comes first.

    The symptom was a build that looked fine: 337 pages, no undefined references, and
    every code listing silently unhighlighted behind `Package minted Error: minted
    executable is unavailable`. Putting sys.executable's directory and the TeX binary
    directory on the child's PATH makes the build independent of how the process was
    started, which is the point — the daemon inherits whatever PATH its launcher had.
    """
    import os
    import sys
    env = dict(os.environ)
    dirs = [str(Path(sys.executable).parent)]
    dirs += [str(Path(b).parent) for b in binaries if b]
    seen: list[str] = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    env["PATH"] = os.pathsep.join(seen + [env.get("PATH", "")])
    return env


@tool("build_book", [Capability.BUILD],
      description="Compile the manuscript with XeLaTeX in a disposable workspace.")
async def build_book(
    *, source_dir: str | None = None, full: bool = True, timeout: int | None = None,
) -> dict[str, Any]:
    """Build the book and report errors, undefined references and missing files.

    ``full=True`` runs xelatex → bibtex → xelatex → xelatex so that citations,
    cross-references and the TOC all resolve; ``full=False`` runs a single pass, which
    is enough to catch syntax errors quickly during drafting.
    """
    cfg = get_config()
    log = get_logger()
    xelatex = _find_binary("xelatex")
    if not xelatex:
        raise ToolUnavailable("xelatex not found; install MiKTeX or TeX Live")

    src = Path(source_dir) if source_dir else cfg.manuscript_dir
    if not src.is_absolute():
        src = cfg.root / src
    if not src.exists():
        raise ToolError(f"manuscript directory not found: {src}")

    timeout = timeout or int(cfg.get("qa.build_timeout_seconds", 900))
    main_name = cfg.get("project.main_tex", "main.tex")

    with tempfile.TemporaryDirectory(prefix="livingbook_build_") as tmp:
        workspace = Path(tmp) / "manuscript"
        shutil.copytree(
            src, workspace,
            ignore=shutil.ignore_patterns("_minted*", "*.aux", "*.log", "*.toc",
                                          "*.out", "*.bbl", "*.blg", "*.fls",
                                          "*.fdb_latexmk", "*.synctex.gz", "*.pdf"),
        )

        passes: list[dict[str, Any]] = []
        log_text = ""

        env = _build_env(xelatex, _find_binary("latexminted"))

        async def run(cmd: list[str], label: str) -> dict[str, Any]:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(workspace), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise ToolUnavailable(f"{label} exceeded {timeout}s")
            return {"pass": label, "returncode": proc.returncode,
                    "output": out.decode("utf-8", "replace")}

        xe_cmd = [xelatex, "-interaction=nonstopmode", "-shell-escape", main_name]
        passes.append(await run(xe_cmd, "xelatex-1"))

        if full:
            bibtex = _find_binary("bibtex")
            if bibtex:
                try:
                    passes.append(await run([bibtex, Path(main_name).stem], "bibtex"))
                except ToolUnavailable:
                    pass
            passes.append(await run(xe_cmd, "xelatex-2"))
            passes.append(await run(xe_cmd, "xelatex-3"))

        log_file = workspace / f"{Path(main_name).stem}.log"
        if log_file.exists():
            log_text = log_file.read_text(encoding="utf-8", errors="replace")

        pdf = workspace / f"{Path(main_name).stem}.pdf"
        pdf_bytes = pdf.stat().st_size if pdf.exists() else 0
        pages = _page_count(log_text)

        errors = sorted(set(_LATEX_ERROR_RE.findall(log_text)))
        undefined_refs = sorted(set(_UNDEFINED_REF_RE.findall(log_text)))
        undefined_cites = sorted(set(_UNDEFINED_CITE_RE.findall(log_text)))
        missing_files = sorted(set(_MISSING_FILE_RE.findall(log_text)))

        # A PDF with pages is the real signal. xelatex exits non-zero for warnings
        # that do not prevent a correct document, and treating those as build failures
        # would block every patch on pre-existing noise.
        ok = pdf_bytes > 0 and not errors

        result = {
            "ok": ok,
            "pdf_bytes": pdf_bytes,
            "pages": pages,
            "errors": errors[:40],
            "undefined_references": undefined_refs[:40],
            "undefined_citations": undefined_cites[:40],
            "missing_files": missing_files[:40],
            "passes": [{"pass": p["pass"], "returncode": p["returncode"]} for p in passes],
            "log_tail": log_text[-6000:] if log_text else
                        passes[-1]["output"][-6000:] if passes else "",
        }
        log.info(
            f"build: {'ok' if ok else 'FAILED'} — {pages} pages, "
            f"{len(errors)} errors, {len(undefined_refs)} undefined refs, "
            f"{len(undefined_cites)} undefined citations",
            tool="build_book", status="ok" if ok else "error",
        )
        return result


def _defines_macros(text: str) -> bool:
    return bool(_MACRO_DEF_RE.search(text))


def _page_count(log_text: str) -> int:
    m = re.search(r"Output written on .*?\((\d+) pages?", log_text)
    return int(m.group(1)) if m else 0


@tool("latex_lint", [Capability.FS_READ],
      description="Deterministic LaTeX structural checks without compiling.")
async def latex_lint(*, files: list[str] | None = None) -> dict[str, Any]:
    """Catch structural breakage in seconds, before paying for a full build."""
    cfg = get_config()
    manuscript = cfg.manuscript_dir
    parser = LatexParser(manuscript, cfg.get("project.main_tex", "main.tex"))
    targets = files or parser.input_order()

    problems: list[dict[str, Any]] = []
    for rel in targets:
        path = manuscript / rel
        if not path.exists():
            problems.append({"file": rel, "severity": "error", "kind": "missing_file",
                             "detail": "referenced by \\input but not present"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        stripped = re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)

        # Code blocks must be excised before any brace or environment counting.
        # Their contents are literal text, not LaTeX: a Python regex such as
        # r"\\boxed\{([^}]*)\}" inside a minted block is perfectly valid but
        # unbalanced as braces, and counting it produces a false error on a file
        # that compiles cleanly. A linter that cries wolf would block every patch.
        code_free = _CODE_ENV_RE.sub(" ", stripped)

        opens = code_free.count("{") - code_free.count("\\{")
        closes = code_free.count("}") - code_free.count("\\}")
        if opens != closes:
            problems.append({"file": rel, "severity": "error", "kind": "unbalanced_braces",
                             "detail": f"{opens} open vs {closes} close "
                                       "(code blocks excluded)"})

        begins = re.findall(r"\\begin\{([^}]+)\}", code_free)
        ends = re.findall(r"\\end\{([^}]+)\}", code_free)
        for env in set(begins) | set(ends):
            if begins.count(env) != ends.count(env):
                problems.append({
                    "file": rel, "severity": "error", "kind": "unbalanced_environment",
                    "detail": f"\\begin{{{env}}} x{begins.count(env)} vs "
                              f"\\end{{{env}}} x{ends.count(env)}",
                })

        for m in re.finditer(r"\\(section|subsection|subsubsection|chapter)\*?\s*\{",
                             code_free):
            if match_brace(code_free, code_free.index("{", m.end() - 1)) == -1:
                problems.append({"file": rel, "severity": "error", "kind": "unterminated_heading",
                                 "detail": f"\\{m.group(1)} at offset {m.start()}"})

        # style.tex *defines* \bookimage, so the macro legitimately appears there
        # without arguments; likewise inside any \newcommand body.
        if "\\bookimage" in code_free and not _defines_macros(code_free):
            for m in re.finditer(r"\\bookimage", code_free):
                tail = code_free[m.end():m.end() + 40]
                if not re.match(r"\s*(\[[^\]]*\])?\s*\{", tail):
                    problems.append({"file": rel, "severity": "error",
                                     "kind": "malformed_bookimage",
                                     "detail": "\\bookimage without a key argument"})

    return {
        "ok": not any(p["severity"] == "error" for p in problems),
        "files_checked": len(targets),
        "problems": problems,
        "error_count": sum(1 for p in problems if p["severity"] == "error"),
    }


@tool("bib_validate", [Capability.FS_READ],
      description="Validate references.bib structurally and against the manuscript.")
async def bib_validate() -> dict[str, Any]:
    cfg = get_config()
    bib = Bibliography(cfg.bib_path)
    problems = bib.validate()

    parser = LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex"))
    book = parser.parse()
    cited = {c for n in book.nodes for c in n.cites}
    keys = bib.keys()

    orphans = sorted(cited - keys)
    uncited = sorted(keys - cited)
    for key in orphans:
        problems.append({"key": key, "severity": "error", "kind": "orphan_citation",
                         "detail": "cited in the manuscript but absent from references.bib"})

    return {
        "ok": not any(p["severity"] == "error" for p in problems),
        "entries": len(bib),
        "cited_keys": len(cited),
        "orphan_citations": orphans,
        "uncited_entries": uncited,
        "uncited_count": len(uncited),
        "duplicate_titles": bib.dedupe_report(),
        "problems": problems,
        "error_count": sum(1 for p in problems if p["severity"] == "error"),
    }


@tool("check_figures", [Capability.FS_READ],
      description="Check every referenced figure resolves, with alt text and licence.")
async def check_figures() -> dict[str, Any]:
    cfg = get_config()
    parser = LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex"))
    book = parser.parse()
    images_dir = cfg.images_dir

    missing: list[dict[str, Any]] = []
    present: list[str] = []
    no_caption: list[str] = []

    for node in book.nodes:
        for fig in node.figures:
            found = None
            if fig.kind == "includegraphics" and fig.path:
                direct = cfg.manuscript_dir / fig.path
                if direct.exists():
                    found = direct
            if not found:
                for ext in (".png", ".jpg", ".jpeg", ".pdf"):
                    candidate = images_dir / f"{fig.key}{ext}"
                    if candidate.exists():
                        found = candidate
                        break
            if found:
                present.append(fig.key)
            else:
                missing.append({
                    "key": fig.key, "kind": fig.kind, "file": node.file, "line": fig.line,
                    "section": node.title, "requirement": fig.requirement[:400],
                })
            if not fig.caption:
                no_caption.append(fig.key)

    # SOURCES.md is the manuscript's existing licence ledger; a figure missing from it
    # has no recorded provenance.
    sources_md = images_dir / "SOURCES.md"
    documented: set[str] = set()
    if sources_md.exists():
        text = sources_md.read_text(encoding="utf-8", errors="replace")
        documented = set(re.findall(r"`([A-Za-z0-9_\-]+)\.(?:png|jpg|jpeg)`", text))

    undocumented = sorted(set(present) - documented)

    return {
        "ok": not missing,
        "total": len(present) + len(missing),
        "present": len(present),
        "missing": missing,
        "without_caption": sorted(set(no_caption)),
        "undocumented_in_sources_md": undocumented,
        "orphan_images": sorted(
            p.name for p in images_dir.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and p.stem not in set(present)
        ) if images_dir.exists() else [],
    }


@tool("search_bibliography", [Capability.FS_READ],
      description="Search the book's own bibliography for a source supporting a claim.")
async def search_bibliography(*, query: str, limit: int = 6) -> list[dict[str, Any]]:
    """Look in references.bib before going to the web.

    The book already cites 228 curated sources, and for a claim about content the book
    covers, the primary source is often one of them. Measured: the claim "DeepSeek-V3
    reports 85-90% acceptance for the extra predicted token" sent the citation finder
    to the web, which returned a third-party analysis; the verifier correctly rejected
    it as needs_primary, and the pipeline looped — while `deepseekai2024v3`, the
    DeepSeek-V3 Technical Report, had been sitting in references.bib the whole time.

    Reusing an existing key is better than adding a new entry on every axis that
    matters here: it is the source the author already vetted, it keeps the
    bibliography from growing duplicates, and it costs no network call.
    """
    cfg = get_config()
    bib = Bibliography(cfg.bib_path)

    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{2,}", query.lower())
             if t not in _BIB_STOPWORDS]
    if not terms:
        return []

    # Whole words, not substrings or prefixes. "extra" must match neither
    # "extracting" nor "vincent2008extracting" — with a leading boundary alone it
    # still matched the start of "Extracting", and the DeepSeek-V3 report came
    # back ranked joint-fourth behind three unrelated papers. A trailing boundary
    # still allows "deepseek" to match "DeepSeek-V3", where the hyphen ends the word.
    matchers = {t: re.compile(rf"\b{re.escape(t)}\b", re.I)
                for t in terms}

    fields_for = {}
    for key, entry in bib.entries.items():
        fields_for[key] = (
            (entry.fields.get("title") or "").lower(),
            f"{key} {entry.fields.get('title','')} {entry.fields.get('author','')} "
            f"{entry.fields.get('journal','')} {entry.fields.get('booktitle','')}".lower(),
        )

    # A term matching three entries identifies one of them; a term matching two
    # hundred identifies nothing. Weight by how rare the term is in this bibliography.
    breadth = {t: sum(1 for _, hay in fields_for.values() if m.search(hay)) or 1
               for t, m in matchers.items()}
    total = len(bib.entries) or 1

    scored: list[tuple[float, dict[str, Any]]] = []
    for key, entry in bib.entries.items():
        title, haystack = fields_for[key]

        hits = [t for t, m in matchers.items() if m.search(haystack)]
        if not hits:
            continue
        score = 0.0
        for t in hits:
            rarity = math.log(total / breadth[t]) + 1.0
            # Title matches carry the weight; a term that only appears in the key or
            # the venue is weak evidence that this is the right entry.
            score += rarity * (2.0 if matchers[t].search(title) else 1.0)
        score /= len(terms)
        scored.append((score, {
            "bib_key": key,
            "title": entry.fields.get("title", ""),
            "authors": entry.fields.get("author", ""),
            "year": entry.fields.get("year", ""),
            "doi": entry.fields.get("doi", ""),
            "url": entry.fields.get("url", ""),
            "venue": entry.fields.get("journal") or entry.fields.get("booktitle", ""),
            "source": "book_bibliography",
            "already_in_bibliography": True,
            "matched_terms": hits,
            "score": round(score, 3),
        }))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


#: Words too common in a claim to identify an entry.
_BIB_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "has", "have", "are", "was",
    "reports", "report", "achieves", "achieve", "shows", "show", "using", "used",
    "model", "models", "token", "tokens", "method", "methods", "approach", "results",
}


#: Hosts that appear in the book as examples, not as links. `localhost:8000` is the
#: vLLM server a reader starts in chapter 2.6; reporting it as a broken link buries
#: the real ones. RFC 2606 reserves example.com/.invalid for exactly this purpose.
ILLUSTRATIVE_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                      "example.com", "example.org", "example.net",
                      "your-domain.com", "api.example.com")


#: Signatures of a host declining to be probed rather than a URL being gone.
_BOT_REFUSAL = ("HTTP 403", "HTTP 405", "HTTP 429", "blocked",
                "RemoteProtocolError", "Server disconnected")


def _is_bot_refusal(exc: Exception | None) -> bool:
    return exc is not None and any(m in str(exc) for m in _BOT_REFUSAL)


def _is_illustrative(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return (host in ILLUSTRATIVE_HOSTS
            or host.endswith(".local") or host.endswith(".invalid")
            or host.startswith("192.168.") or host.startswith("10."))


@tool("check_links", [Capability.FS_READ, Capability.FETCH],
      description="Check URLs in the manuscript and bibliography resolve.")
async def check_links(
    *, limit: int = 60, timeout: float = 15.0, concurrency: int = 8,
) -> dict[str, Any]:
    cfg = get_config()
    urls: dict[str, list[str]] = {}

    url_re = re.compile(r"https?://[^\s{}\\,\)\]\"']+")
    parser = LatexParser(cfg.manuscript_dir, cfg.get("project.main_tex", "main.tex"))
    for rel in parser.input_order():
        path = cfg.manuscript_dir / rel
        if not path.exists():
            continue
        for m in url_re.finditer(path.read_text(encoding="utf-8", errors="replace")):
            urls.setdefault(m.group(0).rstrip(".,;"), []).append(rel)
    if cfg.bib_path.exists():
        for m in url_re.finditer(cfg.bib_path.read_text(encoding="utf-8", errors="replace")):
            urls.setdefault(m.group(0).rstrip(".,;"), []).append("references.bib")

    skipped = [u for u in urls if _is_illustrative(u)]
    targets = [u for u in urls if u not in set(skipped)][:limit]
    sem = asyncio.Semaphore(concurrency)
    broken: list[dict[str, Any]] = []
    #: Reachable, but the host will not answer an automated probe. Reported apart from
    #: `broken` so a real 404 is not lost among them.
    refused: list[dict[str, Any]] = []

    async def probe(method: str, url: str):
        return await request(method, url, timeout=timeout, retries=1,
                             browser_ua=True, polite=False)

    async def check(url: str) -> None:
        async with sem:
            # HEAD first because it is cheap, then GET — but the GET has to run when
            # the HEAD *raised*, not only when it returned >= 400. request() turns a
            # 403 into ToolUnavailable and never returns it, so the old fallback was
            # unreachable for exactly the hosts that need it: aclanthology.org and
            # doi.org disconnect on HEAD, and Wikipedia 403s it. Every one of those
            # URLs opens fine in a browser.
            first: Exception | None = None
            try:
                resp = await probe("HEAD", url)
                if resp.status_code < 400:
                    return
            except Exception as exc:
                first = exc
            try:
                resp = await probe("GET", url)
            except Exception as exc:
                # A host that refuses to be probed is telling us about its bot policy,
                # not about the link. Flagging it trains the reader to ignore this
                # report, which is worse than not running it.
                if _is_bot_refusal(exc) or _is_bot_refusal(first):
                    refused.append({"url": url, "why": str(exc)[:120],
                                    "in": sorted(set(urls[url]))[:3]})
                    return
                broken.append({"url": url,
                               "status": f"{type(exc).__name__}: {exc}"[:160],
                               "in": sorted(set(urls[url]))[:3]})
                return
            if resp.status_code in (403, 405, 429):
                refused.append({"url": url, "why": f"HTTP {resp.status_code}",
                                "in": sorted(set(urls[url]))[:3]})
            elif resp.status_code >= 400:
                broken.append({"url": url, "status": resp.status_code,
                               "in": sorted(set(urls[url]))[:3]})

    await asyncio.gather(*(check(u) for u in targets))

    return {
        "ok": not broken,
        "checked": len(targets),
        "total_urls": len(urls),
        "skipped_illustrative": sorted(skipped),
        "refused_probe": sorted(refused, key=lambda d: str(d["url"])),
        "broken": sorted(broken, key=lambda d: str(d["url"])),
    }


@tool("run_tests", [Capability.BUILD],
      description="Run the repository test suite.")
async def run_tests(*, path: str = "tests", timeout: int = 600) -> dict[str, Any]:
    cfg = get_config()
    import sys
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", path, "-q", "--no-header",
        cwd=str(cfg.root),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise ToolUnavailable(f"tests exceeded {timeout}s")

    text = out.decode("utf-8", "replace")
    m = re.search(r"(\d+) passed", text)
    f = re.search(r"(\d+) failed", text)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "passed": int(m.group(1)) if m else 0,
        "failed": int(f.group(1)) if f else 0,
        "output": text[-8000:],
    }
