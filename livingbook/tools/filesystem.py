"""Filesystem tools, scoped to the repository.

Reads are unrestricted within the repo; writes go through
``AgentContext.check_write_path``, which enforces the per-agent ``write_paths`` and
``forbidden_paths`` from ``config/agents.yaml``. That is what makes the permission
matrix real — FS_WRITE on its own would otherwise mean "write anywhere".

``patch_file`` is the primary editing primitive: it applies an exact-match replacement
rather than rewriting a file wholesale, so a writer cannot accidentally reformat a
section it was not asked to touch.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import re
from pathlib import Path
from typing import Any

from ..config import get_config
from .registry import AgentContext, Capability, ToolError, tool

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "_minted",
              ".build", "state", "logs", "secrets"}


def _repo_path(rel: str | Path) -> Path:
    root = get_config().root
    p = Path(rel)
    resolved = (root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ToolError(f"path escapes the repository: {rel}")
    return resolved


def _rel(path: Path) -> str:
    return path.relative_to(get_config().root.resolve()).as_posix()


@tool("read_file", [Capability.FS_READ],
      description="Read a repository file, optionally a line range.")
async def read_file(
    path: str, *, start_line: int | None = None, end_line: int | None = None,
    max_chars: int = 200_000,
) -> dict[str, Any]:
    target = _repo_path(path)
    if not target.exists():
        raise ToolError(f"no such file: {path}")
    if target.is_dir():
        raise ToolError(f"{path} is a directory; use list_files")

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = min(len(lines), end_line or len(lines))
        selected = lines[s - 1:e]
        content = "\n".join(selected)
        return {"path": _rel(target), "content": content[:max_chars],
                "start_line": s, "end_line": e, "total_lines": len(lines),
                "sha256": hashlib.sha256(text.encode()).hexdigest()}

    return {"path": _rel(target), "content": text[:max_chars],
            "start_line": 1, "end_line": len(lines), "total_lines": len(lines),
            "truncated": len(text) > max_chars,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


@tool("list_files", [Capability.FS_READ],
      description="List files under a repository directory, optionally by glob.")
async def list_files(
    directory: str = ".", *, pattern: str = "*", recursive: bool = True, limit: int = 500,
) -> list[dict[str, Any]]:
    base = _repo_path(directory)
    if not base.exists():
        raise ToolError(f"no such directory: {directory}")

    out: list[dict[str, Any]] = []
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    for p in iterator:
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        out.append({"path": _rel(p), "bytes": p.stat().st_size,
                    "suffix": p.suffix})
        if len(out) >= limit:
            break
    return sorted(out, key=lambda d: d["path"])


@tool("search_repository", [Capability.FS_READ],
      description="Regex search across repository files.")
async def search_repository(
    pattern: str, *, glob: str = "**/*.tex", max_results: int = 80,
    directory: str = "manuscript", ignore_case: bool = False,
) -> list[dict[str, Any]]:
    base = _repo_path(directory)
    flags = re.I if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc

    results: list[dict[str, Any]] = []
    for p in base.rglob("*"):
        if len(results) >= max_results:
            break
        if not p.is_file() or any(part in _SKIP_DIRS for part in p.parts):
            continue
        if not fnmatch.fnmatch(p.relative_to(base).as_posix(), glob.replace("**/", "")) \
           and not fnmatch.fnmatch(p.as_posix(), glob):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append({"path": _rel(p), "line": i, "text": line.strip()[:300]})
                if len(results) >= max_results:
                    break
    return results


@tool("write_file", [Capability.FS_WRITE],
      description="Write a file within the agent's permitted write scope.",
      destructive=True)
async def write_file(
    ctx: AgentContext, path: str, content: str, *, create_dirs: bool = True,
) -> dict[str, Any]:
    target = ctx.check_write_path(path)
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)

    before = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    target.write_text(content, encoding="utf-8", newline="\n")
    return {
        "path": _rel(target),
        "bytes": len(content.encode()),
        "created": not before,
        "diff": _unified(before, content, _rel(target)),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


@tool("patch_file", [Capability.FS_WRITE],
      description="Replace an exact string in a file; the primary editing primitive.",
      destructive=True)
async def patch_file(
    ctx: AgentContext, path: str, old: str, new: str, *, count: int = 1,
) -> dict[str, Any]:
    """Exact-match replacement.

    Exact matching is the point: it fails loudly when the file is not what the agent
    believed it to be, instead of writing a plausible-looking edit into the wrong
    place. A verdict that has gone stale should fail here, not corrupt a chapter.
    """
    target = ctx.check_write_path(path)
    if not target.exists():
        raise ToolError(f"cannot patch missing file: {path}")

    before = target.read_text(encoding="utf-8", errors="replace")
    occurrences = before.count(old)
    if occurrences == 0:
        raise ToolError(
            f"patch target not found in {path}. The file may have changed since it "
            f"was read. Looked for {old[:120]!r}…"
        )
    if count == 1 and occurrences > 1:
        raise ToolError(
            f"patch target appears {occurrences} times in {path}; "
            "include more surrounding context to make it unique"
        )

    after = before.replace(old, new, occurrences if count < 0 else count)
    target.write_text(after, encoding="utf-8", newline="\n")
    return {
        "path": _rel(target),
        "replacements": occurrences if count < 0 else min(count, occurrences),
        "diff": _unified(before, after, _rel(target)),
        "lines_added": after.count("\n") - before.count("\n"),
        "sha256": hashlib.sha256(after.encode()).hexdigest(),
    }


@tool("append_to_file", [Capability.FS_WRITE],
      description="Append text to a file within the agent's write scope.",
      destructive=True)
async def append_to_file(ctx: AgentContext, path: str, content: str) -> dict[str, Any]:
    target = ctx.check_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    return {"path": _rel(target), "appended_bytes": len(content.encode())}


def _unified(before: str, after: str, path: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3,
    )
    return "".join(diff)[:40_000]


def make_unified_diff(before: str, after: str, path: str) -> str:
    """Exposed for the writer skill, which builds diffs without writing."""
    return _unified(before, after, path)


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply a unified diff produced by ``make_unified_diff``.

    Deliberately strict — every context line must match. A fuzzy patcher would let a
    draft written against a stale read land in the wrong place, which is exactly the
    failure the exact-match policy elsewhere is designed to prevent.
    """
    lines = original.splitlines(keepends=True)
    out: list[str] = []
    src = 0
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    diff_lines = diff.splitlines()
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        m = hunk_re.match(line)
        if not m:
            i += 1
            continue
        start = int(m.group(1))
        # Copy untouched lines before the hunk.
        while src < start - 1:
            out.append(lines[src])
            src += 1
        i += 1
        while i < len(diff_lines) and not hunk_re.match(diff_lines[i]):
            d = diff_lines[i]
            if d.startswith("---") or d.startswith("+++"):
                i += 1
                continue
            if d.startswith(" "):
                if src >= len(lines) or lines[src].rstrip("\n") != d[1:]:
                    raise ToolError(
                        f"diff context mismatch at source line {src + 1}: "
                        f"expected {d[1:]!r}"
                    )
                out.append(lines[src])
                src += 1
            elif d.startswith("-"):
                if src >= len(lines) or lines[src].rstrip("\n") != d[1:]:
                    found = repr(lines[src].rstrip("\n")) if src < len(lines) else "EOF"
                    raise ToolError(
                        f"diff removal mismatch at source line {src + 1}: "
                        f"expected {d[1:]!r}, found {found}"
                    )
                src += 1
            elif d.startswith("+"):
                out.append(d[1:] + "\n")
            i += 1

    out.extend(lines[src:])
    return "".join(out)
