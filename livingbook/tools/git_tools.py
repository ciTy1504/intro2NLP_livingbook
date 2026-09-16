"""Git and GitHub publishing tools.

Three guards matter more than the plumbing here:

  1. **Remote assertion.** Every push checks the resolved remote URL contains
     ``git.expected_remote_substring``. The baseline manuscript came from a different
     repository, and pushing agent-authored changes there would be the single worst
     failure this system could have.
  2. **Branch protection.** ``main`` is never committed to directly; the Git Agent's
     contract lists it under ``forbidden_branches``.
  3. **Token handling.** The token is passed as an HTTP extra-header for the duration
     of one command. It is never written into the remote URL, a config file, a commit,
     or a log line — a URL-embedded token would end up in ``.git/config`` on disk.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from ..config import get_config, get_secrets
from ..obs import get_logger
from .http import get_json, github_headers, request
from .registry import Capability, ToolError, ToolUnavailable, tool

_SAFE_BRANCH = re.compile(r"^[A-Za-z0-9._/\-]+$")


class GitError(ToolError):
    pass


async def _git(
    *args: str, cwd: str | None = None, token: str | None = None, timeout: int = 180,
) -> tuple[int, str, str]:
    """Run a git command, injecting auth as a transient header when needed."""
    cfg = get_config()
    cmd: list[str] = ["git"]
    if token:
        # -c is per-invocation: nothing is persisted to .git/config.
        import base64
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraheader=Authorization: Basic {basic}"]
    cmd += list(args)

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd or str(cfg.root),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise ToolUnavailable(f"git {args[0]} timed out after {timeout}s")

    stdout = out.decode("utf-8", "replace")
    stderr = err.decode("utf-8", "replace")
    if token:
        stderr = stderr.replace(token, "[REDACTED]")
        stdout = stdout.replace(token, "[REDACTED]")
    return proc.returncode or 0, stdout, stderr


async def _assert_correct_remote(remote: str = "origin") -> str:
    """Refuse to operate against anything but the Living Book repository."""
    cfg = get_config()
    expected = cfg.get("git.expected_remote_substring", "intro2NLP_livingbook")
    rc, out, err = await _git("remote", "get-url", remote)
    if rc != 0:
        raise GitError(f"remote {remote!r} is not configured: {err.strip()}")
    url = out.strip()
    if expected not in url:
        raise GitError(
            f"refusing to operate on remote {url!r}: it does not contain "
            f"{expected!r}. The Living Book must never push to the baseline repository."
        )
    return url


@tool("git_status", [Capability.GIT, Capability.FS_READ],
      description="Report branch, remote and working-tree status.")
async def git_status() -> dict[str, Any]:
    rc, branch, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
    _, porcelain, _ = await _git("status", "--porcelain")
    _, remotes, _ = await _git("remote", "-v")
    _, head, _ = await _git("rev-parse", "--short", "HEAD")

    changes = [
        {"status": ln[:2].strip(), "path": ln[3:].strip()}
        for ln in porcelain.splitlines() if ln.strip()
    ]
    return {
        "is_repo": rc == 0,
        "branch": branch.strip(),
        "head": head.strip(),
        "clean": not changes,
        "changes": changes,
        "remotes": remotes.strip(),
    }


@tool("git_diff", [Capability.GIT, Capability.FS_READ],
      description="Show the working-tree or staged diff.")
async def git_diff(*, staged: bool = False, paths: list[str] | None = None,
                   stat_only: bool = False, max_chars: int = 60_000) -> dict[str, Any]:
    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    if paths:
        args.append("--")
        args.extend(paths)
    rc, out, err = await _git(*args)
    if rc not in (0, 1):
        raise GitError(f"git diff failed: {err.strip()}")

    _, stat, _ = await _git(*(["diff"] + (["--cached"] if staged else []) + ["--shortstat"]))
    return {"diff": out[:max_chars], "truncated": len(out) > max_chars,
            "shortstat": stat.strip()}


@tool("create_branch", [Capability.GIT],
      description="Create and check out a branch for an agent change.")
async def create_branch(name: str, *, base: str | None = None) -> dict[str, Any]:
    cfg = get_config()
    if not _SAFE_BRANCH.match(name):
        raise GitError(f"unsafe branch name: {name!r}")
    forbidden = set(cfg.agent_spec("git_agent").get("constraints", {})
                    .get("forbidden_branches", ["main", "master"]))
    if name in forbidden:
        raise GitError(f"refusing to create/checkout a protected branch: {name}")

    base = base or cfg.get("git.base_branch", "main")
    rc, out, err = await _git("rev-parse", "--verify", name)
    if rc == 0:
        rc, out, err = await _git("checkout", name)
    else:
        rc, out, err = await _git("checkout", "-b", name, *( [base] if await _ref_exists(base) else []))
    if rc != 0:
        raise GitError(f"could not create branch {name}: {err.strip()}")

    _, current, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
    return {"branch": current.strip(), "created": True, "base": base}


async def _ref_exists(ref: str) -> bool:
    rc, _, _ = await _git("rev-parse", "--verify", ref)
    return rc == 0


@tool("git_commit", [Capability.GIT, Capability.FS_WRITE],
      description="Stage paths and create a commit with provenance trailers.",
      destructive=True)
async def git_commit(
    message: str, *, paths: list[str] | None = None, trailers: dict[str, str] | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    _, branch, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
    branch = branch.strip()
    forbidden = set(cfg.agent_spec("git_agent").get("constraints", {})
                    .get("forbidden_branches", ["main", "master"]))
    if branch in forbidden:
        raise GitError(
            f"refusing to commit directly to {branch!r}; create an agent branch first"
        )

    if paths:
        rc, _, err = await _git("add", "--", *paths)
    else:
        rc, _, err = await _git("add", "-A")
    if rc != 0:
        raise GitError(f"git add failed: {err.strip()}")

    rc, staged, _ = await _git("diff", "--cached", "--name-only")
    if not staged.strip():
        return {"committed": False, "reason": "nothing staged", "branch": branch}

    body = message.rstrip()
    if trailers:
        body += "\n\n" + "\n".join(f"{k}: {v}" for k, v in trailers.items())

    author_name = cfg.get("git.commit_author_name", "livingbook-agent")
    author_email = cfg.get("git.commit_author_email", "")
    args = ["-c", f"user.name={author_name}"]
    if author_email:
        args += ["-c", f"user.email={author_email}"]

    rc, out, err = await _git(*args, "commit", "-m", body)
    if rc != 0:
        raise GitError(f"git commit failed: {err.strip() or out.strip()}")

    _, sha, _ = await _git("rev-parse", "HEAD")
    return {
        "committed": True, "sha": sha.strip(), "short_sha": sha.strip()[:8],
        "branch": branch, "files": [f for f in staged.splitlines() if f],
        "message": body,
    }


@tool("git_push", [Capability.GIT],
      description="Push a branch to the Living Book repository.", destructive=True)
async def git_push(*, branch: str | None = None, remote: str = "origin",
                   set_upstream: bool = True) -> dict[str, Any]:
    cfg = get_config()
    if not cfg.get("git.push_enabled", True):
        return {"pushed": False, "reason": "git.push_enabled is false (dry run)"}

    url = await _assert_correct_remote(remote)
    if not branch:
        _, branch, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
        branch = branch.strip()

    token = get_secrets().get("GITHUB_TOKEN")
    if not token:
        raise ToolUnavailable(
            "GITHUB_TOKEN is not set; cannot push. Add it to secrets/.env "
            "(fine-grained PAT with Contents and Pull requests write access)."
        )

    args = ["push"]
    if set_upstream:
        args.append("-u")
    args += [remote, branch]
    rc, out, err = await _git(*args, token=token, timeout=300)
    if rc != 0:
        raise GitError(f"git push failed: {err.strip() or out.strip()}")

    get_logger().info(f"pushed {branch} to {url}", tool="git_push", status="ok")
    return {"pushed": True, "branch": branch, "remote": remote, "url": url,
            "output": (out + err).strip()[:2000]}


@tool("create_pull_request", [Capability.GIT],
      description="Open a pull request on the Living Book repository.", destructive=True)
async def create_pull_request(
    *, title: str, body: str, head: str, base: str | None = None, draft: bool = False,
) -> dict[str, Any]:
    cfg = get_config()
    secrets = get_secrets()
    token = secrets.get("GITHUB_TOKEN")
    if not token:
        raise ToolUnavailable("GITHUB_TOKEN is not set; cannot open a pull request")

    url = await _assert_correct_remote()
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    if not m:
        raise GitError(f"could not parse owner/repo from remote {url!r}")
    owner, repo = m.group(1), m.group(2)
    base = base or cfg.get("git.base_branch", "main")

    headers = {**github_headers(), "Authorization": f"Bearer {token}"}
    resp = await request(
        "POST", f"https://api.github.com/repos/{owner}/{repo}/pulls",
        json_body={"title": title, "body": body, "head": head, "base": base,
                   "draft": draft},
        headers=headers, timeout=60,
    )
    if resp.status_code == 422:
        # Already open for this head — return the existing one rather than failing a
        # pipeline that is otherwise complete.
        existing = await get_json(
            f"https://api.github.com/repos/{owner}/{repo}/pulls",
            params={"head": f"{owner}:{head}", "state": "open"},
            headers=headers, timeout=45)
        if existing:
            pr = existing[0]
            return {"created": False, "existing": True, "number": pr["number"],
                    "url": pr["html_url"]}
        raise GitError(f"pull request rejected: {resp.text[:400]}")
    if resp.status_code >= 400:
        raise GitError(f"pull request failed (HTTP {resp.status_code}): {resp.text[:400]}")

    pr = resp.json()
    get_logger().info(f"opened PR #{pr['number']}: {pr['html_url']}",
                      tool="create_pull_request", status="ok")
    return {"created": True, "number": pr["number"], "url": pr["html_url"],
            "head": head, "base": base}


@tool("git_log", [Capability.GIT, Capability.FS_READ],
      description="Read recent commit history.")
async def git_log(*, limit: int = 20, branch: str | None = None) -> list[dict[str, str]]:
    args = ["log", f"-{limit}", "--pretty=format:%H%x1f%an%x1f%aI%x1f%s"]
    if branch:
        args.append(branch)
    rc, out, err = await _git(*args)
    if rc != 0:
        return []
    entries = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            entries.append({"sha": parts[0], "short_sha": parts[0][:8],
                            "author": parts[1], "date": parts[2], "subject": parts[3]})
    return entries


@tool("init_repository", [Capability.GIT, Capability.FS_WRITE],
      description="Initialise the local repository and attach the Living Book remote.",
      destructive=True)
async def init_repository(*, remote_url: str | None = None) -> dict[str, Any]:
    cfg = get_config()
    url = remote_url or cfg.get("git.remote_url")
    expected = cfg.get("git.expected_remote_substring", "intro2NLP_livingbook")
    if expected not in (url or ""):
        raise GitError(f"refusing to attach remote {url!r}: expected {expected!r} in the URL")

    rc, _, _ = await _git("rev-parse", "--git-dir")
    created = False
    if rc != 0:
        rc, out, err = await _git("init", "-b", cfg.get("git.base_branch", "main"))
        if rc != 0:
            raise GitError(f"git init failed: {err.strip()}")
        created = True

    rc, _, _ = await _git("remote", "get-url", "origin")
    if rc == 0:
        await _git("remote", "set-url", "origin", url)
    else:
        await _git("remote", "add", "origin", url)

    return {"initialised": created, "remote": url,
            "branch": (await _git("rev-parse", "--abbrev-ref", "HEAD"))[1].strip()}


def branch_name_for(topic: str, when: datetime | None = None) -> str:
    """agent/research/<date>-<topic-slug>, per the architecture."""
    cfg = get_config()
    fmt = cfg.get("git.branch_format", "agent/research/{date}-{topic}")
    when = when or datetime.now(timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:48] or "update"
    return fmt.format(date=when.strftime("%Y-%m-%d"), topic=slug)
