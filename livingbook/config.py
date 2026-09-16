"""Configuration and secrets layer.

Two separate things live here on purpose:

``Config``  — everything in ``config/*.yaml``. Non-sensitive, committed, diffable.
``Secrets`` — everything in ``secrets/.env`` and the key pool file. Never committed,
              never logged, never written into an artifact.

Agents receive a ``Config``. Only the LLM provider, the git tool and the email tool
ever touch ``Secrets``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_get(data: Any, dotted: str, default: Any = None) -> Any:
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class Config:
    """Read-only view over the merged YAML configuration."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else REPO_ROOT
        cfg_dir = self.root / "config"
        self.data: dict[str, Any] = self._load(cfg_dir / "config.yaml")
        self.agents: dict[str, Any] = self._load(cfg_dir / "agents.yaml")
        self.sources: dict[str, Any] = self._load(cfg_dir / "sources.yaml")

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self.data, dotted, default)

    def source(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self.sources, dotted, default)

    def agent_spec(self, name: str) -> dict[str, Any]:
        agents = self.agents.get("agents", {})
        if name not in agents:
            raise KeyError(f"No agent named {name!r} in config/agents.yaml")
        spec = dict(self.agents.get("defaults", {}))
        spec.update(agents[name])
        spec["name"] = name
        return spec

    def all_agent_names(self) -> list[str]:
        return sorted(self.agents.get("agents", {}))

    # -- paths -------------------------------------------------------------
    def path(self, dotted: str, default: str | None = None) -> Path:
        """Resolve a configured relative path against the repo root."""
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"No path configured at {dotted!r}")
        p = Path(raw)
        return p if p.is_absolute() else self.root / p

    @property
    def manuscript_dir(self) -> Path:
        return self.root / self.get("project.manuscript_dir", "manuscript")

    @property
    def main_tex(self) -> Path:
        return self.manuscript_dir / self.get("project.main_tex", "main.tex")

    @property
    def bib_path(self) -> Path:
        return self.manuscript_dir / self.get("project.bib_file", "references.bib")

    @property
    def images_dir(self) -> Path:
        return self.manuscript_dir / self.get("project.images_dir", "images")

    @property
    def state_db(self) -> Path:
        return self.path("paths.state_db", "state/livingbook.db")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("paths.artifacts", "state/artifacts")

    @property
    def log_dir(self) -> Path:
        return self.path("observability.log_dir", "logs")

    def model_chain(self, role: str) -> list[str]:
        chain = self.get(f"llm.model_roles.{role}")
        if not chain:
            raise KeyError(f"No model chain configured for role {role!r}")
        return list(chain)


@dataclass
class Secrets:
    """Credentials, loaded from the environment and ``secrets/.env``.

    ``__repr__`` is overridden so a Secrets object can never leak into a log line
    or a traceback.
    """

    root: Path = REPO_ROOT
    _env: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        env_file = self.root / "secrets" / ".env"
        if env_file.exists():
            self._env.update(self._parse_env(env_file))
        # Real environment variables win over the file.
        for key in (
            "GEMINI_KEYPOOL_PATH", "GEMINI_API_KEYS", "GITHUB_TOKEN",
            "GMAIL_USER", "GMAIL_PASS", "EMAIL_RECIPIENTS",
            "SEMANTIC_SCHOLAR_API_KEY", "BRAVE_SEARCH_API_KEY",
            "SERPER_API_KEY", "TAVILY_API_KEY", "CONTACT_EMAIL",
        ):
            val = os.environ.get(key)
            if val:
                self._env[key] = val

    @staticmethod
    def _parse_env(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'").strip('"')
            if value:
                out[key.strip()] = value
        return out

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._env.get(key, default)

    def require(self, key: str) -> str:
        value = self._env.get(key)
        if not value:
            raise MissingSecret(
                f"{key} is not set. Add it to secrets/.env "
                f"(template: config/secrets.example.env)."
            )
        return value

    def has(self, key: str) -> bool:
        return bool(self._env.get(key))

    @property
    def contact_email(self) -> str:
        return self.get("CONTACT_EMAIL") or "rndvcs@gmail.com"

    def email_recipients(self) -> list[str]:
        raw = self.get("EMAIL_RECIPIENTS", "") or ""
        return [r.strip() for r in raw.split(",") if r.strip()]

    # -- key pool ----------------------------------------------------------
    def gemini_keys(self, accepted_prefixes: Iterable[str]) -> list[str]:
        """Load Gemini keys from the pool file plus any inline GEMINI_API_KEYS.

        Reads ``utf-8-sig`` because the shared pool file carries a BOM, and skips
        ``#`` comment lines — both behaviours inherited from the original
        ``gemini_pool.py`` in the LLM project.
        """
        prefixes = tuple(accepted_prefixes)
        keys: list[str] = []

        pool_path = self.get("GEMINI_KEYPOOL_PATH") or "secrets/gemini_keypool.txt"
        p = Path(pool_path)
        if not p.is_absolute():
            p = self.root / p
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not prefixes or line.startswith(prefixes):
                    keys.append(line)

        inline = self.get("GEMINI_API_KEYS", "") or ""
        for k in inline.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

        if not keys:
            raise MissingSecret(
                f"No Gemini API keys found. Expected a key pool at {p} "
                f"or GEMINI_API_KEYS in secrets/.env."
            )
        # Preserve order but drop duplicates.
        seen: set[str] = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"<Secrets {len(self._env)} entries (redacted)>"

    __str__ = __repr__


class MissingSecret(RuntimeError):
    """Raised when a required credential is absent."""


_DEFAULT_REDACTIONS = (
    r"AIzaSy[A-Za-z0-9_\-]{20,}",
    r"AQ\.Ab8[A-Za-z0-9_\-]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"gho_[A-Za-z0-9]{20,}",
)


class Redactor:
    """Strips credential-shaped substrings from anything bound for a log or artifact."""

    def __init__(self, patterns: Iterable[str] | None = None) -> None:
        pats = list(patterns) if patterns else list(_DEFAULT_REDACTIONS)
        self._re = re.compile("|".join(f"(?:{p})" for p in pats))

    def __call__(self, text: str) -> str:
        return self._re.sub("[REDACTED]", text)

    def scrub(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, dict):
            return {k: self.scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.scrub(v) for v in obj)
        return obj


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()


@lru_cache(maxsize=1)
def get_redactor() -> Redactor:
    try:
        patterns = get_config().get("observability.redact_patterns")
    except Exception:  # config may not exist yet during bootstrap
        patterns = None
    return Redactor(patterns)
