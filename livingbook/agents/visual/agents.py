"""Visual engine agents.

The pipeline is search → verify → licence → refine → generate → QA → place, and the
`AssetManager` is the only agent that may write `manuscript/images/`. That single
choke point is what makes "no unlicensed or irrelevant image reaches the book" an
enforceable property rather than a hope.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ...config import get_config
from ...research.models import VisualRequirement
from ..base import BaseAgent


class VisualNeedDetector(BaseAgent[list[VisualRequirement]]):
    name = "visual_need_detector"
    uses_skills = ("visual_need_detection",)

    def degraded_result(self) -> list[VisualRequirement]:
        return []

    async def execute(
        self, *, text: str = "", node_id: str = "", concept: str = "",
        include_unresolved_placeholders: bool = False, **_: Any,
    ) -> list[VisualRequirement]:
        requirements: list[VisualRequirement] = []

        if text.strip():
            existing = [
                r["key"] for r in self.store.query(
                    "SELECT key FROM figures WHERE node_id = ?", (node_id,))
            ] if node_id else []
            req = await self.skill("visual_need_detection")(
                self.ctx, text=text, concept=concept, node_id=node_id,
                existing_figures=existing)
            if req:
                requirements.append(req)

        # The manuscript's \bookimage macro renders a placeholder box when the image
        # file is absent, and its third argument is already a full specification of
        # what the figure must show. Those are ready-made work items.
        if include_unresolved_placeholders:
            rows = self.store.query(
                "SELECT key, requirement, caption, node_id FROM figures "
                "WHERE status = 'missing' AND requirement != '' LIMIT 20")
            for r in rows:
                requirements.append(VisualRequirement(
                    key=r["key"],
                    concept=r["key"].replace("_", " "),
                    purpose=r["requirement"][:600],
                    expected_elements=_elements_from(r["requirement"]),
                    caption_hint=r["caption"] or "",
                    node_id=r["node_id"] or "",
                ))

        self.log.info(f"visual needs: {len(requirements)} requirement(s)", status="ok")
        return requirements


class ImageSearchAgent(BaseAgent[list[dict[str, Any]]]):
    name = "image_search_agent"
    uses_skills = ("visual_search",)

    def degraded_result(self) -> list[dict[str, Any]]:
        return []

    async def execute(
        self, *, requirement: VisualRequirement, query_override: str = "", **_: Any,
    ) -> list[dict[str, Any]]:
        cfg = get_config()
        candidates = await self.skill("visual_search")(
            self.ctx, requirement=requirement, query_override=query_override,
            max_candidates=int(cfg.get("visual.max_candidates_per_requirement", 8)))

        downloaded: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            if not candidate.get("license_acceptable"):
                continue
            key = requirement.key or "figure"
            ext = _ext_for(candidate.get("mime", "image/png"))
            dest = f"figures/candidates/{key}_{i}{ext}"
            info = await self.ctx.try_call(
                "download_image", default=None, url=candidate["url"], dest=dest)
            if info:
                downloaded.append({**candidate, **info, "local_path": info["path"]})

        self.log.info(
            f"image search '{requirement.concept[:40]}': {len(candidates)} candidates, "
            f"{len(downloaded)} downloaded with acceptable licences", status="ok")
        return downloaded


class SemanticImageVerifier(BaseAgent[dict[str, Any]]):
    name = "semantic_image_verifier"
    uses_skills = ("visual_verification",)

    def degraded_result(self) -> dict[str, Any]:
        return {"verdict": "reject", "reason": "verifier unavailable"}

    async def execute(
        self, *, requirement: VisualRequirement, candidate: dict[str, Any], **_: Any,
    ) -> dict[str, Any]:
        return await self.skill("visual_verification")(
            self.ctx, requirement=requirement, candidate=candidate,
            image_path=candidate.get("local_path", ""))


class LicenseChecker(BaseAgent[dict[str, Any]]):
    """Final licence gate. Unknown licence means rejected."""

    name = "license_checker"
    uses_skills = ("visual_verification",)

    def degraded_result(self) -> dict[str, Any]:
        # fail_pipeline policy: a licence that cannot be checked is not a licence.
        return {"acceptable": False, "reason": "licence check unavailable"}

    async def execute(self, *, candidate: dict[str, Any], **_: Any) -> dict[str, Any]:
        from ...tools.visual import is_license_acceptable, normalise_license

        cfg = get_config()
        allowed = cfg.get("visual.acceptable_licenses", [])
        licence = normalise_license(candidate.get("license"))
        acceptable = is_license_acceptable(licence, allowed)
        attribution = (candidate.get("attribution") or "").strip()
        needs_attribution = (bool(cfg.get("visual.require_attribution", True))
                             and licence not in ("cc0", "pdm", "pd", "publicdomain"))

        problems: list[str] = []
        if not acceptable:
            problems.append(f"licence {licence!r} is not redistributable in a book")
        if needs_attribution and not attribution:
            problems.append("licence requires attribution but no author is recorded")
        if licence in ("unknown", ""):
            problems.append("no licence information could be established")

        return {
            "acceptable": acceptable and not problems,
            "license": licence,
            "license_url": candidate.get("license_url", ""),
            "attribution": attribution,
            "source_url": candidate.get("page_url") or candidate.get("url", ""),
            "problems": problems,
        }


class DiagramGeneratorAgent(BaseAgent[dict[str, Any]]):
    name = "diagram_generator_agent"
    uses_skills = ("figure_generation",)

    def degraded_result(self) -> dict[str, Any]:
        return {"generated": False, "reason": "generator unavailable"}

    async def execute(self, *, requirement: VisualRequirement, **_: Any) -> dict[str, Any]:
        result = await self.skill("figure_generation")(
            self.ctx, requirement=requirement)
        if result.get("generated"):
            self.log.info(
                f"generated original diagram for '{requirement.concept[:40]}' "
                f"via {result.get('backend')}", status="ok")
        return result


class FigureQAAgent(BaseAgent[dict[str, Any]]):
    name = "figure_qa_agent"
    uses_skills = ("figure_qa",)

    def degraded_result(self) -> dict[str, Any]:
        return {"passed": False, "problems": ["figure QA unavailable"]}

    async def execute(
        self, *, image_path: str, caption: str = "", alt_text: str = "",
        requirement: VisualRequirement | None = None, **_: Any,
    ) -> dict[str, Any]:
        return await self.skill("figure_qa")(
            self.ctx, image_path=image_path, caption=caption, alt_text=alt_text,
            requirement=requirement)


class AssetManager(BaseAgent[dict[str, Any]]):
    """Places an approved figure into the manuscript and records its provenance.

    Deterministic by design — it has no LLM grant. Everything it needs has already been
    decided by the agents upstream; its job is to move the file and write the ledger.
    """

    name = "asset_manager"
    uses_skills = ()

    def degraded_result(self) -> dict[str, Any]:
        return {"placed": False}

    async def execute(
        self, *, requirement: VisualRequirement, source_path: str,
        license_info: dict[str, Any], qa: dict[str, Any] | None = None,
        caption: str = "", **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        key = requirement.key or _slug(requirement.concept)

        src = Path(source_path)
        src_abs = (cfg.root / src) if not src.is_absolute() else src
        if not src_abs.exists():
            return {"placed": False, "reason": f"source image missing: {source_path}"}

        ext = src_abs.suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg"):
            ext = ".png"
        dest_rel = f"manuscript/images/{key}{ext}"

        data = src_abs.read_bytes()
        dest_abs = self.ctx.check_write_path(dest_rel)
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        dest_abs.write_bytes(data)

        # Keep an approved copy outside the manuscript so a later revert can restore
        # it without re-running search and verification.
        approved_abs = self.ctx.check_write_path(f"figures/approved/{key}{ext}")
        approved_abs.parent.mkdir(parents=True, exist_ok=True)
        approved_abs.write_bytes(data)

        sha = hashlib.sha256(data).hexdigest()
        alt_text = (qa or {}).get("alt_text") or requirement.purpose[:300]

        await self.ctx.call("kb_upsert", kind="figure", records=[{
            "key": key,
            "node_id": requirement.node_id or None,
            "path": f"images/{key}{ext}",
            "requirement": requirement.purpose,
            "caption": caption or requirement.caption_hint,
            "alt_text": alt_text,
            "source_url": license_info.get("source_url", ""),
            "license": license_info.get("license", ""),
            "license_url": license_info.get("license_url", ""),
            "attribution": license_info.get("attribution", ""),
            "status": "placed",
            "sha256": sha,
        }])

        self._update_sources_md(key, ext, license_info, requirement)
        self.log.info(f"placed figure images/{key}{ext} "
                      f"({license_info.get('license','?')})", status="ok")
        return {
            "placed": True, "key": key, "path": f"images/{key}{ext}",
            "manuscript_path": dest_rel, "sha256": sha, "alt_text": alt_text,
            "license": license_info.get("license", ""),
            "attribution": license_info.get("attribution", ""),
        }

    def _update_sources_md(
        self, key: str, ext: str, license_info: dict[str, Any],
        requirement: VisualRequirement,
    ) -> None:
        """Append to the manuscript's existing image provenance ledger.

        `images/SOURCES.md` is the author's own convention — a table of filename,
        source, paper and licence, with an explicit note about commercial
        republication. Maintaining it rather than starting a parallel record keeps one
        source of truth for figure rights.
        """
        cfg = get_config()
        path = cfg.images_dir / "SOURCES.md"
        rel = path.relative_to(cfg.root).as_posix()
        try:
            self.ctx.check_write_path(rel)
        except Exception:
            return

        row = (f"| `{key}{ext}` | {license_info.get('source_url','—')} | "
               f"{license_info.get('attribution','—')} | "
               f"**{license_info.get('license','unknown')}** |")

        if not path.exists():
            header = (
                "# Nguồn các hình minh họa\n\n"
                "| Tệp | Nguồn | Tác giả / Ghi công | Giấy phép |\n|---|---|---|---|\n")
            path.write_text(header + row + "\n", encoding="utf-8")
            return

        text = path.read_text(encoding="utf-8", errors="replace")
        if f"`{key}{ext}`" in text:
            return  # already recorded

        marker = "\n## Hình còn thiếu"
        block = ("\n<!-- added by the Living Book visual engine -->\n"
                 "| Tệp | Nguồn | Tác giả / Ghi công | Giấy phép |\n|---|---|---|---|\n")
        if "<!-- added by the Living Book visual engine -->" in text:
            # Append into the existing agent-managed table rather than starting another.
            idx = text.rindex("<!-- added by the Living Book visual engine -->")
            end = text.find("\n\n", idx)
            end = end if end != -1 else len(text)
            text = text[:end] + "\n" + row + text[end:]
        elif marker in text:
            text = text.replace(marker, block + row + "\n" + marker, 1)
        else:
            text = text.rstrip("\n") + "\n" + block + row + "\n"
        path.write_text(text, encoding="utf-8")


# -- helpers ---------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:48] or "figure"


def _ext_for(mime: str) -> str:
    return {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
            "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".png")


def _elements_from(requirement: str) -> list[str]:
    """Pull candidate elements out of a Vietnamese \\bookimage description.

    Crude clause splitting, but it gives the search and verification steps concrete
    things to look for rather than one long sentence.
    """
    parts = re.split(r"[;,]|\bvà\b|\bhoặc\b", requirement)
    return [p.strip()[:120] for p in parts if len(p.strip()) > 12][:8]
