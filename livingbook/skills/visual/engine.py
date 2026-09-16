"""Visual engine skills: need detection, licensed search, semantic verification,
refinement, original diagram generation and figure QA.

Two hard gates, both non-negotiable:

  * **Relevance.** An image must demonstrably show the required elements and
    relationships. A keyword or filename match is not a match, and the check is
    multimodal — the model looks at the image, not at its metadata.
  * **Licence.** An image with no provable, redistributable licence is rejected. The
    manuscript's own `images/SOURCES.md` already records source and licence per figure
    and even flags that arXiv figures would need permission for commercial
    publication; this pipeline maintains that ledger rather than inventing a new one.

When nothing acceptable exists, the fallback is an original diagram rendered from a
declarative spec — deterministic, correctly labelled, and unambiguously licensable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...config import get_config
from ...research.models import VisualRequirement
from ...tools import AgentContext
from ...tools.visual import is_license_acceptable, normalise_license
from ..base import Skill, array_of, boolean, integer, json_schema, number, obj, string

NEED_SCHEMA = json_schema(
    {
        "needed": boolean("Would a figure genuinely help here?"),
        "reason": string(),
        "key": string("snake_case figure key"),
        "concept": string(),
        "purpose": string("What the reader should take from it"),
        "expected_elements": array_of(string(), "Concrete things that must appear"),
        "relationships": array_of(string(), "How those elements relate"),
        "figure_kind": string("", ["architecture_diagram", "flow_diagram", "plot",
                                   "comparison_table", "screenshot", "conceptual"]),
        "caption_hint": string("Vietnamese caption"),
    },
    ["needed", "reason"],
)

VERIFY_SCHEMA = json_schema(
    {
        "depicts_concept": boolean("Does the image actually show this concept?"),
        "elements_present": array_of(string(), "Required elements you can SEE"),
        "elements_missing": array_of(string(), "Required elements that are absent"),
        "relationships_shown": boolean(),
        "legibility": string("", ["excellent", "good", "poor", "unreadable"]),
        "text_language": string("Language of any text in the image"),
        "is_technical_diagram": boolean(),
        "match_quality": string("", ["exact", "close", "related", "unrelated"]),
        "problems": array_of(string()),
        "verdict": string("", ["accept", "reject", "refine_search"]),
        "refined_description": string("A better search description, if refine_search"),
    },
    ["depicts_concept", "match_quality", "verdict"],
)

DIAGRAM_SCHEMA = json_schema(
    {
        "title": string("Vietnamese title for the diagram"),
        "layout": string("", ["flow", "grid", "manual"]),
        "nodes": array_of(obj({
            "id": string("short ascii id"),
            "label": string("Vietnamese label, short"),
            "row": integer("for grid layout"),
            "col": integer("for grid layout"),
            "color": string("hex colour, optional"),
        }, ["id", "label"])),
        "edges": array_of(obj({
            "from": string(), "to": string(), "label": string(),
            "style": string("", ["solid", "dashed", "bidirectional"]),
        }, ["from", "to"])),
        "notes": array_of(obj({"text": string(), "x": number(), "y": number()},
                              ["text"])),
        "alt_text": string("Vietnamese alt text describing the diagram"),
    },
    ["title", "layout", "nodes", "edges"],
)

FIGURE_QA_SCHEMA = json_schema(
    {
        "passed": boolean(),
        "legible_at_print_size": boolean(),
        "caption_matches_image": boolean(),
        "alt_text_adequate": boolean(),
        "problems": array_of(string()),
        "suggested_alt_text": string("Vietnamese"),
    },
    ["passed", "problems"],
)


class VisualNeedDetectionSkill(Skill):
    name = "visual_need_detection"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("kb_query", "kb_retrieve_context", "read_file")

    async def run(
        self, ctx: AgentContext, *, text: str, concept: str = "", node_id: str = "",
        existing_figures: list[str] | None = None, **_: Any,
    ) -> VisualRequirement | None:
        prompt = (
            "Decide whether this textbook passage genuinely needs a figure.\n\n"
            "Say yes only when a figure does work prose cannot: showing an "
            "architecture, a data flow, a quantitative relationship, or a structural "
            "comparison. Say no for a passage that is already clear, for decoration, "
            "or where a figure would just restate a list.\n\n"
            "If yes, specify exactly what it must show — the elements and how they "
            "relate. That specification is what the search and generation steps use, "
            "so vagueness there produces a wrong figure.\n\n"
            f"CONCEPT: {concept}\n"
            + (f"FIGURES ALREADY IN THIS AREA: {', '.join(existing_figures)}\n"
               if existing_figures else "")
            + f"\nPASSAGE (Vietnamese):\n{text[:14000]}"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=NEED_SCHEMA, temperature=0.2, role="fast")
        d = result["data"]
        if not d.get("needed"):
            return None
        return VisualRequirement(
            key=_slug(d.get("key") or d.get("concept") or concept),
            concept=d.get("concept") or concept,
            purpose=d.get("purpose", ""),
            expected_elements=(d.get("expected_elements") or [])[:10],
            relationships=(d.get("relationships") or [])[:10],
            style=d.get("figure_kind", "architecture_diagram"),
            caption_hint=d.get("caption_hint", ""),
            node_id=node_id,
        )


class VisualSearchSkill(Skill):
    """Search licence-bearing sources first, then the open web as a last resort."""

    name = "visual_search"
    required_tools = ()
    optional_tools = ("search_wikimedia_images", "search_openverse_images",
                      "web_search", "download_image", "gemini_structured_output")

    async def run(
        self, ctx: AgentContext, *, requirement: VisualRequirement,
        max_candidates: int = 8, query_override: str = "", **_: Any,
    ) -> list[dict[str, Any]]:
        cfg = get_config()
        allowed = cfg.get("visual.acceptable_licenses", [])
        min_width = int(cfg.get("visual.min_image_width", 700))

        queries = [query_override] if query_override else _queries_for(requirement)
        candidates: list[dict[str, Any]] = []

        # Wikimedia and Openverse first: both return licence metadata, so a candidate
        # arrives already carrying what the licence gate needs.
        for tool_name in ("search_wikimedia_images", "search_openverse_images"):
            if tool_name not in ctx.allowed_tools:
                continue
            for query in queries[:3]:
                found = await ctx.try_call(tool_name, default=[], query=query, limit=6)
                candidates.extend(found or [])

        filtered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in candidates:
            url = c.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            if int(c.get("width") or 0) and int(c["width"]) < min_width:
                continue
            c["license_acceptable"] = is_license_acceptable(c.get("license"), allowed)
            filtered.append(c)

        filtered.sort(key=lambda c: (not c["license_acceptable"], -(c.get("width") or 0)))
        return filtered[:max_candidates]


class VisualVerificationSkill(Skill):
    """Look at the image and decide whether it shows what is required."""

    name = "visual_verification"
    required_tools = ()
    optional_tools = ("inspect_image", "gemini_structured_output", "fetch_url")

    async def run(
        self, ctx: AgentContext, *, requirement: VisualRequirement,
        image_path: str = "", image_bytes: bytes | None = None,
        candidate: dict[str, Any] | None = None, **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        candidate = candidate or {}

        licence = normalise_license(candidate.get("license"))
        licence_ok = is_license_acceptable(
            licence, cfg.get("visual.acceptable_licenses", []))
        attribution = (candidate.get("attribution") or "").strip()
        needs_attribution = bool(cfg.get("visual.require_attribution", True))

        # Licence is checked before relevance: verifying an unusable image costs a
        # multimodal call for nothing.
        if not licence_ok:
            return {
                "verdict": "reject", "reason": "license",
                "license": licence, "license_acceptable": False,
                "note": f"licence {licence!r} is not in the acceptable list",
            }
        if needs_attribution and not attribution and licence not in ("cc0", "pdm", "pd"):
            return {
                "verdict": "reject", "reason": "attribution",
                "license": licence, "license_acceptable": True,
                "note": "licence requires attribution but no author is recorded",
            }

        if image_bytes is None and image_path and "inspect_image" in ctx.allowed_tools:
            info = await ctx.try_call("inspect_image", default=None,
                                      path=image_path, with_bytes=True)
            if info:
                image_bytes = info.get("data")

        if not image_bytes:
            return {"verdict": "reject", "reason": "unreadable",
                    "note": "image could not be read for verification"}

        from ...llm import get_provider
        prompt = (
            "Judge whether this image is suitable as a figure in a technical textbook "
            "section. Look at the IMAGE, not at any filename or caption.\n\n"
            f"CONCEPT: {requirement.concept}\n"
            f"PURPOSE: {requirement.purpose}\n"
            f"MUST SHOW: {', '.join(requirement.expected_elements) or '(unspecified)'}\n"
            f"RELATIONSHIPS: {', '.join(requirement.relationships) or '(unspecified)'}\n\n"
            "Reject an image that is merely topically related, is a photograph where a "
            "diagram is needed, is illegible at print size, or is missing the required "
            "elements. Being about the right subject is not enough.\n\n"
            "Return refine_search with a better description if the image suggests what "
            "a correct search would look like."
        )
        try:
            resp = await get_provider().analyse_image_structured(
                image_bytes, prompt, VERIFY_SCHEMA,
                mime_type=candidate.get("mime", "image/png"))
            d = resp.data
        except Exception as exc:
            return {"verdict": "reject", "reason": "verification_failed",
                    "note": f"{type(exc).__name__}: {exc}"}

        verdict = d.get("verdict", "reject")
        if d.get("match_quality") in ("unrelated",):
            verdict = "reject"
        if d.get("legibility") in ("poor", "unreadable"):
            verdict = "reject"
        if d.get("elements_missing") and len(d["elements_missing"]) > max(
                1, len(requirement.expected_elements) // 2):
            verdict = "refine_search" if verdict == "accept" else verdict

        return {
            "verdict": verdict,
            "reason": "semantic",
            "license": licence,
            "license_acceptable": True,
            "attribution": attribution,
            "match_quality": d.get("match_quality"),
            "elements_present": d.get("elements_present", []),
            "elements_missing": d.get("elements_missing", []),
            "legibility": d.get("legibility"),
            "problems": d.get("problems", []),
            "refined_description": d.get("refined_description", ""),
        }


class FigureGenerationSkill(Skill):
    """Produce an original diagram when no acceptable image exists."""

    name = "figure_generation"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("render_diagram", "generate_image", "optimize_image")

    async def run(
        self, ctx: AgentContext, *, requirement: VisualRequirement,
        dest_dir: str = "figures/generated", **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        if not cfg.get("visual.allow_generation", True):
            return {"generated": False, "reason": "generation disabled in config"}

        backend = cfg.get("visual.generation_backend", "matplotlib")
        key = requirement.key or _slug(requirement.concept)

        if backend == "matplotlib" and "render_diagram" in ctx.allowed_tools:
            spec_result = await ctx.call(
                "gemini_structured_output",
                prompt=(
                    "Design a box-and-arrow diagram for a Vietnamese NLP textbook.\n\n"
                    f"CONCEPT: {requirement.concept}\n"
                    f"PURPOSE: {requirement.purpose}\n"
                    f"MUST SHOW: {', '.join(requirement.expected_elements)}\n"
                    f"RELATIONSHIPS: {', '.join(requirement.relationships)}\n\n"
                    "Labels must be in VIETNAMESE and short (2-5 words) — long labels "
                    "overflow their boxes. Use at most 9 nodes; a diagram denser than "
                    "that is unreadable at print size. Node ids are short ASCII and "
                    "must be unique. Every edge must reference existing node ids.\n"
                    "Use layout 'flow' for a pipeline, 'grid' when you set row/col."
                ),
                schema=DIAGRAM_SCHEMA, temperature=0.3,
            )
            spec = spec_result["data"]
            spec = _sanitise_spec(spec)
            if not spec.get("nodes"):
                return {"generated": False, "reason": "diagram spec had no valid nodes"}

            dest = f"{dest_dir.rstrip('/')}/{key}.png"
            rendered = await ctx.call("render_diagram", spec=spec, dest=dest)
            return {
                "generated": True, "backend": "matplotlib", "key": key,
                "alt_text": spec.get("alt_text", ""), "spec": spec, **rendered,
            }

        if "generate_image" in ctx.allowed_tools:
            prompt = (
                f"A clean, minimal technical diagram for a textbook: "
                f"{requirement.concept}. {requirement.purpose}. "
                f"Show: {', '.join(requirement.expected_elements)}. "
                f"Relationships: {', '.join(requirement.relationships)}. "
                "Flat vector style, white background, clearly labelled boxes and arrows, "
                "no photographic elements, no decorative clutter."
            )
            dest = f"{dest_dir.rstrip('/')}/{key}.png"
            out = await ctx.call("generate_image", prompt=prompt, dest=dest)
            return {"generated": True, "backend": "gemini_image", "key": key, **out}

        return {"generated": False, "reason": "no generation backend available"}


class FigureQASkill(Skill):
    name = "figure_qa"
    required_tools = ()
    optional_tools = ("inspect_image", "gemini_structured_output")

    async def run(
        self, ctx: AgentContext, *, image_path: str, caption: str = "",
        alt_text: str = "", requirement: VisualRequirement | None = None, **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        info = await ctx.try_call("inspect_image", default=None,
                                  path=image_path, with_bytes=True)
        if not info:
            return {"passed": False, "problems": ["image could not be read"]}

        min_width = int(cfg.get("visual.min_image_width", 700))
        problems: list[str] = []
        if info.get("width", 0) < min_width:
            problems.append(
                f"image is {info.get('width')}px wide; at least {min_width}px is "
                "needed to stay legible in print")

        data = info.get("data")
        if not data:
            return {"passed": not problems, "problems": problems, **_meta(info)}

        from ...llm import get_provider
        prompt = (
            "Final check on a figure about to be placed in a printed textbook.\n\n"
            f"CAPTION (Vietnamese): {caption or '(none)'}\n"
            f"ALT TEXT: {alt_text or '(none)'}\n"
            + (f"IT IS MEANT TO SHOW: {requirement.purpose}; "
               f"elements: {', '.join(requirement.expected_elements)}\n"
               if requirement else "")
            + "\nCheck: is the text inside the image legible when printed at about "
              "12cm wide? Does the caption describe what is actually shown? Is the alt "
              "text adequate for a reader who cannot see the image? Is any text in the "
              "image garbled or misspelled?\n"
              "Suggest Vietnamese alt text if the current one is missing or weak."
        )
        try:
            resp = await get_provider().analyse_image_structured(
                data, prompt, FIGURE_QA_SCHEMA,
                mime_type=f"image/{info.get('format', 'png')}")
            d = resp.data
        except Exception as exc:
            return {"passed": False,
                    "problems": problems + [f"figure QA failed: {exc}"], **_meta(info)}

        problems.extend(d.get("problems", []) or [])
        passed = bool(d.get("passed")) and not problems
        return {
            "passed": passed,
            "problems": problems,
            "legible": d.get("legible_at_print_size"),
            "caption_matches": d.get("caption_matches_image"),
            "alt_text": alt_text or d.get("suggested_alt_text", ""),
            "suggested_alt_text": d.get("suggested_alt_text", ""),
            **_meta(info),
        }


# -- helpers ---------------------------------------------------------------


def _meta(info: dict[str, Any]) -> dict[str, Any]:
    return {k: info[k] for k in ("width", "height", "format", "bytes", "sha256")
            if k in info}


def _queries_for(req: VisualRequirement) -> list[str]:
    base = req.concept.strip()
    elements = " ".join(req.expected_elements[:3])
    return [
        f"{base} diagram",
        f"{base} architecture {elements}".strip(),
        base,
    ]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:48] or "figure"


def _sanitise_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop edges pointing at non-existent nodes and de-duplicate node ids.

    A generated spec referencing a node it never declared would raise inside the
    renderer; dropping the bad edge yields a slightly poorer diagram instead of a
    failed pipeline step.
    """
    nodes = []
    seen: set[str] = set()
    for n in spec.get("nodes") or []:
        nid = str(n.get("id") or "").strip()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        nodes.append({k: v for k, v in n.items() if v not in (None, "")})
    edges = [
        e for e in (spec.get("edges") or [])
        if str(e.get("from")) in seen and str(e.get("to")) in seen
    ]
    spec["nodes"] = nodes[:9]
    kept = {n["id"] for n in spec["nodes"]}
    spec["edges"] = [e for e in edges
                     if e["from"] in kept and e["to"] in kept][:20]
    return spec
