"""The visual pipeline, end to end.

    need -> search -> semantic verify -> licence -> refine & search again
         -> generate an original diagram -> figure QA -> place

The refinement loop is the part that matters: when nothing acceptable comes back, the
semantic verifier says *why*, that becomes a better search description, and the search
runs again. Only when that is exhausted does the system draw its own diagram — which
is always licensable and always says exactly what it should, at the cost of being
plainer than a figure from a paper.
"""

from __future__ import annotations

from typing import Any

from ..agents import (
    AssetManager,
    DiagramGeneratorAgent,
    FigureQAAgent,
    ImageSearchAgent,
    LicenseChecker,
    SemanticImageVerifier,
)
from ..config import get_config
from ..obs import get_logger
from ..research.models import VisualRequirement


async def run_visual_pipeline(
    requirements: list[VisualRequirement], *, pipeline_id: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve each requirement to a placed figure, or give up cleanly."""
    log = get_logger()
    cfg = get_config()
    max_rounds = int(cfg.get("visual.max_refine_rounds", 2))

    searcher = ImageSearchAgent(pipeline_id=pipeline_id)
    verifier = SemanticImageVerifier(pipeline_id=pipeline_id)
    licensor = LicenseChecker(pipeline_id=pipeline_id)
    generator = DiagramGeneratorAgent(pipeline_id=pipeline_id)
    qa = FigureQAAgent(pipeline_id=pipeline_id)
    assets = AssetManager(pipeline_id=pipeline_id)

    placed: list[dict[str, Any]] = []

    for requirement in requirements:
        log.info(f"visual: resolving '{requirement.concept[:50]}'")
        result = await _resolve_one(
            requirement, searcher, verifier, licensor, generator, qa, assets,
            max_rounds=max_rounds, log=log)
        if result:
            placed.append(result)
        else:
            log.warn(f"visual: no acceptable figure for '{requirement.concept[:50]}'; "
                     "the \\bookimage placeholder will remain")
    return placed


async def _resolve_one(
    requirement: VisualRequirement,
    searcher: ImageSearchAgent,
    verifier: SemanticImageVerifier,
    licensor: LicenseChecker,
    generator: DiagramGeneratorAgent,
    qa: FigureQAAgent,
    assets: AssetManager,
    *,
    max_rounds: int,
    log: Any,
) -> dict[str, Any] | None:
    query_override = ""

    for round_index in range(max_rounds + 1):
        candidates = await searcher.run(requirement=requirement,
                                        query_override=query_override)
        refinement = ""

        for candidate in candidates:
            verdict = await verifier.run(requirement=requirement, candidate=candidate)

            if verdict.get("verdict") == "refine_search":
                refinement = refinement or verdict.get("refined_description", "")
                continue
            if verdict.get("verdict") != "accept":
                continue

            licence = await licensor.run(candidate=candidate)
            if not licence.get("acceptable"):
                log.info(f"visual: rejected on licence — {licence.get('problems')}")
                continue

            checked = await qa.run(
                image_path=candidate["local_path"],
                caption=requirement.caption_hint, requirement=requirement)
            if not checked.get("passed"):
                log.info(f"visual: failed figure QA — {checked.get('problems')}")
                continue

            result = await assets.run(
                requirement=requirement, source_path=candidate["local_path"],
                license_info=licence, qa=checked, caption=requirement.caption_hint)
            if result.get("placed"):
                return {**result, "action": "found", "round": round_index,
                        "source_url": licence.get("source_url", "")}

        if round_index < max_rounds and refinement:
            log.info(f"visual: refining search — {refinement[:120]}")
            query_override = refinement
            continue
        break

    # Nothing acceptable exists: draw it. An original diagram is unambiguously
    # licensable and shows exactly the required elements, which a found image
    # frequently does not.
    generated = await generator.run(requirement=requirement)
    if not generated.get("generated"):
        return None

    checked = await qa.run(
        image_path=generated["path"], caption=requirement.caption_hint,
        alt_text=generated.get("alt_text", ""), requirement=requirement)
    if not checked.get("passed"):
        log.warn(f"visual: generated diagram failed QA — {checked.get('problems')}")
        return None

    licence_info = {
        "license": "cc0",
        "attribution": "Hình gốc do hệ thống Living Book tạo",
        "source_url": "",
        "license_url": "",
    }
    result = await assets.run(
        requirement=requirement, source_path=generated["path"],
        license_info=licence_info, qa=checked, caption=requirement.caption_hint)
    if result.get("placed"):
        return {**result, "action": "generated",
                "backend": generated.get("backend", "matplotlib")}
    return None
