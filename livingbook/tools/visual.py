"""Visual tools: licensed image search, inspection, and original diagram generation.

Source order is deliberate. Wikimedia Commons and Openverse both expose
machine-readable licence metadata, so an image from either arrives with the
information the licence checker needs. A general web image is a last resort and
carries `license: unknown`, which the licence checker rejects by default — an image
with no provable licence cannot go into a book that might be published.

``render_diagram`` generalises the hand-coded matplotlib script from `intro-to-cv`
into a declarative spec, so a diagram is described (boxes, arrows, labels) rather than
drawn by hand-tuned coordinates.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

from ..config import get_config
from ..obs import get_logger
from .http import get_bytes, get_json
from .registry import AgentContext, Capability, ToolError, ToolUnavailable, tool

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1"

#: Licences acceptable for a redistributable textbook, normalised to lower case.
_OPEN_LICENSES = {
    "cc0", "pdm", "pd", "publicdomain", "public domain",
    "cc-by", "cc-by-sa", "cc by", "cc by-sa", "by", "by-sa",
    "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-2.0", "cc-by-sa-2.5", "cc-by-sa-3.0", "cc-by-sa-4.0",
}


def normalise_license(raw: str | None) -> str:
    if not raw:
        return "unknown"
    text = str(raw).strip().lower().replace("_", "-").replace(" ", "-")
    text = re.sub(r"^(creative-commons-|cc-?)", "cc-", text)
    text = text.replace("attribution-sharealike", "by-sa").replace("attribution", "by")
    return text or "unknown"


def is_license_acceptable(raw: str | None, allowed: list[str] | None = None) -> bool:
    norm = normalise_license(raw)
    pool = {normalise_license(a) for a in (allowed or [])} or _OPEN_LICENSES
    if norm in pool:
        return True
    # cc-by-4.0 should satisfy an allowlist that only names cc-by, and vice versa.
    base = re.sub(r"-\d(\.\d)?$", "", norm)
    return base in pool


@tool("search_wikimedia_images", [Capability.SEARCH],
      description="Search Wikimedia Commons; results carry explicit licence metadata.")
async def search_wikimedia_images(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    data = await get_json(
        WIKIMEDIA_API,
        params={
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrnamespace": 6, "gsrlimit": min(limit, 30),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|user",
            "format": "json",
        },
        timeout=45,
    )
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    out: list[dict[str, Any]] = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata", {}) or {}

        def field(name: str) -> str:
            raw = (meta.get(name) or {}).get("value", "")
            return re.sub(r"<[^>]+>", "", str(raw)).strip()

        mime = info.get("mime", "")
        if not mime.startswith("image/") or "svg" in mime:
            continue  # SVG needs conversion; skip rather than half-support it

        out.append({
            "source": "wikimedia",
            "source_id": page.get("title", ""),
            "title": page.get("title", "").replace("File:", ""),
            "url": info.get("url", ""),
            "page_url": info.get("descriptionurl", ""),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "mime": mime,
            "license": normalise_license(field("LicenseShortName") or field("License")),
            "license_url": field("LicenseUrl"),
            "attribution": field("Artist") or info.get("user", ""),
            "description": field("ImageDescription")[:1200],
            "credit": field("Credit")[:400],
        })
    return out


@tool("search_openverse_images", [Capability.SEARCH],
      description="Search Openverse for openly licensed images.")
async def search_openverse_images(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    data = await get_json(
        f"{OPENVERSE_API}/images/",
        params={"q": query, "page_size": min(limit, 20), "mature": "false"},
        timeout=45,
    )
    out: list[dict[str, Any]] = []
    for r in data.get("results", []) or []:
        out.append({
            "source": "openverse",
            "source_id": r.get("id", ""),
            "title": r.get("title", "") or "",
            "url": r.get("url", ""),
            "page_url": r.get("foreign_landing_url", ""),
            "width": r.get("width", 0),
            "height": r.get("height", 0),
            "mime": f"image/{(r.get('filetype') or 'jpeg')}",
            "license": normalise_license(
                f"cc-{r.get('license')}" if r.get("license") not in ("cc0", "pdm")
                else r.get("license")),
            "license_url": r.get("license_url", ""),
            "attribution": r.get("creator", "") or "",
            "description": (r.get("attribution") or "")[:1200],
            "credit": r.get("source", ""),
        })
    return out


@tool("download_image", [Capability.FETCH, Capability.FS_WRITE],
      description="Download a candidate image into the agent's write scope.")
async def download_image(
    ctx: AgentContext, url: str, *, dest: str, max_mb: int = 25,
) -> dict[str, Any]:
    target = ctx.check_write_path(dest)
    data = await get_bytes(url, max_bytes=max_mb * 1024 * 1024, timeout=90)
    if not _looks_like_image(data):
        raise ToolError(f"{url} did not return a recognisable image")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    info = _probe_image(data)
    return {
        "path": target.relative_to(get_config().root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **info,
    }


@tool("inspect_image", [Capability.FS_READ],
      description="Read an image's dimensions, format and bytes for verification.")
async def inspect_image(path: str, *, with_bytes: bool = False) -> dict[str, Any]:
    cfg = get_config()
    p = Path(path)
    target = (cfg.root / p) if not p.is_absolute() else p
    if not target.exists():
        raise ToolError(f"no such image: {path}")
    data = target.read_bytes()
    info = _probe_image(data)
    out = {
        "path": p.as_posix(), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), **info,
    }
    if with_bytes:
        out["data"] = data
    return out


@tool("optimize_image", [Capability.FS_WRITE],
      description="Normalise an image for print: resize, convert, strip metadata.")
async def optimize_image(
    ctx: AgentContext, path: str, *, dest: str | None = None,
    max_width: int = 1800, target_format: str = "PNG",
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolUnavailable("Pillow is not installed; cannot optimise images") from exc

    cfg = get_config()
    src = Path(path)
    src_abs = (cfg.root / src) if not src.is_absolute() else src
    if not src_abs.exists():
        raise ToolError(f"no such image: {path}")
    out_path = ctx.check_write_path(dest or path)

    with Image.open(src_abs) as img:
        img = img.convert("RGB") if img.mode in ("P", "RGBA", "LA") and \
            target_format == "JPEG" else img.convert("RGBA") if img.mode == "P" else img
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=target_format, optimize=True)
        data = buf.getvalue()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {
        "path": out_path.relative_to(cfg.root).as_posix(),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        **_probe_image(data),
    }


@tool("render_diagram", [Capability.FS_WRITE],
      description="Render an original diagram from a declarative spec via matplotlib.")
async def render_diagram(
    ctx: AgentContext, spec: dict[str, Any], *, dest: str, dpi: int = 200,
) -> dict[str, Any]:
    """Draw a box-and-arrow diagram from a spec.

    Generalises the approach from `intro-to-cv/gen_images.py` (FancyBboxPatch nodes,
    annotate arrows) into something an agent can drive: it describes what the diagram
    must show and the layout is computed, rather than hand-tuning coordinates.

    Spec shape::

        {
          "title": "...",
          "layout": "flow" | "grid" | "manual",
          "nodes":  [{"id","label","row"?,"col"?,"x"?,"y"?,"w"?,"h"?,"color"?,"shape"?}],
          "edges":  [{"from","to","label"?,"style"?}],
          "notes":  [{"text","x","y"}]
        }
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        raise ToolUnavailable("matplotlib is not installed") from exc

    target = ctx.check_write_path(dest)
    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    if not nodes:
        raise ToolError("diagram spec has no nodes")

    palette = ["#2c3e50", "#2980b9", "#8e44ad", "#e67e22", "#27ae60",
               "#c0392b", "#16a085", "#7f8c8d"]
    layout = spec.get("layout", "flow")
    placed = _layout_nodes(nodes, layout)

    max_x = max(p["x"] + p["w"] for p in placed.values()) + 0.6
    max_y = max(p["y"] + p["h"] for p in placed.values()) + 1.2

    fig, ax = plt.subplots(figsize=(max(7.0, max_x * 1.05), max(3.2, max_y * 0.95)))
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis("off")

    for i, node in enumerate(nodes):
        p = placed[node["id"]]
        colour = node.get("color") or palette[i % len(palette)]
        rounding = 0.16 if node.get("shape") != "sharp" else 0.01
        ax.add_patch(FancyBboxPatch(
            (p["x"], p["y"]), p["w"], p["h"],
            boxstyle=f"round,pad=0.06,rounding_size={rounding}",
            facecolor=colour, alpha=0.9, edgecolor="white", linewidth=1.6,
        ))
        ax.text(
            p["x"] + p["w"] / 2, p["y"] + p["h"] / 2,
            _wrap(str(node.get("label", node["id"])), int(p["w"] * 9)),
            ha="center", va="center", fontsize=9, color="white",
            fontweight="bold", linespacing=1.35,
        )

    for edge in edges:
        a, b = placed.get(edge.get("from")), placed.get(edge.get("to"))
        if not a or not b:
            continue
        (x1, y1), (x2, y2) = _edge_points(a, b)
        style = "->" if edge.get("style") != "bidirectional" else "<->"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color="#444", linewidth=1.7,
                            linestyle="--" if edge.get("style") == "dashed" else "-",
                            connectionstyle="arc3,rad=0.0"),
        )
        if edge.get("label"):
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, str(edge["label"]),
                    ha="center", va="bottom", fontsize=7.5, color="#333",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    for note in spec.get("notes") or []:
        ax.text(float(note.get("x", max_x / 2)), float(note.get("y", 0.3)),
                str(note.get("text", "")), ha="center", fontsize=8, color="#555")

    if spec.get("title"):
        ax.text(max_x / 2, max_y - 0.35, str(spec["title"]),
                ha="center", fontsize=12.5, fontweight="bold")

    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    data = target.read_bytes()
    get_logger().info(f"rendered diagram -> {target.name}", tool="render_diagram")
    return {
        "path": target.relative_to(get_config().root).as_posix(),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "generated": True, "license": "cc0", "attribution": "Original diagram",
        **_probe_image(data),
    }


def _layout_nodes(nodes: list[dict[str, Any]], layout: str) -> dict[str, dict[str, float]]:
    placed: dict[str, dict[str, float]] = {}
    if layout == "manual" and all("x" in n and "y" in n for n in nodes):
        for n in nodes:
            placed[n["id"]] = {"x": float(n["x"]), "y": float(n["y"]),
                               "w": float(n.get("w", 2.0)), "h": float(n.get("h", 1.0))}
        return placed

    if layout == "grid":
        rows = max(int(n.get("row", 0)) for n in nodes) + 1
        for n in nodes:
            row = int(n.get("row", 0))
            col = int(n.get("col", 0))
            w = float(n.get("w", 2.2))
            h = float(n.get("h", 1.1))
            placed[n["id"]] = {"x": 0.4 + col * (w + 0.6),
                               "y": 0.6 + (rows - 1 - row) * (h + 0.7), "w": w, "h": h}
        return placed

    # "flow": left-to-right, wrapping to a new row every 5 nodes so a long pipeline
    # stays readable instead of becoming an unrenderably wide strip.
    per_row = 5
    for i, n in enumerate(nodes):
        row, col = divmod(i, per_row)
        w = float(n.get("w", 2.2))
        h = float(n.get("h", 1.1))
        placed[n["id"]] = {"x": 0.4 + col * (w + 0.7),
                           "y": 0.6 + (2 - row) * (h + 0.9), "w": w, "h": h}
    return placed


def _edge_points(a: dict[str, float], b: dict[str, float]) -> tuple[tuple[float, float],
                                                                    tuple[float, float]]:
    ax, ay = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bx, by = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    if abs(bx - ax) >= abs(by - ay):
        start = (a["x"] + a["w"], ay) if bx > ax else (a["x"], ay)
        end = (b["x"], by) if bx > ax else (b["x"] + b["w"], by)
    else:
        start = (ax, a["y"] + a["h"]) if by > ay else (ax, a["y"])
        end = (bx, b["y"]) if by > ay else (bx, b["y"] + b["h"])
    return start, end


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, max(8, width)) or [text])


@tool("generate_image", [Capability.LLM, Capability.FS_WRITE],
      description="Generate an image with a multimodal model (last-resort fallback).")
async def generate_image(ctx: AgentContext, prompt: str, *, dest: str) -> dict[str, Any]:
    """Model image generation.

    Deliberately the last fallback: generated images are prone to garbled text and
    invented structure, which is disqualifying for a technical diagram. `render_diagram`
    is preferred whenever the figure can be expressed as boxes and arrows.
    """
    from ..llm import get_provider
    target = ctx.check_write_path(dest)
    resp = await get_provider().generate_image(prompt)
    if not resp.images:
        raise ToolError("image model returned no image")
    data = resp.images[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "path": target.relative_to(get_config().root).as_posix(),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "generated": True, "model": resp.model,
        "license": "generated", "attribution": f"Generated with {resp.model}",
        **_probe_image(data),
    }


# -- image probing ---------------------------------------------------------


def _looks_like_image(data: bytes) -> bool:
    return (
        data[:8] == b"\x89PNG\r\n\x1a\n"
        or data[:3] == b"\xff\xd8\xff"
        or data[:6] in (b"GIF87a", b"GIF89a")
        or data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    )


def _probe_image(data: bytes) -> dict[str, Any]:
    """Dimensions and format without requiring Pillow."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            return {"width": img.width, "height": img.height,
                    "format": (img.format or "").lower()}
    except Exception:
        pass
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return {"width": width, "height": height, "format": "png"}
    return {"width": 0, "height": 0, "format": "unknown"}
