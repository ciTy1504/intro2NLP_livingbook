"""Diagram renderer regression tests.

A real generated spec produced an unreadable figure: eight nodes drew as three
overlapping slabs with their labels stacked on top of one another, because the grid
layout let several nodes claim the same cell and boxes were sized before their text
was known. Figure QA caught it and refused to place it, which is the system working —
but the renderer should not produce it in the first place.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.tools.visual import _layout_nodes, _wrap  # noqa: E402


def labelled(ids_rows_cols):
    return [{"id": i, "text": i, "cols": 10, "rows": 1, "color": None,
             "shape": None, "row": r, "col": c} for i, r, c in ids_rows_cols]


def test_grid_collisions_are_resolved():
    """Two nodes claiming one cell must not be drawn on top of each other."""
    nodes = labelled([("a", 0, 0), ("b", 0, 0), ("c", 0, 0), ("d", 1, 0)])
    placed = _layout_nodes(nodes, "grid", 2.0, 1.0)
    positions = [(round(p["x"], 3), round(p["y"], 3)) for p in placed.values()]
    assert len(set(positions)) == len(positions), (
        f"nodes share a position: {positions}")


def test_every_node_is_placed():
    nodes = labelled([(chr(97 + i), i // 3, i % 3) for i in range(9)])
    placed = _layout_nodes(nodes, "grid", 2.0, 1.0)
    assert len(placed) == 9


def test_flow_layout_places_everything_distinctly():
    nodes = [{"id": f"n{i}", "text": f"n{i}", "cols": 8, "rows": 1,
              "color": None, "shape": None, "row": None, "col": None}
             for i in range(8)]
    placed = _layout_nodes(nodes, "flow", 2.0, 1.0)
    positions = [(round(p["x"], 3), round(p["y"], 3)) for p in placed.values()]
    assert len(set(positions)) == 8


def test_boxes_never_overlap():
    nodes = labelled([("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1)])
    placed = _layout_nodes(nodes, "grid", 2.4, 1.2)
    boxes = list(placed.values())
    for i, p in enumerate(boxes):
        for q in boxes[i + 1:]:
            overlap_x = p["x"] < q["x"] + q["w"] and q["x"] < p["x"] + p["w"]
            overlap_y = p["y"] < q["y"] + q["h"] and q["y"] < p["y"] + p["h"]
            assert not (overlap_x and overlap_y), "two boxes overlap"


def test_row_zero_is_at_the_top():
    """A reader follows a process downward, so row 0 must sit highest."""
    nodes = labelled([("top", 0, 0), ("bottom", 2, 0)])
    placed = _layout_nodes(nodes, "grid", 2.0, 1.0)
    assert placed["top"]["y"] > placed["bottom"]["y"]


def test_long_unbroken_labels_are_wrapped():
    """`w_i.w_j + b_i + b_j` has few spaces and used to run past its box."""
    wrapped = _wrap("w_i.w_j+b_i+b_j_and_more_tokens_without_spaces", 12)
    assert max(len(line) for line in wrapped.split("\n")) <= 12


def test_renders_the_spec_that_failed():
    """End to end on the real spec, checking the output is a plausible image."""
    from livingbook.tools import system_context

    spec = {
        "title": "Kiến trúc và Hàm mất mát GloVe",
        "layout": "grid",
        "nodes": [
            {"id": "x", "label": "Ma trận X_ij", "row": 0, "col": 0},
            {"id": "logx", "label": "Log(X_ij)", "row": 1, "col": 0},
            {"id": "f", "label": "Hàm trọng số f(X_ij)", "row": 1, "col": 1},
            {"id": "w", "label": "Vector w_i, w_j", "row": 0, "col": 1},
            {"id": "b", "label": "Độ chệch b_i, b_j", "row": 0, "col": 2},
            {"id": "pred", "label": "w_i.w_j + b_i + b_j", "row": 1, "col": 2},
            {"id": "err", "label": "Sai số bình phương", "row": 2, "col": 1},
            {"id": "j", "label": "Hàm mất mát J", "row": 3, "col": 1},
        ],
        "edges": [
            {"from": "x", "to": "logx", "label": "Logarit"},
            {"from": "x", "to": "f", "label": "Nhân trọng số"},
            {"from": "w", "to": "pred", "label": "Dự đoán"},
            {"from": "err", "to": "j"},
        ],
        "notes": [{"text": "Mục tiêu: w_i.w_j + b_i + b_j ≈ log(X_ij)"}],
    }

    # Inside the repository: write-path scoping refuses anything outside it, which is
    # the control doing its job, not an inconvenience to work around.
    from livingbook.config import get_config

    rel = "figures/generated/_test_diagram.png"
    dest = get_config().root / rel
    try:
        result = asyncio.run(
            system_context("test").call("render_diagram", spec=spec, dest=rel))
        assert dest.exists()
        # Wide enough for three columns and tall enough for four rows plus a note.
        assert result["width"] > 900, result
        assert result["height"] > 700, result
    finally:
        dest.unlink(missing_ok=True)
