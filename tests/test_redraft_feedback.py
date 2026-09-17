"""Sending a draft back must actually tell the Writer something.

A backward edge into DRAFTED exists so the Writer can soften or drop a claim that
could not be sourced. That only works if the reason travels with it. It did not:
`unsupported_claims` was written into pipeline data and exposed over MCP, but never
read, so every re-draft was built from inputs identical to the draft that had just
been rejected. One pipeline spent seventeen hours failing to source the same five
claims before max_revisions parked it in NEEDS_HUMAN.
"""

from types import SimpleNamespace

from livingbook.agents.writer.writer import WriterAgent


def _target():
    return SimpleNamespace(section_ref="2.6", node_id="sec:serving", file="chap2_6.tex",
                           change="add a paragraph on speculative decoding",
                           estimated_lines=12)


def _cluster():
    return SimpleNamespace(concepts=["speculative decoding", "inference"])


def _verdict():
    from livingbook.research.models import VerdictDecision
    return SimpleNamespace(decision=VerdictDecision.EXTEND_SECTION)


def test_a_first_draft_carries_no_redraft_section():
    out = WriterAgent._instruction(
        None, _verdict(), _target(), _cluster(),
        unsupported_claims=[], technical_findings=[])

    assert "RE-DRAFT" not in out
    assert "TARGET: 2.6" in out


def test_a_redraft_is_told_which_claims_failed_and_why():
    out = WriterAgent._instruction(
        None, _verdict(), _target(), _cluster(),
        unsupported_claims=[
            {"claim": "PELM achieves energy savings up to 52.4% over baseline",
             "why": "needs_primary"},
            {"claim": "LoopSpec achieves 6.83x speedup on HumanEval+",
             "why": "reject"},
        ],
        technical_findings=[])

    # The claim text itself has to be present — naming the count is not enough for the
    # Writer to know which sentence to change.
    assert "PELM achieves energy savings up to 52.4%" in out
    assert "LoopSpec achieves 6.83x speedup" in out
    assert "needs_primary" in out
    # And it must be told not to reach for a secondary source, which is what the
    # citation verifier already rejected.
    assert "laundering" in out
    assert "RE-DRAFT" in out


def test_a_redraft_is_told_about_technical_problems():
    out = WriterAgent._instruction(
        None, _verdict(), _target(), _cluster(),
        unsupported_claims=[],
        technical_findings=[{"detail": "beam search described as greedy decoding"}])

    assert "beam search described as greedy decoding" in out
    assert "TECHNICAL PROBLEMS" in out


def test_the_pipeline_hands_the_stored_feedback_to_the_writer():
    """The wiring, not just the prompt: _draft must read what CITATION_FIND stored."""
    import inspect

    from livingbook.orchestrator import pipeline as pipeline_module

    src = inspect.getsource(pipeline_module.PipelineDriver._draft)
    assert "unsupported_claims" in src, "_draft must pass the rejected claims through"
    assert "technical_findings" in src, "_draft must pass technical findings through"
