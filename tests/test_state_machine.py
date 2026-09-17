"""State machine, evidence-gating and provenance tests.

The evidence tests are the important ones: they encode the rule that a single
unreplicated paper must not change the book, and that a forum thread cannot become
scientific evidence. Those are the guarantees that make the pipeline trustworthy, and
they should fail loudly if someone relaxes them.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.research.models import (  # noqa: E402
    Evidence,
    EvidenceKind,
    EvidenceStrength,
    Maturity,
    ResearchCluster,
    SourceRef,
)
from livingbook.skills.research.synthesis import compute_maturity  # noqa: E402
from livingbook.state.machine import (  # noqa: E402
    BACKWARD_EDGES,
    FORWARD,
    TERMINAL,
    TRANSITIONS,
    IllegalTransition,
    State,
    StateMachine,
)
from livingbook.state.store import Store  # noqa: E402


@pytest.fixture()
def machine():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "test.db")
        yield StateMachine(store)
        store.close()


# ── the state graph itself ────────────────────────────────────────────────


def test_forward_path_is_fully_connected():
    for a, b in zip(FORWARD, FORWARD[1:]):
        assert b in TRANSITIONS[a], f"{a.value} -> {b.value} is not a legal transition"


def test_every_state_can_fail_or_escalate():
    for state, targets in TRANSITIONS.items():
        if state in TERMINAL or state == State.NEEDS_HUMAN:
            continue
        assert State.FAILED in targets, f"{state.value} cannot fail"
        assert State.NEEDS_HUMAN in targets, f"{state.value} cannot escalate"


def test_terminal_states_are_terminal():
    assert not TRANSITIONS[State.COMPLETED]
    assert not TRANSITIONS[State.REJECTED]


def test_backward_edges_are_legal_transitions():
    for frm, to in BACKWARD_EDGES:
        assert to in TRANSITIONS[frm], f"backward edge {frm.value}->{to.value} is illegal"


# ── behaviour ─────────────────────────────────────────────────────────────


def test_illegal_transition_raises(machine):
    pipe = machine.create(cluster_id=None, state=State.SYNTHESIZED)
    with pytest.raises(IllegalTransition):
        machine.transition(pipe, State.GIT_PUBLISH)


def test_transition_is_recorded(machine):
    pipe = machine.create(state=State.SYNTHESIZED)
    pipe = machine.transition(pipe, State.VERDICT_PENDING, note="test")
    history = machine.history(pipe.id)
    assert [h["to_state"] for h in history] == ["SYNTHESIZED", "VERDICT_PENDING"]


def test_state_data_accumulates(machine):
    pipe = machine.create(state=State.SYNTHESIZED)
    pipe = machine.transition(pipe, State.VERDICT_PENDING, data={"a": 1})
    pipe = machine.transition(pipe, State.VERDICT_APPROVED, data={"b": 2})
    assert pipe.data["a"] == 1 and pipe.data["b"] == 2


def test_revision_limit_escalates_to_human(machine):
    """Bounded backward edges: a patch that cannot pass QA must reach a person."""
    pipe = machine.create(state=State.SYNTHESIZED)
    for state in (State.VERDICT_PENDING, State.VERDICT_APPROVED, State.DRAFTED):
        pipe = machine.transition(pipe, state)

    machine.max_revisions = 2
    for _ in range(machine.max_revisions):
        pipe = machine.transition(pipe, State.TECHNICAL_VERIFY)
        pipe = machine.transition(pipe, State.DRAFTED, note="defect found")
    assert pipe.revisions == machine.max_revisions

    pipe = machine.transition(pipe, State.TECHNICAL_VERIFY)
    pipe = machine.transition(pipe, State.DRAFTED, note="defect again")
    assert pipe.state == State.NEEDS_HUMAN, "exceeding max_revisions must escalate"


def test_active_excludes_parked_and_terminal(machine):
    a = machine.create(state=State.SYNTHESIZED)
    b = machine.create(state=State.SYNTHESIZED)
    machine.transition(b, State.VERDICT_PENDING)
    machine.transition(machine.get(b.id), State.MONITORING)
    active_ids = {p.id for p in machine.active()}
    assert a.id in active_ids
    assert b.id not in active_ids, "MONITORING pipelines must not be ticked"


# ── evidence discipline ───────────────────────────────────────────────────


def _ref(source: str, n: int = 0) -> SourceRef:
    return SourceRef(source=source, source_id=f"{source}-{n}", title=f"{source} item {n}")


def test_community_source_cannot_emit_scientific_evidence():
    """A forum thread is a signal, whatever the model decides to call it."""
    evidence = Evidence.for_source(
        "hackernews", EvidenceKind.SCIENTIFIC, "X is 3x faster",
        strength=EvidenceStrength.STRONG, provenance=[_ref("hackernews")])
    assert evidence.kind != EvidenceKind.SCIENTIFIC
    assert evidence.strength == EvidenceStrength.WEAK


def test_github_cannot_emit_scientific_evidence():
    evidence = Evidence.for_source(
        "github", EvidenceKind.SCIENTIFIC, "beats the baseline",
        provenance=[_ref("github")])
    assert evidence.kind in (EvidenceKind.PRACTICAL, EvidenceKind.ADOPTION)


def test_arxiv_may_emit_scientific_evidence():
    evidence = Evidence.for_source(
        "arxiv", EvidenceKind.SCIENTIFIC, "measured 2x speedup",
        provenance=[_ref("arxiv")])
    assert evidence.kind == EvidenceKind.SCIENTIFIC


def test_evidence_requires_provenance():
    with pytest.raises(ValueError):
        Evidence(kind=EvidenceKind.SCIENTIFIC, statement="unattributed", provenance=[])


# ── maturity ──────────────────────────────────────────────────────────────


def test_single_paper_is_speculative():
    """The brief's rule: one new paper does not mean the book changes."""
    evidence = [Evidence.for_source("arxiv", EvidenceKind.SCIENTIFIC, "result",
                                    provenance=[_ref("arxiv")])]
    assert compute_maturity(evidence, [{}]) == Maturity.SPECULATIVE


def test_paper_plus_implementation_is_emerging():
    evidence = [
        Evidence.for_source("arxiv", EvidenceKind.SCIENTIFIC, "r", provenance=[_ref("arxiv")]),
        Evidence.for_source("github", EvidenceKind.PRACTICAL, "impl",
                            provenance=[_ref("github")]),
    ]
    assert compute_maturity(evidence, [{}, {}]) == Maturity.EMERGING


def test_replication_and_adoption_reaches_consolidating():
    evidence = [
        Evidence.for_source("arxiv", EvidenceKind.SCIENTIFIC, "r", provenance=[_ref("arxiv")]),
        Evidence.for_source("openalex", EvidenceKind.INDEPENDENT_VERIFICATION, "repro",
                            provenance=[_ref("openalex")]),
        Evidence.for_source("github", EvidenceKind.ADOPTION, "widely used",
                            provenance=[_ref("github")]),
    ]
    assert compute_maturity(evidence, [{}] * 3) == Maturity.CONSOLIDATING


def test_community_only_never_exceeds_speculative():
    """No amount of discussion makes something a settled result."""
    evidence = [
        Evidence.for_source("hackernews", EvidenceKind.COMMUNITY, f"thread {i}",
                            provenance=[_ref("hackernews", i)])
        for i in range(12)
    ]
    assert compute_maturity(evidence, [{}] * 12) == Maturity.SPECULATIVE


# ── cluster accounting ────────────────────────────────────────────────────


def test_independent_source_count_ignores_duplicates_within_a_source():
    cluster = ResearchCluster(
        id="c1", title="t",
        provenance=[_ref("arxiv", 0), _ref("arxiv", 1), _ref("arxiv", 2)])
    assert cluster.independent_source_count() == 1, (
        "three arXiv papers are one channel, not three")


def test_independent_source_count_across_sources():
    cluster = ResearchCluster(
        id="c1", title="t",
        provenance=[_ref("arxiv"), _ref("github"), _ref("openalex")])
    assert cluster.independent_source_count() == 3


def test_failed_can_resume_from_where_it_died(machine):
    """A transient outage must not cost the work already done.

    A pipeline that failed at CITATION_VERIFY should resume at CITATION_FIND, not be
    forced back to DRAFTED — that would discard a verified draft and a completed
    citation search because the provider had a bad minute.
    """
    pipe = machine.create(state=State.SYNTHESIZED)
    for state in (State.VERDICT_PENDING, State.VERDICT_APPROVED, State.DRAFTED,
                  State.TECHNICAL_VERIFY, State.CITATION_AUDIT, State.CITATION_FIND):
        pipe = machine.transition(pipe, state)

    pipe = machine.fail(pipe, "transient provider outage")
    assert pipe.state == State.FAILED

    pipe = machine.transition(pipe, State.CITATION_FIND, note="retried")
    assert pipe.state == State.CITATION_FIND


def test_failed_can_reach_every_working_state(machine):
    for target in FORWARD[:-1]:
        assert target in TRANSITIONS[State.FAILED], (
            f"FAILED cannot resume at {target.value}")
    assert State.COMPLETED not in TRANSITIONS[State.FAILED], (
        "a failed pipeline must never jump straight to COMPLETED")


# ── scheduler behaviour ───────────────────────────────────────────────────


def test_scheduler_starts_due_jobs_concurrently():
    """A long job must not starve a short one.

    Measured failure: `pipeline_tick`, scheduled every 15 minutes, had not run in
    3.8 hours because `discovery` was still going and the tick ran jobs in sequence.
    """
    import asyncio as aio
    import tempfile as tf
    from livingbook.orchestrator.scheduler import Scheduler

    with tf.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "sched.db")
        sched = Scheduler(store)
        order: list[str] = []

        async def slow():
            order.append("slow-start")
            await aio.sleep(0.4)
            order.append("slow-end")

        async def quick():
            order.append("quick")

        sched.register("slow", slow, interval_seconds=60)
        sched.register("quick", quick, interval_seconds=60)

        async def run():
            started = await sched.tick()
            await aio.sleep(0.1)          # the slow job is still running
            assert "quick" in order, "the quick job waited on the slow one"
            await sched.drain(timeout=5)
            return started

        started = aio.run(run())
        assert set(started) == {"slow", "quick"}
        assert order.index("quick") < order.index("slow-end")
        store.close()


def test_retry_after_is_capped():
    """A server asking for 19 hours is refusing, not pausing."""
    from livingbook.tools.http import MAX_RETRY_AFTER

    assert MAX_RETRY_AFTER <= 120, (
        "an unbounded Retry-After parked a research agent for 19 hours")


# -- lock reclaim ------------------------------------------------------------
#
# A daemon killed mid-job leaves running=1 behind. With only a time-based staleness
# window, its own replacement then sat idle for three hours. These pin the exact
# rule: reclaim a lock whose owning PID is gone on this host, and never one that
# might still be held.

def test_a_lock_from_a_dead_process_on_this_host_is_reclaimed():
    import socket

    from livingbook.orchestrator.scheduler import _owner_is_gone

    import subprocess
    import sys

    # A PID that has certainly exited: start one, wait for it, then ask.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert _owner_is_gone(f"{socket.gethostname()}:{proc.pid}") is True

    # And one the kernel never assigns.
    assert _owner_is_gone(f"{socket.gethostname()}:999999999") is True


def test_a_lock_that_might_still_be_held_is_never_stolen():
    import os
    import socket

    from livingbook.orchestrator.scheduler import _owner_is_gone

    host = socket.gethostname()
    # Our own live process.
    assert _owner_is_gone(f"{host}:{os.getpid()}") is False
    # Another machine's PID number means nothing here — fall back to the time window.
    assert _owner_is_gone("some-other-host:6264") is False
    # Nothing recorded, or unparseable.
    assert _owner_is_gone(None) is False
    assert _owner_is_gone("") is False
    assert _owner_is_gone("weird-value-without-a-pid") is False


def test_probing_a_pid_does_not_kill_it():
    """os.kill(pid, 0) would TerminateProcess on Windows. This must not."""
    import os
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        from livingbook.orchestrator.scheduler import _pid_is_alive

        assert _pid_is_alive(proc.pid) is True
        assert _pid_is_alive(proc.pid) is True          # probing twice is still safe
        assert proc.poll() is None, "the probe terminated the process it asked about"
    finally:
        proc.kill()
        proc.wait(timeout=10)

    assert _pid_is_alive(os.getpid()) is True
