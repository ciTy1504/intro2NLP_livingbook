# intro2NLP — Living Book

An autonomous research-to-publication pipeline wrapped around a Vietnamese textbook on
NLP and large language models. It watches the research ecosystem, decides whether
anything it finds should change the book, writes the change, verifies it, checks every
citation, sources or draws the figures, runs full-book QA, opens a pull request, and
emails a report — and it refuses to do any of that when the evidence does not warrant
it.

The manuscript lives in [`manuscript/`](manuscript/): 2 parts, 15 numbered chapters,
85 sections, ~143,000 words, 227 bibliography entries, 58 figures.

---

## What it actually does

```
         ┌──────────────────────────────── ORCHESTRATOR ───────────────────────────────┐
         │  persistent schedule · state machine · retries · approval gates             │
         └────────────────────────────────────┬───────────────────────────────────────┘
                                              ▼
      arXiv   OpenAlex   OpenReview/ACL   GitHub   HuggingFace   Benchmarks   Blogs   HN
         └────────┴───────────┴──────────────┴─────────┴─────────────┴─────────┴──────┘
                                              ▼
                                   RESEARCH SYNTHESIZER          (no network access)
                            dedupe · cluster · link · assess maturity
                                              ▼
                                     BOOK VERDICT AGENT
                    evidence floor + maturity gate, then an evidence-backed rationale
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
              IGNORE                       MONITOR                       CHANGE
            (recorded)              (re-evaluated weekly)                  │
                                                                           ▼
                                                                     WRITER AGENT
                                                        minimal LaTeX patch, book's voice
                                                                           │
          ┌──────────────┬──────────────┬─────────────────┬────────────────┤
          ▼              ▼              ▼                 ▼                ▼
    TECHNICAL      CITATION        EDITORIAL +       VISUAL ENGINE     FULL BOOK QA
    VERIFIER       SUBSYSTEM       CONSISTENCY       search/verify/    deterministic +
                   audit→find→     verifiers         licence/generate  semantic + global
                   verify→bibtex
          └──────────────┴──────────────┴─────────────────┴────────────────┘
                                              ▼
                                          APPROVED
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                              GIT AGENT              EMAIL AGENT
                         branch → build → commit    only on a real change
                         → push → PR → CI
```

Every arrow is a persisted state transition with a stored artifact. Any change in the
book traces back through verdict → cluster → evidence → source.

---

## Getting started

```bash
pip install -r requirements.txt

cp config/secrets.example.env secrets/.env     # then fill it in
python -m livingbook.cli doctor                # check everything before relying on it
python -m livingbook.cli index                 # build the knowledge base (one-off, ~15 min)

python -m livingbook.cli cycle                 # one full research-to-publication cycle
python -m livingbook.cli daemon                # run continuously on the schedule
```

`doctor` is the command to run first and whenever something looks wrong. It validates
every agent contract, checks the manuscript parses, reports which credentials are
missing and what that disables, and confirms the LaTeX toolchain is present.

### What you need to supply

| Secret | Needed for | Without it |
|---|---|---|
| `secrets/gemini_keypool.txt` | everything | nothing runs (already populated) |
| `GITHUB_TOKEN` | pushing, opening PRs | changes are committed locally only |
| `GMAIL_USER` / `GMAIL_PASS` | notification email | changes publish silently |
| `EMAIL_RECIPIENTS` | notification email | — |
| `SEMANTIC_SCHOLAR_API_KEY` | one extra source | that source stays disabled |

Everything in `secrets/` is gitignored. No credential is ever written into a log line,
an artifact, a commit, or a remote URL.

---

## The commands you will actually use

```bash
python -m livingbook.cli status            # pipelines, research lifecycle, schedule
python -m livingbook.cli qa                # full-book QA on demand
python -m livingbook.cli kb impact "speculative decoding" "draft model"
python -m livingbook.cli kb search "compute-optimal tokens per parameter"
python -m livingbook.cli kb outline        # the whole book in ~3,300 tokens
python -m livingbook.cli trace --pipeline-id pipe_xxx   # the full evidence chain
python -m livingbook.cli approve pipe_xxx  # release a gated change
python -m livingbook.cli llm probe         # re-measure which models respond
python -m livingbook.cli visual            # resolve outstanding figure placeholders
```

`kb impact` is the query the architecture is built around: give it the concepts from a
new paper and it returns exactly the sections, claims and citations at risk — without
reading the manuscript.

---

## How it decides not to act

Most research should not change a textbook, and the system is built to reach that
conclusion cheaply and defensibly. Three gates run **before** any model is asked for an
opinion:

**Evidence floor** — a cluster needs at least one scientific or independently verified
source and at least two *independent* sources. Three arXiv papers count as one source,
because they arrived through one channel.

**Maturity gate** — maturity is computed from the evidence mix, never from recency, and
each decision requires a minimum:

| Maturity | Means | Unlocks |
|---|---|---|
| `speculative` | one source, or author claims only | nothing |
| `emerging` | replicated **or** implemented | reference, footnote, extend |
| `consolidating` | multiple independent sources **and** real adoption | new section, rewrite |
| `mature` | settled, adopted, independently benchmarked | replacing obsolete content |

**Evidence kind** — enforced in the type system, not by prompt. `EvidenceKind` is a
closed enum and each source may only emit the kinds it can justify. A Hacker News
thread reaches the verdict agent as `community`/`anecdotal`; it cannot become
`scientific` because a model decided it should.

No numeric score decides anything. `confidence` is recorded for triage. The decision
rests on an evidence-backed rationale that must name the specific evidence and the
specific affected content.

---

## Citations

The citation subsystem is built to reject.

```
Auditor      finds claims needing evidence they do not have
Finder       searches for the work that ESTABLISHES the claim, not text resembling it
Verifier     reads the source, quotes the supporting passage, checks the numbers
BibTeX       validates against DOI-registry metadata, then writes references.bib
```

The verifier explicitly hunts **citation laundering** — where paper A says "B showed X"
and the book then cites A for X. That is rejected even when X is true, and the real
primary source is pursued instead. A numeric claim accepted without a verbatim quoted
passage is rejected automatically.

A claim that cannot be sourced does not get a weaker citation. It goes back to the
Writer to be softened or removed.

---

## Figures

The manuscript's own `\bookimage{key}{description}` macro is already a figure
specification: it renders a placeholder box containing the description when the image
is absent. The Visual Engine consumes exactly that.

```
need → search (Wikimedia, Openverse — sources that carry licence metadata)
     → semantic verification (look at the image; a keyword match is not a match)
     → licence check (unknown licence = rejected)
     → refine the description and search again
     → draw an original diagram from a declarative spec
     → figure QA (legible at print size? caption honest? alt text present?)
     → AssetManager places it — the only agent permitted to write manuscript/images/
```

Figure provenance is appended to `manuscript/images/SOURCES.md`, the ledger the author
already maintains by hand, rather than to a parallel record.

---

## Repository layout

```
manuscript/          the book — the only publishable content
livingbook/
  config.py  obs.py  cli.py
  llm/               provider interface · Gemini adapter · key pool · cache
  tools/             62 primitives, capability-tagged
  skills/            24 reusable workflows
  agents/            28 agents; contracts live in config/agents.yaml
  knowledge/         LaTeX parser · indexer · knowledge graph · retrieval · BibTeX
  research/          pydantic contracts for the whole pipeline
  state/             SQLite store · state machine · artifacts · provenance
  orchestrator/      scheduler · pipeline driver · visual flow
config/              config.yaml · agents.yaml · sources.yaml · style_guide.md
secrets/             gitignored
knowledge/ research/ figures/ reviews/ changelog/ logs/ state/
tests/  scripts/  .github/workflows/
AUDIT.md             what the existing projects and environment actually contained
ARCHITECTURE.md      the full design, and why each decision was made
```

### Agent · Skill · Tool

They are genuinely different things, and the distinction is enforced:

- A **tool** is a primitive. It does one thing, does no reasoning, and declares the
  capabilities it needs. 62 of them.
- A **skill** is a reusable workflow — prompts plus tool orchestration plus output
  parsing. Stateless, and runs under the calling agent's permissions, so the same
  implementation behaves differently for different agents. 24 of them.
- An **agent** is a declared role: objective, IO contract, allowed skills, allowed
  tools, constraints, failure policy. Its contract is in `config/agents.yaml`, not in
  the code. 28 of them.

An agent may only call a tool that is both in its allowlist *and* whose capabilities
are a subset of its grants, and writes are additionally scoped by path. The tests in
[`tests/test_permissions.py`](tests/test_permissions.py) assert the specific denials the
design depends on: the synthesiser cannot reach the network, only the AssetManager
writes figures, only the BibTeX validator writes the bibliography, and the orchestrator
has no LLM access at all.

---

## Operational notes

**The key pool is shared and contended.** 254 keys. A full book index measured a 14%
success rate, which means the quota is largely upstream of the individual key —
rotating keys does not help, pausing does. The rate limiter watches the recent 429
ratio and backs off globally. Keys are retired only after five consecutive hard
failures; `llm revive` brings them all back after a provider incident.

**No Pro-tier capacity exists on this pool.** Every role resolves to an ordered model
chain and steps sideways on a 503. Re-measure with `llm probe` — the chains in
`config/config.yaml` are worth revisiting periodically.

**Builds happen in a disposable copy.** `minted` v3 cannot resolve
`TEXMF_OUTPUT_DIRECTORY`, so an out-of-tree build fails; building in place inside a
temporary copy keeps `manuscript/` free of `.aux` files the Git agent would otherwise
commit.

**Crashes resume, they do not restart.** Every state transition commits with its
artifact, and completions are cached content-addressably, so re-entering a pipeline
replays cached work rather than re-billing it.

---

## Safety

Three properties the design will not trade away:

1. **Nothing enters the book without evidence.** The Writer may only state facts
   present in the research context it was given, every claim is audited, and every
   citation is verified against the source text.
2. **An agent cannot widen its own authority.** Permissions live in config, writes are
   path-scoped, and [`agent-pr.yml`](.github/workflows/agent-pr.yml) fails any agent PR
   that touches `config/`, `livingbook/`, `tests/` or `.github/`.
3. **Everything is reversible and attributable.** Changes land on
   `agent/research/<date>-<topic>` branches with provenance trailers and a changelog
   entry; `main` is never committed to directly; `auto_merge` is off by default.

---

*Baseline manuscript by Đinh Công Thái. The Living Book system was built around it and
does not alter its voice, conventions or structure.*
