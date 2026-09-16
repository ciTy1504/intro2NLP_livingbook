# STEP 3 — Living Book system architecture

Design decisions here are grounded in the measurements in [AUDIT.md](AUDIT.md). Where a decision exists
because of something we found by probing (no Pro-tier LLM capacity, Reddit blocked, `minted` breaking under
`-output-directory`), that is called out inline.

---

## 1. System architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR    scheduling · state machine · retries · approval gates  │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ invokes, never reasons about research
┌───────────────────────────────▼────────────────────────────────────────┐
│ AGENTS          role + IO contract + allowed skills/tools + failure    │
│                 policy + verification requirements                     │
└───────────────────────────────┬────────────────────────────────────────┘
┌───────────────────────────────▼────────────────────────────────────────┐
│ SKILLS          reusable capability/workflow, shared across agents     │
└───────────────────────────────┬────────────────────────────────────────┘
┌───────────────────────────────▼────────────────────────────────────────┐
│ TOOLS           primitive side-effecting operations, capability-tagged │
└───────────────────────────────┬────────────────────────────────────────┘
┌───────────────────────────────▼────────────────────────────────────────┐
│ EXTERNAL        arXiv OpenAlex Crossref DataCite GitHub HF OpenReview  │
│                 ACL HN Wikimedia Openverse RSS · Gemini · SMTP · git   │
└────────────────────────────────────────────────────────────────────────┘

Cross-cutting (readable from every layer, writable only through their owners):

  Persistent State · Research Memory · Book Knowledge Base · Knowledge Graph
  Artifact Store · Provenance · Structured Logging · Versioning · Configuration
```

The load-bearing rule: **an agent never touches an external system directly and never rotates a key.** It
requests a tool through its `AgentContext`, which enforces permissions, logs the call with run/agent/skill
attribution, and routes LLM traffic through the provider layer.

---

## 2. Agent architecture

An agent is a declared **role**, not a prompt. Every agent has:

| Field | Meaning |
|---|---|
| `objective` | One sentence; goes into the system prompt |
| `input_schema` / `output_schema` | Pydantic models — output is validated, never free text |
| `allowed_skills` | Skills it may invoke |
| `allowed_tools` | Explicit tool allowlist (intersected with capability grants) |
| `capabilities` | Coarse grants: `SEARCH`, `FETCH`, `SCHOLARLY`, `FS_READ`, `FS_WRITE`, `GIT`, `EMAIL`, `LLM`, `BUILD`, `KB_READ`, `KB_WRITE` |
| `constraints` | Hard limits (max patch size, max tokens, forbidden paths) |
| `failure_policy` | `retry(n, backoff)` · `degrade` · `fail_pipeline` · `escalate_to_human` |
| `verification_requirements` | Which downstream verifier must pass before its output is accepted |
| `model_role` | `fast` / `balanced` / `deep` / `embedding` — resolved to a fallback chain by the provider |

### The 25 agents

**Research** (`agents/research/`) — one per source, because source semantics differ:

| Agent | Watches | Produces |
|---|---|---|
| `ArxivAgent` | arXiv cs.CL / cs.LG / cs.AI | papers + methods + claims + benchmarks + limitations |
| `ScholarlyAgent` | OpenAlex (+ Semantic Scholar if keyed) | citation counts, venue, related work, influence |
| `GithubResearchAgent` | GitHub repos, releases, activity | implementations, adoption signals, paper↔code links |
| `HuggingFaceAgent` | HF models/datasets/daily-papers | releases, downloads, ecosystem uptake |
| `ResearchBlogAgent` | curated RSS set | practitioner explanation, lab announcements |
| `ConferenceAgent` | OpenReview, ACL Anthology | peer-reviewed acceptance, reviewer critique |
| `BenchmarkAgent` | benchmark repos/leaderboards/datasets | benchmark existence, saturation, reported numbers |
| `CommunityAgent` | Hacker News, GH issues/discussions, HF discussions | recurring pain points, emerging terminology, disagreement |
| `ResearchSynthesizer` | *(no external access)* | deduplicated, clustered `ResearchCluster` objects |

**Verdict** — `BookVerdictAgent`.
**Writer** — `WriterAgent`.
**Citation** — `CitationAuditor`, `CitationFinder`, `CitationVerifier`, `BibtexValidator`.
**Verification** — `TechnicalVerifier`, `EditorialVerifier`, `CrossChapterConsistencyAgent`, `BookQAAgent`.
**Visual** — `VisualNeedDetector`, `ImageSearchAgent`, `SemanticImageVerifier`, `LicenseChecker`,
`DiagramGeneratorAgent`, `FigureQAAgent`, `AssetManager`.
**Delivery** — `GitAgent`, `EmailAgent`.

`ResearchSynthesizer` deliberately has **no search and no fetch capability**. It can only reason over what
the source agents already collected and persisted. This is what stops the synthesizer from quietly becoming
a 10th research agent with unattributed evidence.

---

## 3. Skill architecture

A skill is a reusable workflow: prompt templates + tool orchestration + output parsing. Skills are stateless
and take an `AgentContext`, so the same skill run by two agents inherits each agent's permissions.

```
skills/research/     research_discovery · paper_analysis · claim_extraction
                     repo_analysis · community_signal_analysis · research_synthesis
skills/book/         book_retrieval · book_placement · chapter_editing
                     cross_chapter_consistency · book_qa
skills/citation/     citation_audit · citation_search · citation_verification
                     bibtex_verification
skills/visual/       visual_need_detection · visual_search · visual_verification
                     figure_generation · figure_qa
skills/verification/ technical_verification · editorial_verification
skills/publishing/   git_publishing · email_reporting
```

Reuse examples: `claim_extraction` is used by `ArxivAgent` (claims made *by* a paper) **and** by the KB
indexer (claims made *by the book*) — same extraction contract, different corpus. `book_retrieval` is used
by the verdict agent, the writer, every verifier and the QA agent.

---

## 4. Tool architecture

Tools are `async` functions registered with `@tool(name, capabilities={...})`. They do one thing, do no
reasoning, and return plain data. Every call passes through `AgentContext.call()`, which checks permission,
times the call, and emits a structured event.

```
tools/web/        web_search · fetch_url · fetch_rss
tools/scholarly/  search_arxiv · fetch_arxiv_paper · search_openalex · fetch_openalex_work
                  search_semantic_scholar · search_crossref · search_openreview
                  search_acl_anthology · verify_doi · resolve_paper_identity
                  retrieve_bibtex · download_pdf · parse_pdf
tools/code/       search_github · fetch_github_repo · fetch_github_activity
                  search_huggingface · fetch_hf_model · search_hf_papers
tools/community/  search_hackernews · search_github_discussions · search_reddit*
tools/filesystem/ read_file · write_file · patch_file · list_files · search_repository
tools/book/       build_book · run_tests · latex_lint · bib_validate
                  check_links · check_figures
tools/kb/         kb_query · kb_retrieve_context · kb_upsert
tools/visual/     search_wikimedia_images · search_openverse_images · download_image
                  inspect_image · render_diagram · generate_image · optimize_image
tools/git/        git_status · git_diff · create_branch · git_commit · git_push
                  create_pull_request
tools/email/      send_email
tools/llm/        gemini_generate · gemini_structured_output · gemini_embedding
```

`*` `search_reddit` is implemented but **disabled by default** — Reddit returns 403 to script user-agents
(measured). It is a config flag, not dead code, so it can be re-enabled if credentials appear.

---

## 5. Agent → Skill mapping

| Agent | Skills |
|---|---|
| ArxivAgent | research_discovery, paper_analysis, claim_extraction |
| ScholarlyAgent | research_discovery, paper_analysis |
| GithubResearchAgent | research_discovery, repo_analysis |
| HuggingFaceAgent | research_discovery, repo_analysis |
| ResearchBlogAgent | research_discovery, paper_analysis |
| ConferenceAgent | research_discovery, paper_analysis, claim_extraction |
| BenchmarkAgent | research_discovery, repo_analysis |
| CommunityAgent | research_discovery, community_signal_analysis |
| ResearchSynthesizer | research_synthesis |
| BookVerdictAgent | book_retrieval, book_placement |
| WriterAgent | book_retrieval, chapter_editing, visual_need_detection |
| CitationAuditor | book_retrieval, citation_audit |
| CitationFinder | citation_search |
| CitationVerifier | citation_verification |
| BibtexValidator | bibtex_verification |
| TechnicalVerifier | book_retrieval, technical_verification |
| EditorialVerifier | book_retrieval, editorial_verification |
| CrossChapterConsistencyAgent | book_retrieval, cross_chapter_consistency |
| BookQAAgent | book_qa, book_retrieval |
| VisualNeedDetector | visual_need_detection |
| ImageSearchAgent | visual_search |
| SemanticImageVerifier | visual_verification |
| LicenseChecker | visual_verification |
| DiagramGeneratorAgent | figure_generation |
| FigureQAAgent | figure_qa |
| AssetManager | *(deterministic — no skill)* |
| GitAgent | git_publishing |
| EmailAgent | email_reporting |

## 6. Skill → Tool mapping

| Skill | Tools |
|---|---|
| research_discovery | search_arxiv, search_openalex, search_github, search_huggingface, search_openreview, search_acl_anthology, search_hackernews, fetch_rss, web_search |
| paper_analysis | fetch_arxiv_paper, fetch_openalex_work, download_pdf, parse_pdf, gemini_structured_output |
| claim_extraction | gemini_structured_output |
| repo_analysis | fetch_github_repo, fetch_github_activity, fetch_hf_model, fetch_url, gemini_structured_output |
| community_signal_analysis | search_hackernews, search_github_discussions, gemini_structured_output |
| research_synthesis | kb_query, gemini_embedding, gemini_structured_output |
| book_retrieval | kb_query, kb_retrieve_context, gemini_embedding, read_file |
| book_placement | kb_query, kb_retrieve_context, gemini_structured_output |
| chapter_editing | read_file, write_file, patch_file, gemini_generate, gemini_structured_output |
| citation_audit | kb_retrieve_context, gemini_structured_output |
| citation_search | search_openalex, search_arxiv, search_crossref, search_semantic_scholar, web_search, gemini_structured_output |
| citation_verification | fetch_openalex_work, download_pdf, parse_pdf, verify_doi, resolve_paper_identity, gemini_structured_output |
| bibtex_verification | retrieve_bibtex, verify_doi, bib_validate, gemini_structured_output |
| technical_verification | kb_retrieve_context, parse_pdf, fetch_url, gemini_structured_output |
| editorial_verification | kb_retrieve_context, gemini_structured_output |
| cross_chapter_consistency | kb_query, gemini_embedding, gemini_structured_output |
| book_qa | build_book, run_tests, latex_lint, bib_validate, check_links, check_figures, kb_query, gemini_structured_output |
| visual_need_detection | kb_retrieve_context, gemini_structured_output |
| visual_search | search_wikimedia_images, search_openverse_images, web_search, download_image |
| visual_verification | inspect_image, gemini_structured_output |
| figure_generation | render_diagram, generate_image, optimize_image, gemini_structured_output |
| figure_qa | inspect_image, gemini_structured_output |
| git_publishing | git_status, git_diff, create_branch, git_commit, git_push, create_pull_request, build_book, run_tests |
| email_reporting | send_email, kb_query |

---

## 7. Tool permission matrix

Rows are agents, columns capability grants. `✓` = granted, `·` = denied. Denial is enforced at call time and
a violation raises `PermissionDenied` and is logged as a `permission_denied` event.

| Agent | SEARCH | FETCH | SCHOLARLY | FS_READ | FS_WRITE | KB_READ | KB_WRITE | LLM | BUILD | GIT | EMAIL |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ArxivAgent | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| ScholarlyAgent | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| GithubResearchAgent | ✓ | ✓ | · | · | · | ✓ | ✓ | ✓ | · | · | · |
| HuggingFaceAgent | ✓ | ✓ | · | · | · | ✓ | ✓ | ✓ | · | · | · |
| ResearchBlogAgent | ✓ | ✓ | · | · | · | ✓ | ✓ | ✓ | · | · | · |
| ConferenceAgent | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| BenchmarkAgent | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| CommunityAgent | ✓ | ✓ | · | · | · | ✓ | ✓ | ✓ | · | · | · |
| **ResearchSynthesizer** | **·** | **·** | **·** | · | · | ✓ | ✓ | ✓ | · | · | · |
| BookVerdictAgent | · | · | · | ✓ | · | ✓ | ✓ | ✓ | · | · | · |
| WriterAgent | · | · | · | ✓ | ✓¹ | ✓ | · | ✓ | · | · | · |
| CitationAuditor | · | · | · | ✓ | · | ✓ | ✓ | ✓ | · | · | · |
| CitationFinder | ✓ | ✓ | ✓ | · | · | ✓ | · | ✓ | · | · | · |
| CitationVerifier | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| BibtexValidator | · | ✓ | ✓ | ✓ | ✓² | ✓ | · | ✓ | · | · | · |
| TechnicalVerifier | · | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | · | · |
| EditorialVerifier | · | · | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| CrossChapterConsistency | · | · | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| BookQAAgent | · | · | · | ✓ | · | ✓ | · | ✓ | ✓ | · | · |
| VisualNeedDetector | · | · | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| ImageSearchAgent | ✓ | ✓ | · | · | ✓³ | ✓ | · | ✓ | · | · | · |
| SemanticImageVerifier | · | · | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| LicenseChecker | · | ✓ | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| DiagramGeneratorAgent | · | · | · | ✓ | ✓³ | ✓ | · | ✓ | · | · | · |
| FigureQAAgent | · | · | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| AssetManager | · | · | · | ✓ | ✓⁴ | ✓ | ✓ | · | · | · | · |
| GitAgent | · | · | · | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| EmailAgent | · | · | · | ✓ | · | ✓ | ✓ | ✓ | · | · | ✓ |
| Orchestrator | · | · | · | ✓ | · | ✓ | ✓ | · | · | · | · |

¹ `WriterAgent` may write **only** under `manuscript/` and **only** into files named by an approved verdict.
² `BibtexValidator` may write **only** `manuscript/references.bib`.
³ Image agents may write **only** under `figures/candidates/` and `figures/generated/`.
⁴ `AssetManager` is the **only** agent that may write `manuscript/images/` — a single choke point where an
image becomes part of the book.

Note the orchestrator has no LLM grant at all. It schedules; it does not think about research.

---

## 8. Agent contracts

Contracts are Pydantic models; a malformed agent output fails the step rather than propagating. Abridged:

```python
class SourceRef:            # provenance, attached to everything
    source: Literal["arxiv","openalex","github","huggingface","blog",
                    "openreview","acl","hackernews","web","wikimedia","openverse"]
    source_id: str;  url: str;  title: str
    retrieved_at: datetime;  content_sha256: str

class Evidence:
    kind: Literal["scientific","practical","adoption","community",
                  "author_claim","independent_verification"]
    statement: str
    strength: Literal["strong","moderate","weak","anecdotal"]
    supports: str | None          # which claim it backs
    provenance: list[SourceRef]

class ResearchItem:         # output of every source agent
    id: str;  source: SourceRef
    title: str;  summary: str
    concepts: list[str];  methods: list[str];  benchmarks: list[BenchmarkResult]
    claims: list[Claim];   limitations: list[str]
    evidence: list[Evidence]
    lifecycle: Literal["DISCOVERED","SEEN","MONITOR","DISMISSED","INTEGRATED","SUPERSEDED"]

class ResearchCluster:      # output of ResearchSynthesizer
    id: str;  title: str;  concepts: list[str]
    papers / implementations / benchmarks / community_signals: list[ResearchItem]
    claims: list[Claim];  evidence: list[Evidence]
    contradictions: list[Contradiction];  limitations: list[str]
    maturity: Literal["speculative","emerging","consolidating","mature","superseded"]
    provenance: list[SourceRef]

class Verdict:              # output of BookVerdictAgent
    decision: Literal["IGNORE","MONITOR","ADD_REFERENCE","ADD_FOOTNOTE",
                      "EXTEND_SECTION","ADD_NEW_SECTION","REWRITE_SECTION",
                      "REPLACE_OBSOLETE_CONTENT"]
    rationale: str                       # evidence-backed prose, required
    targets: list[EditTarget]            # file + node label + what changes
    affected_content: list[str]          # existing node ids impacted
    claims_needing_evidence: list[str]
    citations_required: list[CitationNeed]
    figures_needed: list[VisualRequirement]
    scope: Literal["minimal","moderate","substantial"]
    confidence: float                    # reported, never the deciding factor

class DraftPatch:           # output of WriterAgent
    target_file: str;  unified_diff: str
    new_claims: list[Claim]
    citation_requirements: list[CitationNeed]
    visual_requirements: list[VisualRequirement]
    rationale: str
```

`Verdict.confidence` exists for reporting and triage only. The gate is `rationale` plus the evidence graph —
per the requirement that no simple numeric score decides a book change.

---

## 9. Data model

SQLite (`state/livingbook.db`), WAL mode. One file, transactional, resumable, trivially backed up.

```
runs            (run_id, started_at, ended_at, status, config_hash, trigger)
events          (id, ts, run_id, research_id, pipeline_id, agent, skill, tool,
                 state, status, duration_ms, artifact, extra_json)

-- Research Memory
research_items  (id, source, source_id, url, title, published_at, discovered_at,
                 last_seen_at, lifecycle, dismiss_reason, superseded_by,
                 payload_json, content_sha256)
evidence        (id, research_item_id, kind, statement, strength, supports,
                 provenance_json)
clusters        (id, title, concepts_json, maturity, state, created_at,
                 updated_at, payload_json)
cluster_members (cluster_id, research_item_id, role)

-- Book Knowledge Base
book_nodes      (id, kind, number, title, label, file, start_line, end_line,
                 parent_id, order_idx, word_count, content_sha256)
node_summaries  (node_id, summary, key_points_json, model, created_at)
concepts        (id, canonical_name, aliases_json, definition, first_node_id)
node_concepts   (node_id, concept_id, salience)
claims          (id, node_id, text, claim_type, needs_citation, status)
claim_citations (claim_id, bib_key, support_status, verified_at, verifier_note)
bib_entries     (bib_key, entry_type, title, authors, year, venue, doi, arxiv_id,
                 url, raw, validation_status, validated_at)
cite_edges      (node_id, bib_key, count)
xrefs           (src_node_id, label, dst_node_id, kind)
figures         (key, node_id, path, caption, alt_text, source_url, license,
                 license_url, attribution, status, sha256)
embeddings      (owner_type, owner_id, model, dim, vector BLOB)

-- Knowledge Graph
graph_edges     (src_type, src_id, rel, dst_type, dst_id, weight, provenance_json)

-- Pipeline / provenance / delivery
verdicts        (id, cluster_id, decision, rationale, payload_json, created_at,
                 approved_by, approved_at)
pipelines       (id, cluster_id, verdict_id, state, attempts, last_error,
                 state_data_json, created_at, updated_at)
artifacts       (id, pipeline_id, kind, path, sha256, parent_artifact_id,
                 meta_json, created_at)
versions        (version, created_at, commit_sha, branch, changelog_path,
                 pipeline_ids_json)
emails_sent     (id, pipeline_id, recipient, subject, sent_at, message_id)

-- Infrastructure
llm_cache       (hash, model, request_json, response_json, created_at)
keypool_stats   (key_hash, requests, successes, failures, tokens_in, tokens_out,
                 last_used, cooldown_until, disabled, disable_reason)
```

Keys are stored only as `key_hash` (SHA-256, truncated). The plaintext key never enters the database, a log
line, or an artifact.

## 10. Knowledge graph

```
    Concept ──related_to──▶ Concept
       │                        ▲
   explained_in              mentions
       ▼                        │
    Chapter ──contains──▶ Section ──contains──▶ Subsection
                                │
                            asserts
                                ▼
                             Claim ──cited_by──▶ Citation(bib_key)
                                │                      │
                          needs_evidence          resolves_to
                                │                      ▼
                                └────supported_by──▶ ResearchSource
                                                       ▲
                                              derived_from
                                                       │
                              ResearchCluster ──▶ ResearchItem
```

Edge types: `contains`, `mentions`, `related_to`, `asserts`, `cited_by`, `resolves_to`, `supported_by`,
`contradicts`, `supersedes`, `illustrated_by`, `derived_from`, `explained_in`, `needs_evidence`.

**The query that justifies the whole structure**: given a new `ResearchCluster` with concepts
`{speculative decoding, EAGLE, draft model}`, traverse `concept → mentions → node → asserts → claim →
cited_by → bib_key` and get back exactly the sections, claims and citations at risk — without reading the
manuscript. That is how a 153k-word book stays editable inside a bounded context.

## 11. State machine

```
DISCOVERED → COLLECTED → DEDUPLICATED → SYNTHESIZED → VERDICT_PENDING
   → VERDICT_APPROVED → DRAFTED → TECHNICAL_VERIFY → CITATION_AUDIT
   → CITATION_FIND → CITATION_VERIFY → EDITORIAL_VERIFY → BOOK_QA
   → APPROVED → GIT_PUBLISH → EMAIL → COMPLETED

Terminal off-ramps (all legal, all recorded):
   VERDICT_PENDING  → MONITORING   (verdict = MONITOR; re-evaluated on a cadence)
   VERDICT_PENDING  → REJECTED     (verdict = IGNORE)
   any              → FAILED       (retries exhausted)
   any              → NEEDS_HUMAN  (approval gate or repeated verification failure)

Backward edges (bounded by max_revisions, default 2):
   TECHNICAL_VERIFY → DRAFTED      (technical defect found)
   CITATION_VERIFY  → CITATION_FIND (citation rejected, try another source)
   CITATION_VERIFY  → DRAFTED      (claim unsupportable → claim must be softened or cut)
   EDITORIAL_VERIFY → DRAFTED      (pedagogical/style defect)
   BOOK_QA          → DRAFTED      (QA failure attributable to the patch)
```

Transitions are a table, not `if` statements: an illegal transition raises. Every transition persists
`(pipeline_id, state, state_data_json, updated_at)` inside one transaction with the artifact it produced, so
a crash at any point resumes at the last committed state instead of replaying the pipeline.

## 12. Research pipeline

```
Orchestrator fans out to 8 source agents (bounded concurrency)
        │
        ├─ each: discover → retrieve → extract → normalize → attach provenance
        │          (skip if source_id already in Research Memory and unchanged)
        ▼
  Research Memory  ← lifecycle: DISCOVERED / SEEN / MONITOR / DISMISSED
        │                       INTEGRATED / SUPERSEDED
        ▼
  ResearchSynthesizer  (no network access)
        ├─ dedupe: exact source_id, then DOI/arXiv identity, then embedding cosine ≥ τ
        ├─ cluster: agglomerative over concept embeddings
        ├─ link papers ↔ implementations ↔ benchmarks ↔ community threads
        ├─ classify evidence kind and strength per signal
        ├─ detect contradictions across sources
        └─ score maturity: speculative → emerging → consolidating → mature
        ▼
  ResearchCluster → BookVerdictAgent
```

**Signal vs evidence** is enforced structurally, not by prompt instruction: `Evidence.kind` is assigned by
the *source agent* that produced it, and a source agent can only emit the kinds its source can justify.
`CommunityAgent` cannot emit `scientific` — the field is constrained at the schema level. So a Hacker News
thread reaches the verdict agent as a `community` signal with `strength: anecdotal`, and the verdict agent
sees that, always.

Maturity is likewise a function of the evidence mix, not of recency: a single new paper with no independent
verification, no implementation and no benchmark lands at `speculative` and routes to `MONITOR`, not to a
book edit.

## 13. Citation pipeline

```
  Citation Auditor         claims in the patch + claims in affected existing sections
        │                  → CitationGap{claim, location, reason, preferred_source_type}
        ▼                    types: numerical, benchmark, historical, causal, SOTA,
  Citation Finder            definitional, architectural, attribution
        │                  searches for evidence supporting THE CLAIM, not the keywords;
        │                  ranks primary > original paper > official benchmark >
        ▼                  authoritative > reliable secondary; keeps provenance per candidate
  Citation Verifier        identity · authors · title · date · venue · DOI · arXiv id
        │                  claim support · experimental setup · benchmark · reported number
        │                  limitations · context
        │                  ── citation-laundering check ──
        ▼                  if the candidate only *reports* the claim from elsewhere,
  BibTeX Validator         it is rejected and the true primary source is pursued
        │                  title/author/year/venue/DOI/arXiv/URL, duplicates, malformed
        ▼
  manuscript/references.bib   (BibtexValidator is the only writer)
```

DOI verification routes by prefix — `10.48550/*` to DataCite, everything else to Crossref — because Crossref
404s on arXiv DOIs (measured). A citation that cannot be verified never reaches the manuscript; the claim it
was meant to support is sent back to the Writer to be softened or removed.

## 14. Visual pipeline

```
 Writer emits VisualRequirement{concept, purpose, expected_elements,
                                relationships, style}         (never an image)
        ▼
 VisualNeedDetector      is a figure actually warranted here? also scans existing
        │                \bookimage placeholders with no file
        ▼
 ImageSearchAgent        Wikimedia Commons → Openverse → web  (license-bearing first)
        ▼
 SemanticImageVerifier   does the image show expected_elements and relationships?
        │                multimodal check, not filename/keyword matching
        ▼
 LicenseChecker          license id + URL + attribution must resolve;
        │                unknown or non-redistributable → rejected
        ▼                                    ┌──── no acceptable candidate ────┐
 Image Description Refiner ──── search again ─┘  (max refine_rounds, default 2)  │
        ▼                                                                       ▼
 accepted image                                              DiagramGeneratorAgent
        │                                        declarative DiagramSpec → matplotlib
        │                                        (the intro-to-cv renderer, generalized)
        │                                        → Gemini image model as last resort
        └──────────────────┬────────────────────────────────┘
                           ▼
                      FigureQA        legible at print size · correct aspect · alt text
                           ▼          · caption matches content · no text-in-image errors
                      AssetManager    the only writer of manuscript/images/
                                      writes images/<key>.png so \bookimage resolves,
                                      records license + attribution + provenance
```

Relevance and license are both hard gates. A keyword match alone never places an image in the book.

## 15. QA pipeline

Never sends the whole book to one context. Three tiers:

**Deterministic** (no LLM, fast, runs on every patch): XeLaTeX build in a disposable in-place workspace ·
`\ref`/`\label` resolution · orphan and duplicate citations · BibTeX well-formedness · figure files exist ·
`\bookimage` keys resolve · alt text present · image paths valid · unbalanced braces/environments · URL
liveness.

**Semantic** (LLM, scoped to the patch + its graph neighbourhood): terminology consistency · claim support ·
citation appropriateness · cross-reference correctness · contradiction with neighbouring sections ·
duplicated explanation · chapter flow at the seams.

**Global** (LLM, over *summaries and graph projections*, never raw text): concept coverage · terminology
drift across all 15 chapters · claim-graph contradictions · citation-graph anomalies · TOC coherence. This
tier reads `node_summaries` + `graph_edges`, which is a few hundred KB regardless of manuscript size.

`CrossChapterConsistencyAgent` runs in the semantic tier over the union of nodes sharing a concept with the
patch — that is how "Chapter A says X, Chapter B says not-X" gets caught without a full-book read.

## 16. Git pipeline

```
branch  agent/research/<YYYY-MM-DD>-<topic-slug>
  → apply accepted patch + approved figures + validated references.bib
  → deterministic QA re-run on the branch (build + tests must pass)
  → generate changelog/<version>.md from the provenance chain
  → commit  (Conventional Commits, provenance trailers: cluster id, verdict, sources)
  → push to https://github.com/ciTy1504/intro2NLP_livingbook
  → open PR with the full evidence trail in the body
  → CI runs; merge per config (manual | auto_after_qa)
```

Guards: the remote is asserted to be `intro2NLP_livingbook` before any push, `main` is never committed to
directly, and the token is supplied as an HTTPS auth header, never written into the remote URL, a config
file, or a commit.

## 17. Email pipeline

Triggered **only** by a pipeline reaching `EMAIL` state, which is only reachable through `APPROVED`.
`MONITOR` and `IGNORE` verdicts terminate before it. No accepted manuscript change ⇒ no email. A separate,
explicitly opt-in digest covers "discovered but not integrated".

Content: research discovered · why it matters · what changed · chapters · sections · citation changes ·
figure changes · verification status per gate · branch/PR link · version. Sent via `smtplib` over Gmail with
an App Password (the EngChi credential model), with per-recipient error isolation and a persisted
`emails_sent` record so a retry never double-sends.

## 18. Gemini provider architecture

```
        Agent  ──►  AgentContext (permission check, logging)
                        │
                        ▼
              LLMProvider  (abstract: generate / generate_structured / embed)
                        │
                        ▼
                 GeminiAdapter
      ┌─────────┬────────────┬──────────┬───────────┬──────────┬─────────┐
   KeyPool  RateLimiter  RetryPolicy  ModelSelector  Usage    Response
  rotation   global sem   classify &   role → chain  Tracker   Cache
  cooldown   + interval   backoff                    tokens    (sqlite)
   stats                                             per key
                        │
                        ▼
                   Gemini API
```

**ModelSelector resolves a role to an ordered chain**, which the audit showed is mandatory rather than nice
to have:

```yaml
fast:      [gemini-3.5-flash-lite, gemini-3-flash-preview, gemini-3.1-flash-lite]
balanced:  [gemini-3.6-flash, gemini-3-flash-preview, gemini-3.5-flash-lite]
deep:      [gemini-3.6-flash, gemini-3.1-pro-preview, gemini-3-flash-preview]
embedding: [gemini-embedding-001, gemini-embedding-2]
image:     [gemini-3.1-flash-image, gemini-3-pro-image]
```

Error classification drives behaviour: `429`/`403 quota` → cool the key down and retry on a different key ·
`503`/timeout → the *model* is saturated, advance the chain · `400 API_KEY_INVALID` → permanently disable
that key · `400` other → non-retryable, fail the step · `5xx` other → exponential backoff with jitter.

The response cache is keyed on `sha256(model, prompt, schema, generation_config)`. Beyond saving quota it is
what makes a mid-pipeline crash cheap to resume: re-running a step replays cached completions instead of
re-billing them.

Adding a provider means implementing `LLMProvider` and registering it. No agent, skill or prompt changes.

## 19. Failure & retry architecture

| Layer | Failure | Response |
|---|---|---|
| Key | 429 / 403 quota | cooldown (90s, configurable), pick a different key, same model |
| Key | 400 `API_KEY_INVALID` | permanently disable, record reason, continue |
| Model | 503 / timeout | advance the model fallback chain |
| Tool | network/5xx | 3 retries, exponential backoff + jitter |
| Tool | source unavailable | **degrade**: that source contributes nothing, the cycle continues |
| Skill | schema validation failure | 1 repair attempt with the validation error fed back, then fail |
| Agent | per its `failure_policy` | `retry` · `degrade` · `fail_pipeline` · `escalate_to_human` |
| Verification | defect found | backward transition to `DRAFTED`, bounded by `max_revisions` |
| Pipeline | retries exhausted | `FAILED`, full context persisted, other pipelines unaffected |
| Process | crash | resume from last committed state; no pipeline restarts from the beginning |

One source being down never fails a cycle, and one pipeline failing never touches its siblings.

## 20. Scheduling architecture

```
continuous ──┬── discovery      every  6h   all 8 source agents
             ├── synthesis      every 12h   over undigested research memory
             ├── verdict        every 12h   over new clusters
             ├── monitor sweep  every  7d   re-evaluate MONITOR items against new evidence
             ├── pipeline tick  every 15m   advance every non-terminal pipeline one step
             ├── kb reindex     on change   manuscript file hash differs
             └── full book QA   weekly      global tier, independent of any patch
```

The scheduler is a persisted cron-like table, not `sleep()`: due times survive restarts, a missed window
fires on next start, and every job is idempotent (guarded by `last_run_at` + a per-job lock). Bounded global
concurrency prevents the 8 source agents from saturating the key pool.

Approval gates (`config/config.yaml`): `require_human_approval_for` accepts a list of verdict decisions —
default `[REWRITE_SECTION, REPLACE_OBSOLETE_CONTENT]` — which park the pipeline in `NEEDS_HUMAN` with a
rendered review file in `reviews/` instead of proceeding. `auto_merge: false` by default.

---

## Repository layout

```
intro2NLP_livingbook/
├── manuscript/             the book (baseline copy; the only publishable content)
├── livingbook/             the system
│   ├── config.py  obs.py  cli.py
│   ├── llm/                provider · keypool · gemini · ratelimit · cache · errors
│   ├── tools/              web scholarly code community filesystem book kb visual git email llm
│   ├── skills/             research book citation visual verification publishing
│   ├── agents/             research verdict writer citation verification visual delivery
│   ├── knowledge/          latex parser · indexer · graph · retrieval · bib
│   ├── research/           memory · models
│   ├── state/              machine · store · artifacts
│   └── orchestrator/       orchestrator · scheduler · pipeline
├── config/                 config.yaml · agents.yaml · sources.yaml · style_guide.md
├── secrets/                gitignored — keypool + .env
├── knowledge/              seeds/ · exports/
├── research/               inbox/ processed/ monitoring/ rejected/
├── figures/                requirements/ candidates/ approved/ generated/
├── reviews/  changelog/  logs/  state/  tests/  scripts/  .github/workflows/
└── AUDIT.md  ARCHITECTURE.md  README.md  requirements.txt
```
