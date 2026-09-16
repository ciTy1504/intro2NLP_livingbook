-- Living Book persistent state.
-- One SQLite file: transactional, resumable, trivially backed up.
-- Every state transition commits together with the artifact it produced, so a crash
-- resumes at the last committed state instead of replaying the pipeline.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─────────────────────────────── runs & events ───────────────────────────────

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    status      TEXT NOT NULL DEFAULT 'running',   -- running|completed|failed|aborted
    trigger     TEXT,                              -- manual|scheduler|resume
    config_hash TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    level       TEXT,
    run_id      TEXT,
    research_id TEXT,
    pipeline_id TEXT,
    agent       TEXT,
    skill       TEXT,
    tool        TEXT,
    state       TEXT,
    status      TEXT,
    duration_ms INTEGER,
    artifact    TEXT,
    message     TEXT,
    extra_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run      ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_research ON events(research_id);
CREATE INDEX IF NOT EXISTS idx_events_pipeline ON events(pipeline_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_agent    ON events(agent, ts);

-- ───────────────────────────── research memory ──────────────────────────────

CREATE TABLE IF NOT EXISTS research_items (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL,        -- arxiv|openalex|github|huggingface|blog|...
    source_id      TEXT NOT NULL,        -- stable id within that source
    url            TEXT,
    title          TEXT NOT NULL,
    summary        TEXT,
    published_at   TEXT,
    discovered_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    lifecycle      TEXT NOT NULL DEFAULT 'DISCOVERED',
                   -- DISCOVERED|SEEN|MONITOR|DISMISSED|INTEGRATED|SUPERSEDED
    lifecycle_at   TEXT,
    dismiss_reason TEXT,
    superseded_by  TEXT,
    relevance      REAL,
    concepts_json  TEXT,
    payload_json   TEXT,                 -- the full normalised ResearchItem
    content_sha256 TEXT,
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_research_lifecycle ON research_items(lifecycle, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_research_source    ON research_items(source, published_at);

CREATE TABLE IF NOT EXISTS evidence (
    id               TEXT PRIMARY KEY,
    research_item_id TEXT NOT NULL REFERENCES research_items(id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,   -- scientific|practical|adoption|community|
                                      -- author_claim|independent_verification
    statement        TEXT NOT NULL,
    strength         TEXT NOT NULL,   -- strong|moderate|weak|anecdotal
    supports         TEXT,
    provenance_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_item ON evidence(research_item_id);
CREATE INDEX IF NOT EXISTS idx_evidence_kind ON evidence(kind, strength);

CREATE TABLE IF NOT EXISTS clusters (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    concepts_json TEXT,
    maturity      TEXT,               -- speculative|emerging|consolidating|mature|superseded
    state         TEXT NOT NULL DEFAULT 'SYNTHESIZED',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    payload_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_clusters_state ON clusters(state, updated_at);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id       TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    research_item_id TEXT NOT NULL REFERENCES research_items(id) ON DELETE CASCADE,
    role             TEXT,            -- paper|implementation|benchmark|community|blog
    PRIMARY KEY (cluster_id, research_item_id)
);

-- ──────────────────────────── book knowledge base ───────────────────────────

CREATE TABLE IF NOT EXISTS book_nodes (
    id             TEXT PRIMARY KEY,   -- stable: file + label or file + ordinal
    kind           TEXT NOT NULL,      -- part|chapter|section|subsection|subsubsection
    number         TEXT,               -- rendered numbering, e.g. "5.2.1"
    title          TEXT NOT NULL,
    label          TEXT,               -- \label{...} if present
    file           TEXT NOT NULL,      -- path relative to manuscript/
    start_line     INTEGER NOT NULL,
    end_line       INTEGER NOT NULL,
    parent_id      TEXT REFERENCES book_nodes(id) ON DELETE CASCADE,
    order_idx      INTEGER NOT NULL,   -- reading order across the whole book
    word_count     INTEGER DEFAULT 0,
    content_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_file   ON book_nodes(file);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON book_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_order  ON book_nodes(order_idx);
CREATE INDEX IF NOT EXISTS idx_nodes_label  ON book_nodes(label);

CREATE TABLE IF NOT EXISTS node_summaries (
    node_id         TEXT PRIMARY KEY REFERENCES book_nodes(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    key_points_json TEXT,
    prerequisites_json TEXT,
    level           TEXT,             -- introductory|intermediate|advanced
    model           TEXT,
    source_sha256   TEXT,             -- content hash the summary was made from
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    id            TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    aliases_json  TEXT,
    definition    TEXT,
    first_node_id TEXT REFERENCES book_nodes(id) ON DELETE SET NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS node_concepts (
    node_id    TEXT NOT NULL REFERENCES book_nodes(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    salience   REAL DEFAULT 0.5,
    PRIMARY KEY (node_id, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_node_concepts_concept ON node_concepts(concept_id);

CREATE TABLE IF NOT EXISTS claims (
    id             TEXT PRIMARY KEY,
    node_id        TEXT NOT NULL REFERENCES book_nodes(id) ON DELETE CASCADE,
    text           TEXT NOT NULL,
    claim_type     TEXT,     -- numerical|benchmark|historical|causal|sota|
                             -- definitional|architectural|attribution
    needs_citation INTEGER DEFAULT 0,
    status         TEXT DEFAULT 'unverified',  -- unverified|supported|unsupported|disputed
    created_at     TEXT NOT NULL,
    source_sha256  TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_node   ON claims(node_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status, needs_citation);

CREATE TABLE IF NOT EXISTS claim_citations (
    claim_id       TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    bib_key        TEXT NOT NULL,
    support_status TEXT,     -- supports|partial|does_not_support|laundered|unverified
    verified_at    TEXT,
    verifier_note  TEXT,
    PRIMARY KEY (claim_id, bib_key)
);

CREATE TABLE IF NOT EXISTS bib_entries (
    bib_key           TEXT PRIMARY KEY,
    entry_type        TEXT,
    title             TEXT,
    authors           TEXT,
    year              INTEGER,
    venue             TEXT,
    doi               TEXT,
    arxiv_id          TEXT,
    url               TEXT,
    raw               TEXT NOT NULL,
    validation_status TEXT DEFAULT 'unvalidated',  -- unvalidated|valid|suspect|invalid
    validation_note   TEXT,
    validated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_bib_doi   ON bib_entries(doi);
CREATE INDEX IF NOT EXISTS idx_bib_arxiv ON bib_entries(arxiv_id);

CREATE TABLE IF NOT EXISTS cite_edges (
    node_id TEXT NOT NULL REFERENCES book_nodes(id) ON DELETE CASCADE,
    bib_key TEXT NOT NULL,
    count   INTEGER DEFAULT 1,
    PRIMARY KEY (node_id, bib_key)
);
CREATE INDEX IF NOT EXISTS idx_cite_edges_key ON cite_edges(bib_key);

CREATE TABLE IF NOT EXISTS xrefs (
    src_node_id TEXT NOT NULL REFERENCES book_nodes(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    dst_node_id TEXT,
    kind        TEXT,          -- ref|eqref|autoref
    PRIMARY KEY (src_node_id, label, kind)
);

CREATE TABLE IF NOT EXISTS figures (
    key          TEXT PRIMARY KEY,   -- the \bookimage / includegraphics key
    node_id      TEXT REFERENCES book_nodes(id) ON DELETE SET NULL,
    path         TEXT,               -- relative to manuscript/
    requirement  TEXT,               -- the \bookimage description, i.e. the spec
    caption      TEXT,
    alt_text     TEXT,
    source_url   TEXT,
    license      TEXT,
    license_url  TEXT,
    attribution  TEXT,
    status       TEXT DEFAULT 'missing',  -- missing|requested|candidate|approved|placed|rejected
    sha256       TEXT,
    updated_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_figures_status ON figures(status);

CREATE TABLE IF NOT EXISTS embeddings (
    owner_type TEXT NOT NULL,        -- node|concept|claim|research_item|cluster
    owner_id   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_type, owner_id, model)
);

-- ───────────────────────────── knowledge graph ──────────────────────────────

CREATE TABLE IF NOT EXISTS graph_edges (
    src_type        TEXT NOT NULL,
    src_id          TEXT NOT NULL,
    rel             TEXT NOT NULL,   -- contains|mentions|related_to|asserts|cited_by|
                                     -- resolves_to|supported_by|contradicts|supersedes|
                                     -- illustrated_by|derived_from|explained_in|needs_evidence
    dst_type        TEXT NOT NULL,
    dst_id          TEXT NOT NULL,
    weight          REAL DEFAULT 1.0,
    provenance_json TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (src_type, src_id, rel, dst_type, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_src ON graph_edges(src_type, src_id, rel);
CREATE INDEX IF NOT EXISTS idx_graph_dst ON graph_edges(dst_type, dst_id, rel);
CREATE INDEX IF NOT EXISTS idx_graph_rel ON graph_edges(rel);

-- ───────────────────────── verdicts, pipelines, artifacts ───────────────────

CREATE TABLE IF NOT EXISTS verdicts (
    id           TEXT PRIMARY KEY,
    cluster_id   TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    decision     TEXT NOT NULL,
    rationale    TEXT NOT NULL,
    scope        TEXT,
    confidence   REAL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    approved_by  TEXT,
    approved_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_cluster ON verdicts(cluster_id);

CREATE TABLE IF NOT EXISTS pipelines (
    id              TEXT PRIMARY KEY,
    cluster_id      TEXT REFERENCES clusters(id) ON DELETE CASCADE,
    verdict_id      TEXT REFERENCES verdicts(id) ON DELETE SET NULL,
    state           TEXT NOT NULL,
    previous_state  TEXT,
    attempts        INTEGER DEFAULT 0,
    revisions       INTEGER DEFAULT 0,
    last_error      TEXT,
    state_data_json TEXT,
    run_id          TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    next_attempt_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipelines_state ON pipelines(state, updated_at);

CREATE TABLE IF NOT EXISTS pipeline_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(id) ON DELETE CASCADE,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    at          TEXT NOT NULL,
    note        TEXT,
    artifact_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_transitions_pipeline ON pipeline_transitions(pipeline_id, at);

CREATE TABLE IF NOT EXISTS artifacts (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT REFERENCES pipelines(id) ON DELETE CASCADE,
    kind               TEXT NOT NULL,  -- cluster|verdict|draft_patch|citation_gap|
                                       -- citation_candidate|verification_report|
                                       -- qa_report|figure|changelog|email
    path               TEXT,
    sha256             TEXT,
    parent_artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
    meta_json          TEXT,
    created_at         TEXT NOT NULL,
    created_by         TEXT             -- agent name
);
CREATE INDEX IF NOT EXISTS idx_artifacts_pipeline ON artifacts(pipeline_id, kind);
CREATE INDEX IF NOT EXISTS idx_artifacts_parent   ON artifacts(parent_artifact_id);

-- ───────────────────────────── delivery & versions ──────────────────────────

CREATE TABLE IF NOT EXISTS versions (
    version          TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    commit_sha       TEXT,
    branch           TEXT,
    pull_request_url TEXT,
    changelog_path   TEXT,
    pipeline_ids_json TEXT,
    summary          TEXT
);

CREATE TABLE IF NOT EXISTS emails_sent (
    id          TEXT PRIMARY KEY,
    pipeline_id TEXT REFERENCES pipelines(id) ON DELETE SET NULL,
    recipient   TEXT NOT NULL,
    subject     TEXT NOT NULL,
    sent_at     TEXT NOT NULL,
    message_id  TEXT,
    status      TEXT DEFAULT 'sent'
);
CREATE INDEX IF NOT EXISTS idx_emails_pipeline ON emails_sent(pipeline_id);

-- ──────────────────────────────── scheduler ─────────────────────────────────

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    name         TEXT PRIMARY KEY,
    interval_seconds INTEGER NOT NULL,
    last_run_at  TEXT,
    next_run_at  TEXT,
    last_status  TEXT,
    last_error   TEXT,
    running      INTEGER DEFAULT 0,
    lock_owner   TEXT,
    lock_at      TEXT,
    runs         INTEGER DEFAULT 0
);

-- ─────────────────────────── schema bookkeeping ─────────────────────────────

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1');
