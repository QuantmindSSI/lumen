-- A1 + D3 + D16: Canonical SQLite Palace Schema
-- Input wire: SQLite 3.45+ (WAL mode)
-- Output wire: A2 (FTS5 bridge), A3 (vector table), A5 (provenance),
--              C5 (context assembly), D3 (feedback_log), D16 (user_profile)

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS room (
    room_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- e.g. "preferences", "travel:japan"
    created_at  INTEGER DEFAULT (unixepoch()),
    last_entry_at INTEGER,
    locus_count INTEGER DEFAULT 0,
    room_type   TEXT CHECK(room_type IN ('domain','project','person','ephemeral')),
    topological_order REAL DEFAULT 0.0         -- for palace map rendering
);

CREATE TABLE IF NOT EXISTS locus (
    locus_id    INTEGER PRIMARY KEY,
    room_id     INTEGER NOT NULL REFERENCES room(room_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    vector_mean BLOB,                          -- cached centroid of resident chunks (optical degrade)
    created_at  INTEGER DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0,
    last_access_at INTEGER,
    UNIQUE(room_id, name)
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id        INTEGER PRIMARY KEY,
    locus_id        INTEGER REFERENCES locus(locus_id) ON DELETE SET NULL,
    room_id         INTEGER NOT NULL REFERENCES room(room_id) ON DELETE CASCADE,
    content         TEXT NOT NULL,             -- raw text / memory payload
    content_hash    TEXT NOT NULL,             -- SHA-256 of content for dedup
    created_at      INTEGER DEFAULT (unixepoch()),
    valid_from      INTEGER DEFAULT (unixepoch()),  -- bi-temporal: Engram pattern
    valid_to        INTEGER,                        -- NULL = still valid
    superseded_by   INTEGER REFERENCES chunk(chunk_id),
    resolution      TEXT DEFAULT 'FP32' CHECK(resolution IN ('FP32','FP16','INT8','BINARY','RELEASED')),
    vm_score        REAL DEFAULT 0.5,          -- V(m) scalar (A9)
    vm_factors      BLOB,                      -- JSON: {goal_relevance:0.8, ...}
    access_count    INTEGER DEFAULT 0,
    last_access_at  INTEGER,
    optical_level   INTEGER DEFAULT 0,         -- 0=full, 1=degraded, 2=forgotten
    provenance_root INTEGER REFERENCES provenance(provenance_id)
);

CREATE TABLE IF NOT EXISTS provenance (
    provenance_id   INTEGER PRIMARY KEY,
    chunk_id        INTEGER NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    source_type     TEXT CHECK(source_type IN ('user_input','agent_reasoning','consolidation','import','p2p_share')),
    source_ref      TEXT,                      -- e.g. session_id, turn_number
    confidence      REAL DEFAULT 1.0,
    extraction_method TEXT,
    parent_provenance INTEGER REFERENCES provenance(provenance_id)
);

-- D3: Feedback Log Schema
CREATE TABLE IF NOT EXISTS feedback_log (
    feedback_id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunk(chunk_id),
    user_id TEXT DEFAULT 'default',
    positive INTEGER NOT NULL CHECK(positive IN (0,1)),
    feedback_type TEXT DEFAULT 'implicit' CHECK(feedback_type IN ('implicit','explicit','repair')),
    session_id TEXT,
    turn_id INTEGER,
    created_at INTEGER DEFAULT (unixepoch())
);

-- D18: Audit Log
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor TEXT,
    resource_type TEXT,
    resource_id INTEGER,
    action TEXT,
    metadata_json TEXT,
    client_ip TEXT,
    request_id TEXT,
    created_at INTEGER DEFAULT (unixepoch())
);

-- D16: User Profile Storage
CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY DEFAULT 'default',
    goals_json TEXT DEFAULT '[]',          -- list of goal strings
    values_json TEXT DEFAULT '[]',         -- list of value strings
    goal_embeddings BLOB,                  -- np.ndarray of shape (n_goals, 384)
    value_embeddings BLOB,
    vm_weights_json TEXT,                  -- serialized 7-factor weights
    ebbinghaus_half_life_days REAL DEFAULT 7.0
);

-- C4: Goal Tree Persistence
CREATE TABLE IF NOT EXISTS goals (
    goal_id     INTEGER PRIMARY KEY,
    user_id     TEXT DEFAULT 'default',
    parent_id   INTEGER REFERENCES goals(goal_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    is_active   INTEGER DEFAULT 0 CHECK(is_active IN (0,1)),
    created_at  INTEGER DEFAULT (unixepoch()),
    updated_at  INTEGER DEFAULT (unixepoch()),
    UNIQUE(user_id, name)
);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id);

-- C3: Epistemic State Persistence
CREATE TABLE IF NOT EXISTS epistemic_state (
    user_id             TEXT PRIMARY KEY DEFAULT 'default',
    known_facts_json    TEXT DEFAULT '[]',
    assumed_gaps_json   TEXT DEFAULT '[]',
    established_truths_json TEXT DEFAULT '[]',
    updated_at          INTEGER DEFAULT (unixepoch())
);

-- Migration tracking
CREATE TABLE IF NOT EXISTS event_buffer_meta (
    meta_id INTEGER PRIMARY KEY,
    last_consolidation_at INTEGER,
    last_decay_run_at INTEGER,
    version INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunk_locus ON chunk(locus_id, vm_score DESC);
CREATE INDEX IF NOT EXISTS idx_chunk_room ON chunk(room_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunk_valid ON chunk(valid_from, valid_to) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_chunk_hash ON chunk(content_hash);
CREATE INDEX IF NOT EXISTS idx_feedback_chunk ON feedback_log(chunk_id, positive);

-- Full-text search bridge (A2)
-- External content table mapping to chunk.chunk_id
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    content, content_hash UNINDEXED,
    content='chunk',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);
