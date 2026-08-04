def run(conn):
    conn.execute("ALTER TABLE room ADD COLUMN tenant_id TEXT DEFAULT 'default'")
    conn.execute("ALTER TABLE locus ADD COLUMN tenant_id TEXT DEFAULT 'default'")
    conn.execute("ALTER TABLE chunk ADD COLUMN tenant_id TEXT DEFAULT 'default'")
    conn.execute("ALTER TABLE provenance ADD COLUMN tenant_id TEXT DEFAULT 'default'")
    conn.execute("ALTER TABLE feedback_log ADD COLUMN tenant_id TEXT DEFAULT 'default'")
    conn.execute("""
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
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_tenant ON chunk(tenant_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_room_tenant ON room(tenant_id, name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback_log(tenant_id)")
    conn.commit()
