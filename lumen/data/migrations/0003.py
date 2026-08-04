def run(conn):
    """Add encrypted flag to chunk for field-level encryption tracking."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunk)").fetchall()}
    if "encrypted" not in cols:
        conn.execute("ALTER TABLE chunk ADD COLUMN encrypted INTEGER DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_encrypted ON chunk(encrypted)")
    conn.commit()
