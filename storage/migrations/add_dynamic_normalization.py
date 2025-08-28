from storage.db_utils import get_connection_pool

with get_connection_pool().get_connection() as conn:
    try:
        conn.execute("ALTER TABLE facts ADD COLUMN subject_cluster_id TEXT")
    except Exception:
        pass  # Column may already exist
    conn.execute("UPDATE facts SET subject_cluster_id = subject WHERE subject_cluster_id IS NULL")
    conn.commit() 