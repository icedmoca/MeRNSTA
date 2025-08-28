#!/usr/bin/env python3
"""
Database cleanup utilities for tests.
Provides safe cleanup functions that handle foreign key constraints.
"""

import sqlite3
import logging
from typing import Optional
import os
import glob

logger = logging.getLogger(__name__)


def safe_cleanup_database(db_path: str) -> None:
    """
    Safely clean up all tables in the database by disabling foreign key constraints
    and deleting rows in the proper order.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Disable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # Delete rows in proper order (child tables first)
        tables_to_clean = [
            "contradictions",      # References facts
            "fact_history",        # References facts  
            "drift_events",        # References facts
            "facts",               # References memory
            "clusters",            # Independent
            "episodes",            # Independent
            "memory",              # Independent
            "summaries"            # Independent
        ]
        
        for table in tables_to_clean:
            try:
                cursor.execute(f"DELETE FROM {table}")
                logger.debug(f"Cleaned table: {table}")
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    logger.debug(f"Table {table} doesn't exist, skipping")
                else:
                    logger.warning(f"Error cleaning table {table}: {e}")
        
        # Re-enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Successfully cleaned database: {db_path}")
        
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")
        raise


def cleanup_memory_log(memory_log) -> None:
    """
    Clean up a MemoryLog instance's database safely.
    
    Args:
        memory_log: MemoryLog instance to clean
    """
    if hasattr(memory_log, 'db_path'):
        safe_cleanup_database(memory_log.db_path)
    else:
        logger.warning("MemoryLog has no db_path attribute")


def reset_database_for_test(db_path: str):
    """
    Fully reset the database:
    🔹 Close and clear connection pool
    🔹 Delete DB and WAL/SHM files
    🔹 Recreate full schema with MemoryLog
    """
    import os
    from storage.memory_log import MemoryLog
    from storage.db_utils import get_connection_pool, ConnectionConfig
    import storage.db_utils  # Needed to reset _connection_pool

    # Close and clear the connection pool
    try:
        pool = get_connection_pool(ConnectionConfig(db_path=db_path))
        if hasattr(pool, 'close_all_connections'):
            pool.close_all_connections()
        # 💡 Clear global pool to prevent reuse of stale pool
        storage.db_utils._connection_pool = None
    except Exception:
        pass

    # Delete DB + WAL/SHM files
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"{db_path}{ext}")
        except FileNotFoundError:
            pass

    # Recreate full schema
    memory_log = MemoryLog(db_path)
    memory_log.init_database()
    del memory_log  # Ensure no lingering connections 


def safe_cleanup_database():
    """Delete all .db files in the project root and subdirectories."""
    db_files = glob.glob('*.db') + glob.glob('**/*.db', recursive=True)
    for db_file in db_files:
        try:
            os.remove(db_file)
            print(f"Deleted: {db_file}")
        except Exception as e:
            print(f"Failed to delete {db_file}: {e}") 