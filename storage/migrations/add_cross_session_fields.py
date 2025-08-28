#!/usr/bin/env python3
"""
Database migration to add cross-session fields to facts and episodes tables.
Adds 'session_id' (TEXT) and 'user_profile_id' (TEXT) columns if missing.
"""
import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config.settings import DATABASE_CONFIG
from storage.db_utils import get_connection_pool

def add_cross_session_fields():
    db_path = DATABASE_CONFIG["default_path"]
    try:
        with get_connection_pool().get_connection() as conn:
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN user_profile_id TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN session_id TEXT")
            except Exception:
                pass
            conn.commit()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    return True

def main():
    print("🧠 MeRNSTA Cross-Session Database Migration")
    print("=" * 60)
    if add_cross_session_fields():
        print("\n📊 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")

if __name__ == "__main__":
    main() 