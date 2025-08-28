#!/usr/bin/env python3
"""
Database migration to add multi-modal fields to facts table.
Ensures 'media_type' and 'media_data' columns exist for multi-modal memory.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.settings import DATABASE_CONFIG
from storage.db_utils import get_conn

def add_multimodal_fields():
    """
    Add 'media_type' (TEXT) and 'media_data' (BLOB) columns to facts table if missing.
    """
    db_path = DATABASE_CONFIG["default_path"]

    try:
        with get_conn(db_path) as conn:
            # Check if fields already exist
            cursor = conn.execute("PRAGMA table_info(facts)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            new_fields = [
                ("media_type", "TEXT DEFAULT 'text'"),
                ("media_data", "BLOB")
            ]

            added_fields = []
            for field_name, field_type in new_fields:
                if field_name not in existing_columns:
                    try:
                        conn.execute(
                            f"ALTER TABLE facts ADD COLUMN {field_name} {field_type}"
                        )
                        added_fields.append(field_name)
                        print(f"✅ Added column: {field_name}")
                    except Exception as e:
                        print(f"❌ Error adding {field_name}: {e}")
                else:
                    print(f"ℹ️ Column {field_name} already exists")

            if added_fields:
                print(f"\n🎉 Successfully added {len(added_fields)} multi-modal fields to facts table")
            else:
                print("\nℹ️ All multi-modal fields already exist in the database")

            conn.commit()

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

    return True

def main():
    print("🧠 MeRNSTA Multi-Modal Database Migration")
    print("=" * 60)
    if add_multimodal_fields():
        print("\n📊 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")

if __name__ == "__main__":
    main() 