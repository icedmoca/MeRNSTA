"""
Migration 004: Fix memory system tables
Ensures facts, contradictions, and episodes tables have proper structure.
"""

import sqlite3
import logging
from config.environment import get_settings

def run_migration():
    """
    Ensure database tables are properly structured for MeRNSTA memory system.
    Creates facts, contradictions, and episodes tables with correct columns.
    """
    settings = get_settings()
    
    # Get database path from settings
    db_path = getattr(settings, 'database_url', 'sqlite:///memory.db')
    if db_path.startswith('sqlite:///'):
        db_path = db_path[10:]  # Remove sqlite:/// prefix
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("🔧 Creating/fixing facts table...")
        
        # Create facts table with all required columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                user_profile_id TEXT,
                source_message_id INTEGER,
                frequency INTEGER DEFAULT 1,
                contradiction_score REAL DEFAULT 0.0,
                volatility_score REAL DEFAULT 0.0,
                last_reinforced TEXT DEFAULT CURRENT_TIMESTAMP,
                episode_id INTEGER DEFAULT 1,
                emotion_score REAL DEFAULT NULL,
                context TEXT DEFAULT NULL,
                media_type TEXT DEFAULT 'text',
                media_data BLOB,
                session_id TEXT,
                subject_cluster_id INTEGER DEFAULT NULL,
                embedding TEXT DEFAULT NULL
            )
        """)
        
        print("🔧 Creating/fixing contradictions table...")
        
        # Create contradictions table for tracking fact conflicts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER NOT NULL,
                fact_b_id INTEGER NOT NULL,
                fact_a_text TEXT NOT NULL,
                fact_b_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                resolution_notes TEXT DEFAULT NULL,
                FOREIGN KEY (fact_a_id) REFERENCES facts(id) ON DELETE CASCADE,
                FOREIGN KEY (fact_b_id) REFERENCES facts(id) ON DELETE CASCADE
            )
        """)
        
        print("🔧 Creating/fixing episodes table...")
        
        # Create episodes table for conversation summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_profile_id TEXT,
                start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                end_time TEXT DEFAULT NULL,
                fact_count INTEGER DEFAULT 0,
                summary TEXT,
                personality_profile TEXT DEFAULT 'neutral',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("🔧 Adding missing columns to existing tables...")
        
        # Add missing columns to facts table if they don't exist
        missing_columns = [
            ("user_profile_id", "TEXT"),
            ("session_id", "TEXT"),
            ("last_reinforced", "TEXT DEFAULT CURRENT_TIMESTAMP"),
            ("episode_id", "INTEGER DEFAULT 1"),
            ("emotion_score", "REAL DEFAULT NULL"),
            ("context", "TEXT DEFAULT NULL"),
            ("media_type", "TEXT DEFAULT 'text'"),
            ("media_data", "BLOB"),
            ("contradiction_score", "REAL DEFAULT 0.0"),
            ("volatility_score", "REAL DEFAULT 0.0"),
            ("subject_cluster_id", "INTEGER DEFAULT NULL"),
            ("embedding", "TEXT DEFAULT NULL")
        ]
        
        for column_name, column_type in missing_columns:
            try:
                cursor.execute(f"ALTER TABLE facts ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added {column_name} column to facts table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"   ✅ Column {column_name} already exists")
                else:
                    print(f"   ❌ Failed to add {column_name}: {e}")
        
        # Create indexes for better performance
        print("🔧 Creating database indexes...")
        
        indexes = [
            ("idx_facts_user_profile", "facts", "user_profile_id"),
            ("idx_facts_session", "facts", "session_id"),
            ("idx_facts_timestamp", "facts", "timestamp"),
            ("idx_facts_subject", "facts", "subject"),
            ("idx_contradictions_fact_a", "contradictions", "fact_a_id"),
            ("idx_contradictions_fact_b", "contradictions", "fact_b_id"),
            ("idx_episodes_session", "episodes", "session_id"),
            ("idx_episodes_user", "episodes", "user_profile_id")
        ]
        
        for index_name, table_name, column_name in indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})")
                print(f"   ✅ Created index {index_name}")
            except sqlite3.OperationalError as e:
                print(f"   ❌ Failed to create index {index_name}: {e}")
        
        # Verify table structure
        print("🔧 Verifying table structure...")
        
        tables_to_check = ["facts", "contradictions", "episodes"]
        for table in tables_to_check:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print(f"   ✅ Table '{table}' has {len(columns)} columns")
        
        conn.commit()
        print("✅ Migration 004 completed successfully!")
        
        return True
        
    except Exception as e:
        logging.error(f"Migration 004 failed: {e}")
        print(f"❌ Migration 004 failed: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_migration() 