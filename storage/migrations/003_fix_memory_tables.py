#!/usr/bin/env python3
"""
Database migration to fix MeRNSTA memory system tables.
Ensures all required tables exist with proper schema for:
- facts table with user_profile_id and session_id
- contradictions table for contradiction tracking
- episodes table for episodic memory
- summaries table for conversation summaries
"""

import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

def get_db_connection():
    """Get database connection using environment settings."""
    try:
        from config.environment import get_settings
        settings = get_settings()
        db_url = settings.database_url
        
        if db_url.startswith("sqlite"):
            import sqlite3
            db_path = db_url.replace("sqlite:///", "")
            return sqlite3.connect(db_path)
        else:
            # For PostgreSQL support in future
            import psycopg2
            return psycopg2.connect(db_url)
    except Exception as e:
        # Fallback to config.yaml settings
        try:
            from config.settings import DATABASE_CONFIG
            from storage.db_utils import get_connection_pool
            return get_connection_pool().get_connection()
        except:
            # Final fallback to simple SQLite
            import sqlite3
            return sqlite3.connect("memory.db")

def fix_memory_tables():
    """Create or update all required memory tables."""
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("🔧 Fixing facts table...")
        
        # Create facts table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source_message_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                frequency INTEGER DEFAULT 1,
                contradiction_score REAL DEFAULT 0.0,
                volatility_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 1.0,
                last_reinforced TEXT DEFAULT CURRENT_TIMESTAMP,
                episode_id INTEGER DEFAULT 1,
                emotion_score REAL DEFAULT NULL,
                context TEXT DEFAULT NULL,
                media_type TEXT DEFAULT 'text',
                media_data BLOB,
                session_id TEXT,
                user_profile_id TEXT
            )
        """)
        
        # Add missing columns to facts table
        facts_columns = [
            ("user_profile_id", "TEXT"),
            ("session_id", "TEXT"),
            ("last_reinforced", "TEXT DEFAULT CURRENT_TIMESTAMP"),
            ("episode_id", "INTEGER DEFAULT 1"),
            ("emotion_score", "REAL DEFAULT NULL"),
            ("context", "TEXT DEFAULT NULL"),
            ("media_type", "TEXT DEFAULT 'text'"),
            ("media_data", "BLOB"),
            ("contradiction_score", "REAL DEFAULT 0.0"),
            ("volatility_score", "REAL DEFAULT 0.0")
        ]
        
        for column_name, column_type in facts_columns:
            try:
                cursor.execute(f"ALTER TABLE facts ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to facts table")
            except Exception:
                # Column already exists
                pass
        
        print("🔧 Fixing contradictions table...")
        
        # Create contradictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                fact_a_text TEXT,
                fact_b_text TEXT,
                confidence REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT,
                FOREIGN KEY (fact_a_id) REFERENCES facts (id),
                FOREIGN KEY (fact_b_id) REFERENCES facts (id)
            )
        """)
        
        print("🔧 Fixing episodes table...")
        
        # Create episodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                end_time TEXT,
                subject_count INTEGER DEFAULT 0,
                fact_count INTEGER DEFAULT 0,
                summary TEXT,
                session_id TEXT,
                user_profile_id TEXT
            )
        """)
        
        # Add missing columns to episodes table
        episodes_columns = [
            ("session_id", "TEXT"),
            ("user_profile_id", "TEXT"),
            ("summary", "TEXT")
        ]
        
        for column_name, column_type in episodes_columns:
            try:
                cursor.execute(f"ALTER TABLE episodes ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to episodes table")
            except Exception:
                # Column already exists
                pass
        
        print("🔧 Fixing summaries table...")
        
        # Create summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                summary_text TEXT,
                fact_count INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add missing columns to summaries table  
        try:
            cursor.execute("ALTER TABLE summaries ADD COLUMN last_updated TEXT DEFAULT CURRENT_TIMESTAMP")
            print("   ✅ Added column last_updated to summaries table")
        except Exception:
            pass
        
        print("🔧 Creating indexes for performance...")
        
        # Create performance indexes
        indexes = [
            ("idx_facts_user_profile", "facts", "user_profile_id"),
            ("idx_facts_subject", "facts", "subject"),
            ("idx_facts_predicate", "facts", "predicate"),
            ("idx_facts_timestamp", "facts", "timestamp"),
            ("idx_facts_confidence", "facts", "confidence"),
            ("idx_contradictions_resolved", "contradictions", "resolved"),
            ("idx_episodes_session", "episodes", "session_id"),
            ("idx_summaries_subject", "summaries", "subject")
        ]
        
        for index_name, table_name, column_name in indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})")
                print(f"   ✅ Created index {index_name}")
            except Exception as e:
                print(f"   ⚠️ Could not create index {index_name}: {e}")
        
        print("🔧 Creating initial episode if none exists...")
        
        # Create initial episode if none exists
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episode_count = cursor.fetchone()[0]
        
        if episode_count == 0:
            cursor.execute("""
                INSERT INTO episodes (id, start_time, subject_count, fact_count, summary)
                VALUES (1, CURRENT_TIMESTAMP, 0, 0, 'Initial episode')
            """)
            print("   ✅ Created initial episode")
        
        # Commit all changes
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_tables():
    """Verify that all tables exist and have required columns."""
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("\n🔍 Verifying database schema...")
        
        # Check facts table
        cursor.execute("PRAGMA table_info(facts)")
        facts_columns = [row[1] for row in cursor.fetchall()]
        required_facts_columns = [
            'id', 'subject', 'predicate', 'object', 'confidence', 'timestamp',
            'user_profile_id', 'session_id', 'contradiction_score', 'volatility_score'
        ]
        
        facts_ok = all(col in facts_columns for col in required_facts_columns)
        print(f"   Facts table: {'✅' if facts_ok else '❌'} ({len(facts_columns)} columns)")
        
        # Check contradictions table
        cursor.execute("PRAGMA table_info(contradictions)")
        contradictions_columns = [row[1] for row in cursor.fetchall()]
        required_contradictions_columns = [
            'id', 'fact_a_id', 'fact_b_id', 'fact_a_text', 'fact_b_text', 'confidence', 'resolved'
        ]
        
        contradictions_ok = all(col in contradictions_columns for col in required_contradictions_columns)
        print(f"   Contradictions table: {'✅' if contradictions_ok else '❌'} ({len(contradictions_columns)} columns)")
        
        # Check episodes table
        cursor.execute("PRAGMA table_info(episodes)")
        episodes_columns = [row[1] for row in cursor.fetchall()]
        required_episodes_columns = [
            'id', 'start_time', 'fact_count', 'summary', 'session_id', 'user_profile_id'
        ]
        
        episodes_ok = all(col in episodes_columns for col in required_episodes_columns)
        print(f"   Episodes table: {'✅' if episodes_ok else '❌'} ({len(episodes_columns)} columns)")
        
        # Check summaries table
        cursor.execute("PRAGMA table_info(summaries)")
        summaries_columns = [row[1] for row in cursor.fetchall()]
        required_summaries_columns = [
            'id', 'subject', 'summary_text', 'fact_count', 'timestamp'
        ]
        
        summaries_ok = all(col in summaries_columns for col in required_summaries_columns)
        print(f"   Summaries table: {'✅' if summaries_ok else '❌'} ({len(summaries_columns)} columns)")
        
        conn.close()
        
        return facts_ok and contradictions_ok and episodes_ok and summaries_ok
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Run the migration."""
    
    print("🧠 MeRNSTA Memory System Database Migration")
    print("=" * 60)
    print("Fixing tables: facts, contradictions, episodes, summaries")
    print("")
    
    if fix_memory_tables():
        if verify_tables():
            print("\n✅ Migration completed successfully!")
            print("   All required tables and columns are now present.")
            return True
        else:
            print("\n⚠️  Migration completed with warnings.")
            print("   Some tables may be missing required columns.")
            return False
    else:
        print("\n❌ Migration failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 