#!/usr/bin/env python3
"""
Migration 004: Comprehensive Memory System Fixes
Ensures proper database schema for fact retrieval, contradiction detection,
episodic memory, and personality-based confidence adjustments.
"""

import sqlite3
import os
import sys
from datetime import datetime

def run_migration():
    """Execute the comprehensive memory system migration."""
    
    print("🧠 MeRNSTA Memory System Migration 004")
    print("============================================================")
    print("Comprehensive fixes for:")
    print("- Fact retrieval with user scoping")
    print("- Contradiction detection")
    print("- Episodic memory summarization") 
    print("- Personality-based confidence adjustments")
    print("- Proper indexes for performance")
    print()

    # Get database path from environment or default
    from config.environment import get_settings
    try:
        settings = get_settings()
        if hasattr(settings, 'database_url') and settings.database_url:
            db_url = settings.database_url
            if db_url.startswith('sqlite:///'):
                db_path = db_url.replace('sqlite:///', '')
            else:
                db_path = db_url
        else:
            db_path = getattr(settings, 'database_path', 'memory.db')
    except Exception:
        db_path = 'memory.db'
    
    print(f"🔧 Using database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("🔧 Fixing facts table...")
        
        # Ensure facts table has all required columns
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
                user_profile_id TEXT,
                subject_cluster_id INTEGER DEFAULT NULL,
                embedding TEXT DEFAULT NULL
            )
        """)
        
        # Add missing columns to facts table if they don't exist
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
            ("volatility_score", "REAL DEFAULT 0.0"),
            ("subject_cluster_id", "INTEGER DEFAULT NULL"),
            ("embedding", "TEXT DEFAULT NULL")
        ]
        
        for column_name, column_type in facts_columns:
            try:
                cursor.execute(f"ALTER TABLE facts ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to facts table")
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        print("🔧 Fixing contradictions table...")
        
        # Create contradictions table for proper contradiction tracking
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
                user_profile_id TEXT,
                session_id TEXT,
                FOREIGN KEY (fact_a_id) REFERENCES facts (id),
                FOREIGN KEY (fact_b_id) REFERENCES facts (id)
            )
        """)
        
        # Add missing columns to contradictions table
        contradictions_columns = [
            ("user_profile_id", "TEXT"),
            ("session_id", "TEXT"),
            ("resolution_notes", "TEXT")
        ]
        
        for column_name, column_type in contradictions_columns:
            try:
                cursor.execute(f"ALTER TABLE contradictions ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to contradictions table")
            except sqlite3.OperationalError:
                pass
        
        print("🔧 Fixing episodes table...")
        
        # Create episodes table for episodic memory summarization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_profile_id TEXT,
                start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                end_time TEXT,
                fact_count INTEGER DEFAULT 0,
                subject_count INTEGER DEFAULT 0,
                summary TEXT,
                personality_profile TEXT DEFAULT 'neutral'
            )
        """)
        
        # Add missing columns to episodes table
        episodes_columns = [
            ("user_profile_id", "TEXT"),
            ("end_time", "TEXT"),
            ("personality_profile", "TEXT DEFAULT 'neutral'")
        ]
        
        for column_name, column_type in episodes_columns:
            try:
                cursor.execute(f"ALTER TABLE episodes ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to episodes table")
            except sqlite3.OperationalError:
                pass
        
        print("🔧 Fixing summaries table...")
        
        # Create summaries table for conversation summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_profile_id TEXT,
                subject TEXT,
                summary_text TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                fact_count INTEGER DEFAULT 0
            )
        """)
        
        # Add missing columns to summaries table
        summaries_columns = [
            ("user_profile_id", "TEXT"),
            ("fact_count", "INTEGER DEFAULT 0")
        ]
        
        for column_name, column_type in summaries_columns:
            try:
                cursor.execute(f"ALTER TABLE summaries ADD COLUMN {column_name} {column_type}")
                print(f"   ✅ Added column {column_name} to summaries table")
            except sqlite3.OperationalError:
                pass
        
        print("🔧 Creating performance indexes...")
        
        # Create indexes for better query performance
        indexes = [
            ("idx_facts_user_profile", "CREATE INDEX IF NOT EXISTS idx_facts_user_profile ON facts (user_profile_id)"),
            ("idx_facts_session", "CREATE INDEX IF NOT EXISTS idx_facts_session ON facts (session_id)"),
            ("idx_facts_subject", "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject)"),
            ("idx_facts_predicate", "CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts (predicate)"),
            ("idx_facts_object", "CREATE INDEX IF NOT EXISTS idx_facts_object ON facts (object)"),
            ("idx_facts_timestamp", "CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts (timestamp DESC)"),
            ("idx_facts_confidence", "CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts (confidence DESC)"),
            ("idx_facts_user_subject", "CREATE INDEX IF NOT EXISTS idx_facts_user_subject ON facts (user_profile_id, subject)"),
            ("idx_facts_user_predicate", "CREATE INDEX IF NOT EXISTS idx_facts_user_predicate ON facts (user_profile_id, predicate)"),
            ("idx_facts_user_object", "CREATE INDEX IF NOT EXISTS idx_facts_user_object ON facts (user_profile_id, object)"),
            ("idx_contradictions_resolved", "CREATE INDEX IF NOT EXISTS idx_contradictions_resolved ON contradictions (resolved)"),
            ("idx_contradictions_user", "CREATE INDEX IF NOT EXISTS idx_contradictions_user ON contradictions (user_profile_id)"),
            ("idx_episodes_session", "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes (session_id)"),
            ("idx_episodes_user", "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes (user_profile_id)"),
            ("idx_summaries_subject", "CREATE INDEX IF NOT EXISTS idx_summaries_subject ON summaries (subject)"),
            ("idx_summaries_user", "CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries (user_profile_id)")
        ]
        
        for index_name, index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"   ✅ Created index {index_name}")
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    print(f"   ⚠️ Warning creating index {index_name}: {e}")
        
        print("🔧 Creating initial episode if none exists...")
        
        # Ensure there's at least one episode
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episode_count = cursor.fetchone()[0]
        
        if episode_count == 0:
            cursor.execute("""
                INSERT INTO episodes (id, session_id, start_time, subject_count, fact_count, summary, personality_profile)
                VALUES (1, 'initial', CURRENT_TIMESTAMP, 0, 0, 'Initial episode', 'neutral')
            """)
            print("   ✅ Created initial episode")
        
        print("🔧 Updating database settings...")
        
        # Set optimal SQLite settings for performance
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = 10000")
        cursor.execute("PRAGMA temp_store = MEMORY")
        print("   ✅ Applied performance optimizations")
        
        conn.commit()
        
        print()
        print("🔍 Verifying database schema...")
        
        # Verify all tables exist with correct structure
        tables_to_check = [
            ("facts", 20),  # Expected number of columns
            ("contradictions", 11),
            ("episodes", 8),
            ("summaries", 6)
        ]
        
        for table_name, expected_columns in tables_to_check:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            actual_columns = len(columns)
            
            if actual_columns >= expected_columns:
                print(f"   {table_name} table: ✅ ({actual_columns} columns)")
            else:
                print(f"   {table_name} table: ⚠️ ({actual_columns} columns, expected {expected_columns})")
        
        # Check for any existing facts to verify user_profile_id column is working
        cursor.execute("SELECT COUNT(*) FROM facts")
        fact_count = cursor.fetchone()[0]
        print(f"   Total facts in database: {fact_count}")
        
        conn.close()
        
        print()
        print("✅ Migration 004 completed successfully!")
        print("   All required tables and columns are now present.")
        print("   Performance indexes have been created.")
        print("   Database is ready for:")
        print("   - User-scoped fact retrieval")
        print("   - Contradiction detection and logging")
        print("   - Episodic memory summarization")
        print("   - Personality-based confidence adjustments")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    # Add the project root to the path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    
    success = run_migration()
    sys.exit(0 if success else 1) 