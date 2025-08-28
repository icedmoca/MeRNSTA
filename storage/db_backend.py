"""
Unified database backend for MeRNSTA
Supports switching between SQLite, PostgreSQL, and pgvector
"""

import os
import logging
from typing import Optional, Union, Any
from dataclasses import dataclass
from urllib.parse import urlparse

import sqlite3
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False
    
try:
    import pgvector
    from pgvector.psycopg2 import register_vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

from config.environment import get_settings
from storage.db_utils import ConnectionConfig

logger = logging.getLogger(__name__)

@dataclass
class DatabaseBackend:
    """Unified database backend configuration"""
    backend_type: str  # 'sqlite', 'postgresql', 'pgvector'
    connection_string: str
    is_memory: bool = False
    
    @classmethod
    def from_url(cls, url: str) -> 'DatabaseBackend':
        """Create backend from database URL"""
        parsed = urlparse(url)
        
        if parsed.scheme in ['sqlite', 'sqlite3']:
            # SQLite URL format: sqlite:///path/to/db.db
            db_path = url.replace('sqlite:///', '')
            is_memory = db_path == ':memory:' or db_path == ''
            return cls(
                backend_type='sqlite',
                connection_string=db_path or ':memory:',
                is_memory=is_memory
            )
        elif parsed.scheme in ['postgresql', 'postgres']:
            # Check if pgvector extension is requested
            if 'pgvector' in url or HAS_PGVECTOR:
                return cls(
                    backend_type='pgvector',
                    connection_string=url,
                    is_memory=False
                )
            else:
                return cls(
                    backend_type='postgresql',
                    connection_string=url,
                    is_memory=False
                )
        else:
            raise ValueError(f"Unsupported database scheme: {parsed.scheme}")

class UnifiedDatabaseConnection:
    """Unified connection interface for all database backends"""
    
    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection based on backend type"""
        if self.backend.backend_type == 'sqlite':
            self.connection = sqlite3.connect(
                self.backend.connection_string,
                check_same_thread=False,
                timeout=30.0
            )
            self._setup_sqlite()
            
        elif self.backend.backend_type in ['postgresql', 'pgvector']:
            if not HAS_PSYCOPG:
                raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
            
            self.connection = psycopg2.connect(self.backend.connection_string)
            
            if self.backend.backend_type == 'pgvector':
                if not HAS_PGVECTOR:
                    raise ImportError("pgvector not installed. Run: pip install pgvector")
                self._setup_pgvector()
            else:
                self._setup_postgresql()
        else:
            raise ValueError(f"Unknown backend type: {self.backend.backend_type}")
    
    def _setup_sqlite(self):
        """Configure SQLite connection"""
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA busy_timeout=30000;")
        cursor.execute("PRAGMA cache_size=10000;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        self.connection.commit()
        logger.info("SQLite database configured")
    
    def _setup_postgresql(self):
        """Configure PostgreSQL connection"""
        cursor = self.connection.cursor()
        # PostgreSQL-specific setup
        cursor.execute("SET work_mem = '256MB';")
        cursor.execute("SET maintenance_work_mem = '256MB';")
        self.connection.commit()
        logger.info("PostgreSQL database configured")
    
    def _setup_pgvector(self):
        """Configure pgvector extension"""
        cursor = self.connection.cursor()
        
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Register vector type with psycopg2
        register_vector(self.connection)
        
        # Create vector-specific tables/indexes
        self._create_vector_tables()
        
        self.connection.commit()
        logger.info("pgvector database configured")
    
    def _create_vector_tables(self):
        """Create pgvector-specific tables with vector columns"""
        cursor = self.connection.cursor()
        
        # Convert embedding columns to vector type
        cursor.execute("""
            ALTER TABLE facts 
            ALTER COLUMN embedding TYPE vector(384) 
            USING embedding::vector
        """)
        
        # Create HNSW index for similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS facts_embedding_hnsw_idx 
            ON facts USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        
        logger.info("pgvector tables and indexes created")
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute query with backend-specific parameter handling"""
        cursor = self.connection.cursor()
        
        # Convert SQLite-style parameters to PostgreSQL style if needed
        if self.backend.backend_type in ['postgresql', 'pgvector']:
            # Replace ? with %s for PostgreSQL
            query = query.replace('?', '%s')
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        return cursor
    
    def executemany(self, query: str, params_list: list) -> Any:
        """Execute many queries with backend-specific handling"""
        cursor = self.connection.cursor()
        
        if self.backend.backend_type in ['postgresql', 'pgvector']:
            query = query.replace('?', '%s')
        
        cursor.executemany(query, params_list)
        return cursor
    
    def commit(self):
        """Commit transaction"""
        self.connection.commit()
    
    def rollback(self):
        """Rollback transaction"""
        self.connection.rollback()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()

def get_database_backend() -> DatabaseBackend:
    """Get database backend from configuration"""
    settings = get_settings()
    
    # Check environment variable first
    db_url = settings.database_url
    
    # Fall back to config.yaml
    if not db_url:
        from config.settings import _cfg
        db_url = _cfg.get('storage_uri', 'sqlite:///memory.db')
    
    return DatabaseBackend.from_url(db_url)

def get_unified_connection() -> UnifiedDatabaseConnection:
    """Get a unified database connection"""
    backend = get_database_backend()
    return UnifiedDatabaseConnection(backend) 