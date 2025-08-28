import pytest
import tempfile
import os
import sqlite3
import sys
import shutil
import warnings
import numpy as np
import pytest
import spacy
from spacy.cli import download
import io
import importlib.util
import pytest
from tests.utils.db_cleanup import safe_cleanup_database

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage
from storage.memory_log import MemoryLog
from storage.db_utils import get_connection_pool, ConnectionConfig, DatabaseConnectionPool, reset_pool

# Import settings and configure for testing
from config.environment import get_settings
settings = get_settings()
from config.settings import DATABASE_CONFIG

class DLASCLFilter(io.TextIOWrapper):
    def write(self, s):
        if "On entry to DLASCL" not in s:
            super().write(s)

def run_cross_session_migration(db_path):
    """Run the add_cross_session_fields migration for the given db_path."""
    migration_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage', 'migrations', 'add_cross_session_fields.py')
    spec = importlib.util.spec_from_file_location("add_cross_session_fields", migration_path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    # Patch DATABASE_CONFIG to use the test db_path
    from config import settings as config_settings
    orig_path = config_settings.DATABASE_CONFIG['path']
    config_settings.DATABASE_CONFIG['path'] = db_path
    migration.run()
    config_settings.DATABASE_CONFIG['path'] = orig_path

@pytest.fixture(scope="session", autouse=True)
def setup_global_database():
    """Global database setup for all tests"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce noise in tests
    
    # Disable rate limiting and auth for tests
    os.environ['DISABLE_RATE_LIMIT'] = 'true'
    os.environ['API_SECURITY_TOKEN'] = 'test_token_for_testing'
    
    # Reload settings to pick up the new environment variables
    from config.environment import reload_settings
    reload_settings()
    
    yield

@pytest.fixture(scope="function")
def temp_db():
    """Create a temporary database for each test"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Initialize the database
    memory_log = MemoryLog(db_path)
    memory_log.init_database()
    run_cross_session_migration(db_path)
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass

@pytest.fixture(scope="function")
def memory_log():
    reset_pool()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    DATABASE_CONFIG['path'] = db_path
    log = None
    try:
        log = __import__('storage.memory_log', fromlist=['MemoryLog']).MemoryLog(db_path)
        yield log
    finally:
        reset_pool()
        if os.path.exists(db_path):
            os.remove(db_path)

@pytest.fixture(autouse=True)
def ensure_db_pool():
    """Ensure database connection pool is properly initialized and not shut down"""
    # Get the global connection pool
    pool = get_connection_pool()
    
    # If pool is shut down, re-initialize it
    if hasattr(pool, '_shutdown') and pool._shutdown:
        new_pool = DatabaseConnectionPool()
        # Replace the global pool reference
        import storage.db_utils
        storage.db_utils._connection_pool = new_pool
    
    yield

@pytest.fixture(autouse=True)
def reset_db_pool():
    """Reset database pool in app state and global state"""
    # Reset the global memory_log instance if it exists
    try:
        from api.main import app
        if hasattr(app.state, 'memory_log'):
            app.state.memory_log._connection_pool = DatabaseConnectionPool()
    except:
        pass
    
    # Reset any global memory_log instances
    try:
        import api.routes.agent
        if hasattr(api.routes.agent, 'memory_log'):
            api.routes.agent.memory_log._connection_pool = DatabaseConnectionPool()
    except:
        pass
    
    try:
        import api.routes.memory
        if hasattr(api.routes.memory, 'memory_log'):
            api.routes.memory.memory_log._connection_pool = DatabaseConnectionPool()
    except:
        pass
    
    yield

@pytest.fixture
def isolated_db(tmp_path):
    db_path = tmp_path / "test.db"
    # fully reset global pool
    try:
        pool = storage.db_utils.get_connection_pool(storage.db_utils.ConnectionConfig(db_path=str(db_path)))
        if hasattr(pool, 'close_all_connections'):
            pool.close_all_connections()
    except Exception:
        pass
    storage.db_utils._connection_pool = None

    # remove file & WAL/SHM
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"{db_path}{ext}")
        except FileNotFoundError:
            pass

    # init fresh DB & schema
    memory_log = MemoryLog(str(db_path))
    memory_log.init_database()
    run_cross_session_migration(str(db_path))

    yield memory_log

    # cleanup after test
    if hasattr(memory_log._connection_pool, 'close_all_connections'):
        memory_log._connection_pool.close_all_connections()
    storage.db_utils._connection_pool = None
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"{db_path}{ext}")
        except FileNotFoundError:
            pass

@pytest.fixture(scope="function")
def sample_facts():
    """Sample facts for testing"""
    return [
        ("Alice", "likes", "pizza", 0.9),
        ("Bob", "hates", "sushi", 0.8),
        ("Alice", "works", "at Google", 0.7),
        ("Bob", "lives", "in New York", 0.6),
    ]

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce noise in tests
    
    # Disable rate limiting and auth for tests
    os.environ['DISABLE_RATE_LIMIT'] = 'true'
    os.environ['API_SECURITY_TOKEN'] = 'test_token_for_testing'
    
    # Reload settings to pick up the new environment variables
    from config.environment import reload_settings
    reload_settings()
    
    yield
    
    # Cleanup after each test
    pass 

@pytest.fixture(scope="session")
def nlp():
    """Ensure en_core_web_sm is available and loaded."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

def pytest_configure():
    import spacy
    from spacy.cli import download
    orig_load = spacy.load
    def patched_load(name, *args, **kwargs):
        try:
            return orig_load(name, *args, **kwargs)
        except OSError:
            if name == "en_core_web_sm":
                download("en_core_web_sm")
                return orig_load(name, *args, **kwargs)
            raise
    spacy.load = patched_load
    # Monkey-patch sys.stderr to filter DLASCL warnings
    sys.stderr = DLASCLFilter(sys.stderr.buffer, sys.stderr.encoding) 

@pytest.fixture(autouse=True, scope='function')
def cleanup_and_migrate():
    safe_cleanup_database()
    # For each test DB, run the migration
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    for db_file in db_files:
        run_cross_session_migration(db_file) 

@pytest.fixture(scope="function")
def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    DATABASE_CONFIG["path"] = str(db_path)
    with get_connection_pool().get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                embedding TEXT,
                media_type TEXT,
                user_profile_id TEXT,
                session_id TEXT,
                subject_cluster_id TEXT,
                confidence REAL,
                contradiction_score REAL,
                volatility_score REAL
            )
        """)
        conn.commit()
    yield 