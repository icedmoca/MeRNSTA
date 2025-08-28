import logging
import queue
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, Optional
from config.settings import MAX_CONNECTIONS, RETRY_DELAY, RETRY_ATTEMPTS

# Configure logging for database operations
db_logger = logging.getLogger("database")


@dataclass
class ConnectionConfig:
    """Configuration for database connections"""

    db_path: str = "memory.db"
    max_connections: int = MAX_CONNECTIONS
    connection_timeout: float = 30.0
    retry_attempts: int = RETRY_ATTEMPTS
    retry_delay: float = RETRY_DELAY
    enable_wal: bool = True
    enable_foreign_keys: bool = True


class DatabaseConnectionPool:
    """
    Thread-safe connection pool for SQLite database operations.
    Handles connection lifecycle, retries, and proper cleanup.
    For in-memory databases, uses a single shared connection.
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pool: Queue = Queue(maxsize=config.max_connections)
        self._lock = threading.Lock()
        self._active_connections = 0
        self._total_connections_created = 0
        self._total_connections_used = 0
        self._shutdown = False
        
        # For in-memory databases, use a single shared connection
        self._is_memory_db = config.db_path == ":memory:"
        self._shared_memory_connection = None
        self._memory_lock = threading.RLock()  # Re-entrant lock for memory DB

        # Initialize the pool with initial connections
        if not self._is_memory_db:
            self._initialize_pool()
        else:
            # For memory DB, create the shared connection
            self._initialize_memory_connection()

        db_logger.info(f"Database connection pool initialized: {config.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool (alias for compatibility)."""
        return self.get_db_connection()

    def _initialize_memory_connection(self):
        """Initialize a single shared connection for in-memory database"""
        if self._is_memory_db:
            with self._memory_lock:
                if self._shared_memory_connection is None:
                    self._shared_memory_connection = self._create_connection()
                    if self._shared_memory_connection:
                        db_logger.info("Created shared in-memory database connection")

    def _initialize_pool(self):
        """Initialize the connection pool with initial connections"""
        try:
            for _ in range(min(3, self.config.max_connections)):
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn)
                    self._total_connections_created += 1
        except Exception as e:
            db_logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection with proper configuration"""
        try:
            # Handle URI connections (including shared in-memory databases)
            if self.config.db_path.startswith("file:"):
                conn = sqlite3.connect(
                    self.config.db_path, 
                    check_same_thread=False, 
                    uri=True,
                    timeout=self.config.connection_timeout
                )
            else:
                conn = sqlite3.connect(
                    self.config.db_path, 
                    check_same_thread=False,
                    timeout=self.config.connection_timeout
                )

            # Configure connection for production use with WAL mode
            if self.config.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                # WAL mode allows concurrent readers and writers
                conn.execute("PRAGMA wal_autocheckpoint=1000;")

            if self.config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON;")

            # Set timeout for busy database (30 seconds)
            conn.execute("PRAGMA busy_timeout=30000;")

            # Enable better performance
            conn.execute("PRAGMA cache_size=10000;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            
            # Set isolation level for better concurrency
            conn.isolation_level = None  # Autocommit mode for better performance

            return conn

        except sqlite3.Error as e:
            db_logger.error(f"Database error: {e}")
            reset_pool()
            return None
        except Exception as e:
            db_logger.error(f"Unexpected error creating connection: {e}")
            return None

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Get a connection from the pool or create a new one"""
        if self._shutdown:
            raise RuntimeError("Connection pool has been shut down")

        # For in-memory databases, return the shared connection
        if self._is_memory_db:
            with self._memory_lock:
                if self._shared_memory_connection is None:
                    self._shared_memory_connection = self._create_connection()
                    if self._shared_memory_connection:
                        db_logger.info("Created shared in-memory database connection")
                
                if self._shared_memory_connection:
                    self._total_connections_used += 1
                    db_logger.debug("Returning shared in-memory database connection")
                    return self._shared_memory_connection
                else:
                    db_logger.error("Failed to create in-memory database connection")
                    return None

        # For file-based databases, use the regular pool logic
        # Try to get from pool first
        try:
            conn = self._pool.get_nowait()
            self._total_connections_used += 1
            db_logger.info(f"Acquired connection from pool. Active: {self._active_connections}")
            return conn
        except Empty:
            pass

        # Create new connection if pool is empty and under limit
        with self._lock:
            if self._active_connections < self.config.max_connections:
                conn = self._create_connection()
                if conn:
                    self._active_connections += 1
                    self._total_connections_created += 1
                    self._total_connections_used += 1
                    db_logger.info(f"Created new connection. Active: {self._active_connections}")
                    return conn

        # Wait for a connection to become available
        try:
            conn = self._pool.get(timeout=self.config.connection_timeout)
            self._total_connections_used += 1
            db_logger.info(f"Waited for and acquired connection. Active: {self._active_connections}")
            return conn
        except Empty:
            db_logger.error(f"Timeout waiting for database connection after {self.config.connection_timeout}s")
            raise TimeoutError(
                f"Timeout waiting for database connection after {self.config.connection_timeout}s"
            )

    def _return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool or close it if pool is full"""
        if self._shutdown:
            try:
                conn.close()
            except Exception as e:
                db_logger.warning(f"Error closing connection during shutdown: {e}")
            return

        # For in-memory databases, don't actually return the shared connection
        # Just validate it's still working
        if self._is_memory_db:
            try:
                # Quick validation that connection is still alive
                conn.execute("SELECT 1").fetchone()
                db_logger.debug("In-memory database connection validated and kept active")
            except Exception as e:
                db_logger.warning(f"In-memory database connection validation failed: {e}")
                with self._memory_lock:
                    if self._shared_memory_connection is conn:
                        self._shared_memory_connection = None
                        db_logger.info("Reset shared in-memory connection due to validation failure")
            return

        # For file-based databases, use regular pool return logic
        if not self._validate_connection(conn):
            with self._lock:
                self._active_connections -= 1
            return

        try:
            self._pool.put_nowait(conn)
            db_logger.info(f"Returned connection to pool. Active: {self._active_connections}")
        except:
            # Pool is full, close the connection
            try:
                conn.close()
                with self._lock:
                    self._active_connections -= 1
                db_logger.info(f"Closed excess connection. Active: {self._active_connections}")
            except Exception as e:
                db_logger.warning(f"Error closing excess connection: {e}")

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Validate that a connection is still usable"""
        try:
            # Quick validation query
            conn.execute("SELECT 1").fetchone()
            # Reset connection state
            conn.rollback()
            return True
        except Exception as e:
            db_logger.warning(f"Connection validation failed: {e}")
            try:
                conn.close()
            except Exception:
                pass  # Ignore errors when closing invalid connection
            return False

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup"""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception as e:
            if conn:
                # Log the error and close the connection
                db_logger.error(f"Database operation failed: {e}")
                conn.close()
                with self._lock:
                    self._active_connections -= 1
            raise
        finally:
            if conn:
                self._return_connection(conn)

    def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute a database operation with retry logic"""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                with self.get_connection() as conn:
                    return operation()
            except sqlite3.OperationalError as e:
                last_exception = e
                if (
                    "database is locked" in str(e)
                    and attempt < self.config.retry_attempts - 1
                ):
                    db_logger.warning(
                        f"Database locked, retrying ({attempt + 1}/{self.config.retry_attempts})..."
                    )
                    time.sleep(self.config.retry_delay)
                else:
                    db_logger.error(f"Database operation failed after {attempt + 1} attempts: {e}")
                    break
        if last_exception:
            db_logger.error(f"Final database operation failure: {last_exception}")
            raise last_exception

    def shutdown(self):
        """Shutdown the connection pool and close all connections"""
        if self._shutdown:
            return

        self._shutdown = True
        db_logger.info("Shutting down database connection pool")

        # Close shared in-memory connection if it exists
        if self._is_memory_db:
            with self._memory_lock:
                if self._shared_memory_connection:
                    try:
                        self._shared_memory_connection.close()
                        db_logger.info("Closed shared in-memory database connection")
                    except Exception as e:
                        db_logger.warning(f"Error closing shared in-memory connection: {e}")
                    finally:
                        self._shared_memory_connection = None
        else:
            # Close all pooled connections for file-based databases
            while True:
                try:
                    conn = self._pool.get_nowait()
                    try:
                        conn.close()
                    except Exception as e:
                        db_logger.warning(f"Error closing pooled connection: {e}")
                except Empty:
                    break

        db_logger.info("Database connection pool shutdown complete")

    def get_stats(self) -> dict:
        """Get connection pool statistics"""
        return {
            "active_connections": self._active_connections,
            "pool_size": self._pool.qsize(),
            "total_created": self._total_connections_created,
            "total_used": self._total_connections_used,
            "shutdown": self._shutdown,
        }

    def close_all_connections(self):
        """Forcefully close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
        with self._lock:
            self._active_connections = 0


# Global connection pool instance
_connection_pool: Optional[DatabaseConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool(
    config: Optional[ConnectionConfig] = None,
    force_reset: bool = False
) -> DatabaseConnectionPool:
    """Get or create the global connection pool"""
    global _connection_pool

    # Check if we need to reset due to config change or force reset
    should_reset = force_reset
    if _connection_pool is not None and config is not None:
        # Check if the database path has changed
        if _connection_pool.config.db_path != config.db_path:
            should_reset = True

    if should_reset and _connection_pool is not None:
        db_logger.info(f"Resetting connection pool due to config change: {config.db_path if config else 'force reset'}")
        _connection_pool.shutdown()
        _connection_pool = None

    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                if config is None:
                    config = ConnectionConfig()
                _connection_pool = DatabaseConnectionPool(config)

    return _connection_pool


def reset_pool():
    global _connection_pool
    if _connection_pool:
        _connection_pool.shutdown()
        _connection_pool = None


def get_conn(db_path: str = "memory.db", *args, **kwargs) -> sqlite3.Connection:
    """
    Legacy function for backward compatibility.
    Creates a new connection (not from pool) for existing code.
    """
    # Check if this is a URI connection
    if db_path.startswith("file:"):
        conn = sqlite3.connect(db_path, check_same_thread=False, *args, **kwargs)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False, *args, **kwargs)

    # Configure connection
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")

    return conn


def with_retry(fn: Callable, retries: int = 3, delay: float = 0.1):
    """
    Legacy retry function for backward compatibility.
    Use DatabaseConnectionPool.execute_with_retry for new code.
    """
    for attempt in range(retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                db_logger.warning(
                    f"Database is locked, retrying ({attempt + 1}/{retries})..."
                )
                time.sleep(delay * (2**attempt))  # Exponential backoff
            else:
                raise
        except Exception as e:
            db_logger.error(f"Database operation failed: {e}")
            raise

    db_logger.error("Database operation failed after all retries")
    raise Exception("Database operation failed after all retries")


# === Backward-compat shims ===
def get_db_connection(db_path: str = "memory.db", *args, **kwargs) -> sqlite3.Connection:
    """Compatibility alias for legacy code."""
    return get_conn(db_path, *args, **kwargs)


def execute_query(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    """Compatibility helper to execute a query and commit."""
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur


# Cleanup function for application shutdown
def cleanup_database_pool():
    """Cleanup function to be called on application shutdown"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.shutdown()
        _connection_pool = None
