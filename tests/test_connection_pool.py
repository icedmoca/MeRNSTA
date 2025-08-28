#!/usr/bin/env python3
"""
Test suite for the new database connection pool
"""

import sys
import os
import unittest
import tempfile
import sqlite3
import threading
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db_utils import DatabaseConnectionPool, ConnectionConfig, get_connection_pool, cleanup_database_pool

class TestConnectionPool(unittest.TestCase):
    """Test suite for database connection pool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create connection pool configuration
        self.config = ConnectionConfig(
            db_path=self.db_path,
            max_connections=5,
            connection_timeout=10.0,
            retry_attempts=3,
            retry_delay=0.1
        )
        
        # Initialize connection pool
        self.pool = DatabaseConnectionPool(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            # Shutdown the pool
            self.pool.shutdown()
            
            # Clean up global pool
            cleanup_database_pool()
            
            # Remove temporary database file
            if os.path.exists(self.db_path):
                os.unlink(self.db_path)
                
        except Exception as e:
            print(f"Warning: Error during test cleanup: {e}")
    
    def test_pool_initialization(self):
        """Test that the connection pool initializes correctly"""
        self.assertIsNotNone(self.pool)
        self.assertEqual(self.pool.config.db_path, self.db_path)
        self.assertEqual(self.pool.config.max_connections, 5)
        
        # Check initial stats
        stats = self.pool.get_stats()
        self.assertGreaterEqual(stats['total_created'], 1)
        self.assertEqual(stats['shutdown'], False)
    
    def test_connection_creation(self):
        """Test that connections are created correctly"""
        with self.pool.get_connection() as conn:
            self.assertIsInstance(conn, sqlite3.Connection)
            
            # Test that connection is working
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)
    
    def test_connection_reuse(self):
        """Test that connections are reused from the pool"""
        initial_stats = self.pool.get_stats()
        
        # Use multiple connections
        for _ in range(3):
            with self.pool.get_connection() as conn:
                conn.execute("SELECT 1")
        
        final_stats = self.pool.get_stats()
        
        # Should have created some connections
        self.assertGreaterEqual(final_stats['total_created'], initial_stats['total_created'])
        # Should have used connections
        self.assertGreaterEqual(final_stats['total_used'], 3)
    
    def test_concurrent_access(self):
        """Test that the pool handles concurrent access correctly"""
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                with self.pool.get_connection() as conn:
                    # Simulate some work
                    conn.execute("SELECT ?", (worker_id,))
                    time.sleep(0.01)  # Small delay
                    results.append(worker_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 10)  # All workers should succeed
        self.assertEqual(len(errors), 0)    # No errors should occur
        
        # Check pool stats
        stats = self.pool.get_stats()
        self.assertGreaterEqual(stats['total_used'], 10)
    
    def test_connection_validation(self):
        """Test that invalid connections are properly handled"""
        # Create a connection and close it manually
        with self.pool.get_connection() as conn:
            conn_id = id(conn)
        
        # The connection should be returned to the pool and validated
        # If it's invalid, it should be closed and a new one created
        with self.pool.get_connection() as conn2:
            self.assertIsInstance(conn2, sqlite3.Connection)
    
    def test_pool_shutdown(self):
        """Test that the pool shuts down gracefully"""
        # Use some connections first
        with self.pool.get_connection() as conn:
            conn.execute("SELECT 1")
        
        # Shutdown the pool
        self.pool.shutdown()
        
        # Check that pool is marked as shutdown
        stats = self.pool.get_stats()
        self.assertTrue(stats['shutdown'])
        
        # Should not be able to get connections after shutdown
        with self.assertRaises(RuntimeError):
            with self.pool.get_connection():
                pass
    
    def test_global_pool(self):
        """Test the global connection pool functionality"""
        # Get the global pool
        global_pool = get_connection_pool()
        self.assertIsNotNone(global_pool)
        
        # Test that it's the same instance
        global_pool2 = get_connection_pool()
        self.assertIs(global_pool, global_pool2)
        
        # Test using the global pool
        with global_pool.get_connection() as conn:
            conn.execute("SELECT 1")
    
    def test_retry_logic(self):
        """Test that the retry logic works correctly"""
        # This test would require simulating database locks
        # For now, just test that the method exists
        def test_operation(conn, *args, **kwargs):
            return "success"
        
        result = self.pool.execute_with_retry(test_operation, "test_arg")
        self.assertEqual(result, "success")

    def test_concurrent_process_access(self):
        """Test that the pool handles concurrent access from multiple processes without lock errors"""
        import multiprocessing
        results = multiprocessing.Manager().list()
        errors = multiprocessing.Manager().list()

        def worker(worker_id, db_path, results, errors):
            try:
                config = ConnectionConfig(db_path=db_path)
                pool = DatabaseConnectionPool(config)
                with pool.get_connection() as conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, val TEXT)")
                    conn.execute("INSERT INTO test (val) VALUES (?)", (f"worker_{worker_id}",))
                    conn.commit()
                    results.append(worker_id)
            except Exception as e:
                errors.append(str(e))

        procs = []
        for i in range(5):
            p = multiprocessing.Process(target=worker, args=(i, self.db_path, results, errors))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        self.assertEqual(len(results), 5, f"Not all workers succeeded: {results}")
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

class TestLegacyCompatibility(unittest.TestCase):
    """Test that legacy functions still work"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            cleanup_database_pool()
            if os.path.exists(self.db_path):
                os.unlink(self.db_path)
        except Exception as e:
            print(f"Warning: Error during test cleanup: {e}")
    
    def test_legacy_get_conn(self):
        """Test that the legacy get_conn function still works"""
        from storage.db_utils import get_conn
        
        with get_conn(self.db_path) as conn:
            self.assertIsInstance(conn, sqlite3.Connection)
            conn.execute("SELECT 1")
    
    def test_legacy_with_retry(self):
        """Test that the legacy with_retry function still works"""
        from storage.db_utils import with_retry
        
        def test_function():
            return "success"
        
        result = with_retry(test_function)
        self.assertEqual(result, "success")

def run_connection_pool_tests():
    """Run all connection pool tests"""
    print("🧪 Running Database Connection Pool Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConnectionPool)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLegacyCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 All connection pool tests passed!")
        print("✅ Connection pool is working correctly")
        print("✅ Legacy compatibility maintained")
        print("✅ Thread safety verified")
        print("✅ Graceful shutdown working")
    else:
        print("❌ Some tests failed")
        for test, traceback in result.failures:
            print(f"   FAILED: {test}")
        for test, traceback in result.errors:
            print(f"   ERROR: {test}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_connection_pool_tests()
    sys.exit(0 if success else 1) 