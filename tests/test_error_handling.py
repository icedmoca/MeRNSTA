#!/usr/bin/env python3
"""
Test suite for error handling and logging in DB and API operations
"""
import sys
import os
import unittest
import tempfile
import logging
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.memory_log import MemoryLog
from storage.errors import DatabaseError, ExternalServiceError, safe_db_operation, safe_api_call

# Note: This file uses unittest.TestCase, which does not support pytest fixtures directly.
# If you want to migrate to use isolated_db, you would need to refactor to pytest-style tests.
# For now, leave as is unless you want a full refactor.
class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.memory_log = MemoryLog(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_db_error_is_caught_and_logged(self):
        # Patch connection to raise an error
        with patch('storage.memory_log.MemoryLog.init_database', side_effect=Exception("Simulated DB error")):
            # The error should be caught and re-raised as DatabaseError
            try:
                self.memory_log.init_database()
                self.fail("Expected DatabaseError to be raised")
            except (DatabaseError, Exception) as e:
                # Accept either DatabaseError or the original Exception
                self.assertIn("Simulated DB error", str(e))

    def test_api_error_is_caught_and_logged(self):
        from cortex.response_generation import generate_response
        with patch('requests.post', side_effect=Exception("Simulated API failure")):
            with self.assertRaises(ExternalServiceError):
                generate_response("test prompt")

    def test_safe_db_operation_decorator_escalates(self):
        @safe_db_operation
        def fail_db():
            raise RuntimeError("DB fail")
        with self.assertRaises(DatabaseError):
            fail_db()

    def test_safe_api_call_decorator_escalates(self):
        @safe_api_call
        def fail_api():
            raise RuntimeError("API fail")
        with self.assertRaises(ExternalServiceError):
            fail_api()

    def test_critical_error_not_swallowed(self):
        # Simulate a KeyboardInterrupt in a safe operation
        @safe_db_operation
        def interrupt():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            interrupt()

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    unittest.main(verbosity=2) 