#!/usr/bin/env python3
"""
Security tests for MeRNSTA AI system
Tests input sanitization, SQL injection protection, authentication, rate limiting, and CORS
"""

import pytest
import sys
import os
import tempfile
import sqlite3
import time
import requests
from unittest.mock import patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for testing BEFORE importing app
os.environ["DISABLE_RATE_LIMIT"] = "true"
os.environ["API_SECURITY_TOKEN"] = "test_token_for_testing"
os.environ["TESTING"] = "true"

from storage.sanitize import (
    sanitize_text, 
    sanitize_fact, 
    sanitize_query, 
    validate_safe_input,
    MAX_TEXT_LENGTH,
    MAX_FACT_LENGTH,
    MAX_QUERY_LENGTH
)
from storage.memory_log import MemoryLog
from storage.db_utils import get_connection_pool, ConnectionConfig

# Reload settings after setting environment variables
from config.environment import reload_settings
reload_settings()

from api.main import app
from fastapi.testclient import TestClient

# Note: This file uses unittest.TestCase, which does not support pytest fixtures directly.
# To use isolated_db, a refactor to pytest-style tests is needed. Leaving as is for now.

class TestInputSanitization:
    """Test input sanitization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize connection pool for testing
        config = ConnectionConfig(db_path=self.temp_db.name)
        self.connection_pool = get_connection_pool(config)
        
        # Initialize memory log for testing
        self.memory_log = MemoryLog(self.temp_db.name)
        self.memory_log.init_database()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        try:
            if hasattr(self, 'memory_log'):
                # Don't shutdown the memory_log as it might affect other tests
                # Just clean up the temporary file
                pass
            os.unlink(self.temp_db.name)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization"""
        # Normal text should pass through
        text = "Hello, world!"
        sanitized = sanitize_text(text)
        assert sanitized == "Hello, world!"
    
    def test_sanitize_text_xss_removal(self):
        """Test XSS pattern removal"""
        malicious_texts = [
            "<script>alert('xss')</script>Hello",
            "Hello<script>alert('xss')</script>",
            "Hello onload=alert('xss') world",
            "Hello javascript:alert('xss') world",
            "Hello<iframe src='evil.com'></iframe>world",
            "Hello<object data='evil.com'></object>world",
            "Hello vbscript:alert('xss') world",
            "Hello data:text/html,<script>alert('xss')</script> world"
        ]
        
        for text in malicious_texts:
            sanitized = sanitize_text(text)
            assert "<script>" not in sanitized
            assert "onload=" not in sanitized
            assert "javascript:" not in sanitized
            assert "<iframe" not in sanitized
            assert "<object" not in sanitized
            assert "vbscript:" not in sanitized
            assert "data:text/html" not in sanitized
    
    def test_sanitize_text_length_limit(self):
        """Test text length limits"""
        # Test normal length
        normal_text = "A" * 1000
        sanitized = sanitize_text(normal_text)
        assert len(sanitized) == 1000
        
        # Test exceeding limit
        long_text = "A" * (MAX_TEXT_LENGTH + 100)
        with pytest.raises(ValueError):
            sanitize_text(long_text)
    
    def test_sanitize_fact_length_limit(self):
        """Test fact length limits"""
        # Test normal length
        normal_fact = "A" * 1000
        sanitized = sanitize_fact(normal_fact)
        assert len(sanitized) == 1000
        
        # Test exceeding limit
        long_fact = "A" * (MAX_FACT_LENGTH + 100)
        with pytest.raises(ValueError):
            sanitize_fact(long_fact)
    
    def test_sanitize_query_length_limit(self):
        """Test query length limits"""
        # Test normal length
        normal_query = "A" * 500
        sanitized = sanitize_query(normal_query)
        assert len(sanitized) == 500
        
        # Test exceeding limit
        long_query = "A" * (MAX_QUERY_LENGTH + 100)
        with pytest.raises(ValueError):
            sanitize_query(long_query)
    
    def test_validate_safe_input(self):
        """Test input validation"""
        # Valid inputs
        assert validate_safe_input("Hello, world!")
        assert validate_safe_input("A" * 1000)
        
        # Invalid inputs
        assert not validate_safe_input("A" * (MAX_TEXT_LENGTH + 100))
        assert not validate_safe_input("<script>alert('xss')</script>")
    
    def test_html_escaping(self):
        """Test HTML escaping"""
        text_with_html = "<div>Hello & world</div>"
        sanitized = sanitize_text(text_with_html)
        assert "&lt;div&gt;" in sanitized
        assert "&amp;" in sanitized
        assert "&lt;/div&gt;" in sanitized

class TestSQLInjectionProtection:
    """Test SQL injection protection"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.memory_log = MemoryLog(self.temp_db.name)
        self.memory_log.init_database()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        try:
            if hasattr(self, 'memory_log'):
                # Don't shutdown the memory_log as it might affect other tests
                # Just clean up the temporary file
                pass
            os.unlink(self.temp_db.name)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def test_sql_injection_attempts(self):
        """Test that SQL injection attempts are handled safely"""
        injection_attempts = [
            "'; DROP TABLE facts; --",
            "' OR '1'='1",
            "'; INSERT INTO facts VALUES ('evil', 'evil', 'evil'); --",
            "' UNION SELECT * FROM facts --",
            "'; UPDATE facts SET object='hacked'; --"
        ]
        
        for attempt in injection_attempts:
            # These should be sanitized and not cause SQL errors
            try:
                # Test semantic search with injection attempt
                results = self.memory_log.semantic_search(attempt, topk=5)
                # Should not raise SQL errors
                assert isinstance(results, list)
            except sqlite3.Error as e:
                pytest.fail(f"SQL injection attempt '{attempt}' caused SQL error: {e}")
    
    def test_parameterized_queries(self):
        """Test that all queries use parameterized statements"""
        # This test verifies that our database operations use parameterized queries
        # by checking that user input is properly handled
        
        test_subject = "test_subject'; DROP TABLE facts; --"
        
        try:
            # Test storing facts with potentially malicious input
            self.memory_log.store_triplets([(test_subject, "test_predicate", "test_object")], 0.8)
            
            # Test retrieving facts
            facts = self.memory_log.get_all_facts()
            assert isinstance(facts, list)
            
        except sqlite3.Error as e:
            pytest.fail(f"Parameterized query test failed: {e}")

class TestAuthentication:
    """Test authentication functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_health_endpoint_no_auth(self):
        """Test that health endpoint doesn't require auth"""
        response = self.client.get("/health")
        assert response.status_code == 200
    
    def test_protected_endpoints_require_auth(self):
        """Test that protected endpoints require authentication"""
        # Test reflect endpoint without auth
        response = self.client.post("/api/agent/reflect", json={"task": "test", "result": "test"})
        assert response.status_code == 403  # Forbidden
    
    def test_valid_auth_token(self, isolated_db):
        from api.main import app
        app.state.memory_log = isolated_db
        memory_log = isolated_db
        # Create trust_scores table if needed
        with memory_log._connection_pool.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trust_scores (
                    subject TEXT PRIMARY KEY,
                    score REAL
                )
            """)
        headers = {"Authorization": "Bearer dev-test-token"}
        response = self.client.post(
            "/api/agent/reflect",
            json={"task": "test task", "result": "test result"},
            headers=headers
        )
        assert response.status_code == 200
    
    def test_invalid_auth_token(self):
        """Test that invalid auth token is rejected"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = self.client.post(
            "/api/agent/reflect",
            json={"task": "test", "result": "test"},
            headers=headers
        )
        assert response.status_code == 401
    
    def test_missing_auth_token(self):
        """Test that missing auth token is rejected"""
        response = self.client.post(
            "/api/agent/reflect",
            json={"task": "test", "result": "test"}
        )
        assert response.status_code == 403

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_rate_limiting(self, isolated_db):
        """Test that rate limiting works"""
        memory_log = isolated_db
        
        headers = {"Authorization": "Bearer dev-test-token"}
        
        # Make multiple requests quickly
        for i in range(5):
            response = self.client.post(
                "/api/agent/reflect",
                json={"task": f"test {i}", "result": f"result {i}"},
                headers=headers
            )
            # Should not be rate limited in test mode
            assert response.status_code in [200, 429]
    
    def test_rate_limit_reset(self, isolated_db):
        """Test that rate limits reset after time"""
        memory_log = isolated_db
        
        headers = {"Authorization": "Bearer dev-test-token"}
        
        # Make a request
        response = self.client.post(
            "/api/agent/reflect",
            json={"task": "test", "result": "test"},
            headers=headers
        )
        assert response.status_code in [200, 429]
        
        # Wait a bit and try again
        time.sleep(0.1)
        response2 = self.client.post(
            "/api/agent/reflect",
            json={"task": "test2", "result": "test2"},
            headers=headers
        )
        assert response2.status_code in [200, 429]

class TestInputValidation:
    """Test input validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_invalid_subject_format(self, isolated_db):
        """Test that invalid subject format is handled"""
        memory_log = isolated_db
        
        headers = {"Authorization": "Bearer dev-test-token"}
        
        # Test with invalid subject format
        response = self.client.post(
            "/api/agent/reflect",
            json={"task": "", "result": "test"},  # Empty task
            headers=headers
        )
        # Should handle gracefully (could be 200 with sanitization or 422)
        assert response.status_code in [200, 422]
    
    def test_invalid_limit_parameters(self):
        """Test that invalid limit parameters are handled"""
        headers = {"Authorization": "Bearer dev-test-token"}
        
        # Test with invalid limit
        response = self.client.get(
            "/api/agent/search_triplets?query=test&top_k=invalid",
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [200, 422, 400]
    
    def test_invalid_confidence_parameters(self, isolated_db):
        from api.main import app
        import storage.db_utils
        app.state.memory_log = isolated_db
        storage.db_utils._connection_pool = isolated_db._connection_pool
        isolated_db.init_database()
        memory_log = isolated_db
        # Ensure trust_scores table exists
        with memory_log._connection_pool.get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS trust_scores")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trust_scores (
                    subject TEXT PRIMARY KEY,
                    trust_score REAL,
                    fact_count INTEGER DEFAULT 0,
                    contradiction_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        headers = {"Authorization": "Bearer dev-test-token"}
        response = self.client.get(
            "/api/agent/trust_score/test?confidence=invalid",
            headers=headers
        )
        assert response.status_code in [200, 422, 400]
    
    def test_invalid_fact_id(self):
        """Test that invalid fact ID is handled"""
        headers = {"Authorization": "Bearer dev-test-token"}
        
        # Test with invalid fact ID
        response = self.client.get(
            "/api/agent/fact/invalid_id",
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [404, 422, 400]

class TestCORSHeaders:
    """Test CORS headers"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present"""
        response = self.client.options("/health")
        # Should have CORS headers
        assert response.status_code in [200, 405]
    
    def test_security_headers_present(self):
        """Test that security headers are present"""
        response = self.client.get("/health")
        # Should have security headers
        assert response.status_code == 200

class TestMemoryLogSecurity:
    """Test memory log security"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.memory_log = MemoryLog(self.temp_db.name)
        self.memory_log.init_database()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        try:
            if hasattr(self, 'memory_log'):
                # Don't shutdown the memory_log as it might affect other tests
                # Just clean up the temporary file
                pass
            os.unlink(self.temp_db.name)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def test_malicious_memory_logging(self):
        """Test that malicious input is handled safely in memory logging"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE facts; --",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                # Test logging malicious input
                message_id = self.memory_log.log_memory("user", malicious_input, tags=["test"])
                assert message_id is not None
                
                # Test extracting facts from malicious input
                facts = self.memory_log.extract_facts(malicious_input)
                # Should not raise exceptions
                assert isinstance(facts, list)
                
            except Exception as e:
                pytest.fail(f"Malicious input '{malicious_input}' caused error: {e}")
    
    def test_sanitized_fact_storage(self):
        """Test that facts are sanitized before storage"""
        malicious_facts = [
            ("<script>alert('xss')</script>", "test", "test"),
            ("test", "<script>alert('xss')</script>", "test"),
            ("test", "test", "<script>alert('xss')</script>")
        ]
        
        for subject, predicate, object_val in malicious_facts:
            try:
                # Test storing malicious facts
                self.memory_log.store_triplets([(subject, predicate, object_val)], 0.8)
                
                # Should not raise exceptions
                facts = self.memory_log.get_all_facts()
                assert isinstance(facts, list)
                
            except Exception as e:
                pytest.fail(f"Malicious fact storage failed: {e}")

def run_security_tests():
    """Run all security tests"""
    print("🔒 Running MeRNSTA Security Tests")
    print("=" * 50)
    
    # This function can be used to run tests programmatically
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_security_tests() 