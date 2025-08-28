"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

from storage.memory_log import MemoryLog

client = TestClient(app)

# Remove the custom memory_log fixture and use isolated_db directly in tests

def test_get_memory_report():
    response = client.get("/memory_report")
    assert response.status_code == 200
    assert "report" in response.json()

# Add similar tests for other endpoints...

def test_get_clusters():
    response = client.get("/clusters")
    assert response.status_code == 200
    assert "clusters" in response.json()

# ... add more ...
""" 