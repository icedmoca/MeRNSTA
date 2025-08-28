import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_dashboard_facts():
    resp = client.get("/dashboard/facts?page=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "facts" in data and isinstance(data["facts"], list)
    assert "total" in data and "page" in data and "limit" in data

def test_get_dashboard_contradictions():
    resp = client.get("/dashboard/contradictions")
    assert resp.status_code == 200
    data = resp.json()
    assert "contradictions" in data and isinstance(data["contradictions"], list)

def test_get_dashboard_clusters():
    resp = client.get("/dashboard/clusters")
    assert resp.status_code == 200
    data = resp.json()
    assert "clusters" in data and isinstance(data["clusters"], list)

def test_get_dashboard_metrics():
    resp = client.get("/dashboard/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data

def test_dashboard_html_served():
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "MeRNSTA Memory Visualization Dashboard" in resp.text 