import os
import tempfile
import sqlite3
from cortex.memory_ops import evolve_file_with_context
from storage.db import db
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_patch_generation_and_dry_run():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("print('hello')\n")
        tf.flush()
        result = evolve_file_with_context(tf.name, "add copyright", dry_run=True)
        assert result["status"] == "dry_run"
        assert "Evolution goal: add copyright" in result["patch"]
    os.unlink(tf.name)

def test_confirmation_logic():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("print('hi')\n")
        tf.flush()
        # Should require confirmation by default
        result = evolve_file_with_context(tf.name, "add license", dry_run=False, confirm=False)
        assert result["status"] == "confirmation_required"
        # Now actually apply
        result2 = evolve_file_with_context(tf.name, "add license", dry_run=False, confirm=True)
        assert result2["status"] == "applied"
        with open(tf.name) as f:
            content = f.read()
            assert "Evolution goal: add license" in content
    os.unlink(tf.name)

def test_api_evolve_file():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("print('api')\n")
        tf.flush()
        resp = client.post("/memory/evolve_file", json={"file_path": tf.name, "goal": "api test", "dry_run": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert "Evolution goal: api test" in data["patch"]
    os.unlink(tf.name)

def test_evolution_history_storage():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("print('hist')\n")
        tf.flush()
        evolve_file_with_context(tf.name, "history test", dry_run=False, confirm=False)
        # Check DB for history
        with db() as d:
            d.execute("SELECT * FROM code_evolution WHERE file_path=? AND goal=?", (tf.name, "history test"))
            row = d.cur.fetchone()
            assert row is not None
            assert row[1] == tf.name
            assert row[2] == "history test"
            assert row[4] == 0  # applied=False
    os.unlink(tf.name) 