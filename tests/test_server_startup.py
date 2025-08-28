import pytest
import socket
import subprocess
import time
from config.settings import api_port, port_retry_attempts
from monitoring.logger import get_logger

def occupy_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    return s

def test_server_starts_on_free_port():
    # Should start on api_port if free
    process = subprocess.Popen([
        "python3", "start_enterprise.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    # Check health endpoint
    import requests
    resp = requests.get(f"http://localhost:{api_port}/health")
    assert resp.status_code == 200
    process.terminate()
    process.wait()

def test_server_automatic_port_selection_on_conflict():
    # Occupy api_port, server should start on next available port
    s = occupy_port(api_port)
    logger = get_logger("test")
    process = subprocess.Popen([
        "python3", "start_enterprise.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    # Try all retry ports
    found = False
    for i in range(port_retry_attempts):
        port = api_port + i
        try:
            import requests
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                found = True
                break
        except Exception:
            continue
    assert found, "Server did not start on any available port"
    process.terminate()
    process.wait()
    s.close()

def test_logging_of_port_conflicts(caplog):
    # Occupy api_port, check logs for port conflict
    s = occupy_port(api_port)
    logger = get_logger("test")
    process = subprocess.Popen([
        "python3", "start_enterprise.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    # Read logs
    with open("logs/mernsta.log") as f:
        logs = f.read()
    assert "in use" in logs or "trying" in logs
    process.terminate()
    process.wait()
    s.close() 