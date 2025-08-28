#!/usr/bin/env python3
"""
Ollama Health Checker for MeRNSTA
Checks if the custom Ollama instance is running and accessible.
"""

import requests
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_config

class OllamaChecker:
    """Check and manage Ollama connectivity for MeRNSTA."""
    
    def __init__(self):
        """Initialize the checker with configuration."""
        self.config = get_config()
        self.network_config = self.config.get('network', {})
        self.ollama_host = self.network_config.get('ollama_host', 'http://127.0.0.1:11434')
        self.tokenizer_config = self.config.get('tokenizer', {})
        self.tokenizer_host = self.tokenizer_config.get('host', self.ollama_host)
        self.tokenizer_model = self.tokenizer_config.get('model', 'tinyllama')
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.ollama_dir = self.project_root / "external" / "ollama"
        self.ollama_binary = self.ollama_dir / "ollama"
        self.start_script = self.project_root / "scripts" / "start_ollama.sh"
        
        self.timeout = 5  # seconds for API calls
        
    def check_ollama_running(self) -> bool:
        """
        Check if Ollama is running and responding.
        
        Returns:
            True if Ollama is running and responding
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def check_tokenizer_endpoints(self) -> Dict[str, bool]:
        """
        Check if tokenizer and detokenizer endpoints are working.
        
        Returns:
            Dict with 'tokenize' and 'detokenize' status
        """
        results = {'tokenize': False, 'detokenize': False}
        
        # Test tokenize endpoint
        try:
            response = requests.post(
                f"{self.tokenizer_host}/api/tokenize",
                json={"model": self.tokenizer_model, "content": "test"},
                timeout=self.timeout
            )
            results['tokenize'] = response.status_code == 200
        except requests.exceptions.RequestException:
            pass
        
        # Test detokenize endpoint
        try:
            response = requests.post(
                f"{self.tokenizer_host}/api/detokenize",
                json={"model": self.tokenizer_model, "tokens": [1, 2, 3]},
                timeout=self.timeout
            )
            results['detokenize'] = response.status_code == 200
        except requests.exceptions.RequestException:
            pass
        
        return results
    
    def check_ollama_binary(self) -> bool:
        """
        Check if the Ollama binary exists and is executable.
        
        Returns:
            True if binary exists and is executable
        """
        return self.ollama_binary.exists() and os.access(self.ollama_binary, os.X_OK)
    
    def check_start_script(self) -> bool:
        """
        Check if the start script exists and is executable.
        
        Returns:
            True if script exists and is executable
        """
        return self.start_script.exists() and os.access(self.start_script, os.X_OK)
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed status of Ollama setup.
        
        Returns:
            Dict with detailed status information
        """
        status = {
            'ollama_running': self.check_ollama_running(),
            'binary_exists': self.check_ollama_binary(),
            'start_script_exists': self.check_start_script(),
            'endpoints': self.check_tokenizer_endpoints(),
            'config': {
                'ollama_host': self.ollama_host,
                'tokenizer_host': self.tokenizer_host,
                'tokenizer_model': self.tokenizer_model
            },
            'paths': {
                'ollama_dir': str(self.ollama_dir),
                'ollama_binary': str(self.ollama_binary),
                'start_script': str(self.start_script)
            }
        }
        
        return status
    
    def print_status(self, detailed: bool = False):
        """
        Print Ollama status information.
        
        Args:
            detailed: If True, print detailed status
        """
        print("🧠 MeRNSTA Ollama Status Check")
        print("=" * 40)
        
        if detailed:
            status = self.get_detailed_status()
            
            print(f"📁 Ollama Directory: {status['paths']['ollama_dir']}")
            print(f"🔧 Ollama Binary: {status['paths']['ollama_binary']}")
            print(f"📜 Start Script: {status['paths']['start_script']}")
            print()
            
            print(f"🔗 Ollama Host: {status['config']['ollama_host']}")
            print(f"🔗 Tokenizer Host: {status['config']['tokenizer_host']}")
            print(f"🤖 Tokenizer Model: {status['config']['tokenizer_model']}")
            print()
            
            print("✅ Status Checks:")
            print(f"  Ollama Running: {'✅' if status['ollama_running'] else '❌'}")
            print(f"  Binary Exists: {'✅' if status['binary_exists'] else '❌'}")
            print(f"  Start Script: {'✅' if status['start_script_exists'] else '❌'}")
            print(f"  Tokenize Endpoint: {'✅' if status['endpoints']['tokenize'] else '❌'}")
            print(f"  Detokenize Endpoint: {'✅' if status['endpoints']['detokenize'] else '❌'}")
        else:
            ollama_running = self.check_ollama_running()
            binary_exists = self.check_ollama_binary()
            
            print(f"Ollama Running: {'✅' if ollama_running else '❌'}")
            print(f"Binary Available: {'✅' if binary_exists else '❌'}")
            
            if ollama_running:
                endpoints = self.check_tokenizer_endpoints()
                print(f"Tokenizer Endpoints: {'✅' if endpoints['tokenize'] and endpoints['detokenize'] else '⚠️'}")
    
    def start_ollama(self) -> bool:
        """
        Start Ollama using the start script.
        
        Returns:
            True if Ollama was started successfully
        """
        # Prefer script if present; otherwise fall back to running the binary directly
        try:
            if self.check_start_script():
                print("🚀 Starting Ollama via script...")
                result = subprocess.run(
                    [str(self.start_script), "start"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print("✅ Ollama started successfully")
                    return True
                else:
                    print(f"❌ Failed to start Ollama (script): {result.stderr}")
                    # fall through to direct binary attempt
            # Direct binary start
            if not self.check_ollama_binary():
                print(f"❌ Ollama binary not found at {self.ollama_binary}")
                return False
            print("🚀 Starting Ollama via binary (background)...")
            logs_dir = self.project_root / "logs"
            pids_dir = self.project_root / "pids"
            logs_dir.mkdir(exist_ok=True)
            pids_dir.mkdir(exist_ok=True)
            log_file = open(logs_dir / "ollama.log", "ab", buffering=0)
            env = os.environ.copy()
            # Ensure default host matches config
            env.setdefault("OLLAMA_HOST", self.ollama_host.replace("http://", ""))
            proc = subprocess.Popen(
                [str(self.ollama_binary), "serve"],
                cwd=str(self.ollama_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True
            )
            # Write PID file
            with open(pids_dir / "ollama.pid", "w") as f:
                f.write(str(proc.pid))
            # Give it a moment and then check
            try:
                import time
                time.sleep(2)
            except Exception:
                pass
            if self.check_ollama_running():
                print("✅ Ollama started successfully (binary)")
                return True
            print("❌ Ollama did not become ready after start")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Ollama start timed out")
            return False
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return False
    
    def get_startup_instructions(self) -> str:
        """
        Get startup instructions for Ollama.
        
        Returns:
            Formatted instructions string
        """
        instructions = """
🧠 MeRNSTA Ollama Setup Instructions
====================================

This repository uses a custom Ollama build with tokenizer/detokenizer support.

📋 Quick Start:
1. Start Ollama: ./scripts/start_ollama.sh start
2. Check status: ./scripts/start_ollama.sh status
3. Run MeRNSTA: python main.py run

🔧 Manual Start (if script doesn't work):
1. cd external/ollama
2. ./ollama serve
3. Wait for "Listening on 127.0.0.1:11434"
4. Run MeRNSTA in another terminal

📊 Status Commands:
- Check if running: ./scripts/start_ollama.sh check
- View logs: ./scripts/start_ollama.sh logs
- Stop Ollama: ./scripts/start_ollama.sh stop

⚠️ Important Notes:
- Ollama must be running before starting MeRNSTA
- The custom build includes /api/tokenize and /api/detokenize endpoints
- Default port: 11434 (configurable in config.yaml)
- Binary location: external/ollama/ollama

🔗 Configuration:
- Ollama host: config.yaml → network.ollama_host
- Tokenizer host: config.yaml → tokenizer.host
- Tokenizer model: config.yaml → tokenizer.model
"""
        return instructions
    
    def validate_setup(self) -> Tuple[bool, str]:
        """
        Validate the complete Ollama setup.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check binary exists
        if not self.check_ollama_binary():
            return False, f"Ollama binary not found at {self.ollama_binary}"
        
        # Check start script exists
        if not self.check_start_script():
            return False, f"Start script not found at {self.start_script}"
        
        # Check if Ollama is running
        if not self.check_ollama_running():
            return False, "Ollama is not running. Run: ./scripts/start_ollama.sh start"
        
        # Check tokenizer endpoints
        endpoints = self.check_tokenizer_endpoints()
        if not endpoints['tokenize'] or not endpoints['detokenize']:
            return False, "Tokenizer endpoints not responding. Check Ollama logs."
        
        return True, "Ollama setup is valid"
    
    def ensure_ollama_running(self) -> bool:
        """
        Ensure Ollama is running, start it if necessary.
        
        Returns:
            True if Ollama is running after this call
        """
        if self.check_ollama_running():
            return True
        
        print("⚠️ Ollama is not running. Attempting to start...")
        return self.start_ollama()


def check_ollama_health() -> bool:
    """
    Quick health check for Ollama.
    
    Returns:
        True if Ollama is healthy
    """
    checker = OllamaChecker()
    return checker.check_ollama_running()


def validate_ollama_setup() -> Tuple[bool, str]:
    """
    Validate Ollama setup and return status.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    checker = OllamaChecker()
    return checker.validate_setup()


def ensure_ollama_ready() -> bool:
    """
    Ensure Ollama is ready for use, start if necessary.
    
    Returns:
        True if Ollama is ready
    """
    checker = OllamaChecker()
    return checker.ensure_ollama_running()


if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="MeRNSTA Ollama Health Checker")
    parser.add_argument("--detailed", action="store_true", help="Show detailed status")
    parser.add_argument("--start", action="store_true", help="Start Ollama if not running")
    parser.add_argument("--validate", action="store_true", help="Validate complete setup")
    parser.add_argument("--instructions", action="store_true", help="Show setup instructions")
    
    args = parser.parse_args()
    
    checker = OllamaChecker()
    
    if args.instructions:
        print(checker.get_startup_instructions())
    elif args.validate:
        is_valid, message = checker.validate_setup()
        if is_valid:
            print("✅ Ollama setup is valid")
        else:
            print(f"❌ Ollama setup issue: {message}")
            sys.exit(1)
    elif args.start:
        if checker.ensure_ollama_running():
            print("✅ Ollama is ready")
        else:
            print("❌ Failed to start Ollama")
            sys.exit(1)
    else:
        checker.print_status(detailed=args.detailed)
