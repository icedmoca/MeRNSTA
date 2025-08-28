#!/usr/bin/env python3
"""
🚀 MeRNSTA OS Integration Runner - Phase 30
Daemonized runtime that transforms MeRNSTA into a persistent OS-level cognitive service.

This module provides the core daemon functionality that:
- Loads full MeRNSTA architecture
- Loops agent lifecycle, memory analysis, planning, and reflection  
- Supports both headless and interactive control modes
- Supports background scheduling (reflection every 6 hours, planning every 30 min)
- Detects local app context (active window, command usage, shell history)
"""

import asyncio
import argparse
import signal
import sys
import os
import time
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import psutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem


@dataclass
class IntegrationState:
    """State information for the integration runner."""
    start_time: datetime
    last_reflection: Optional[datetime] = None
    last_planning: Optional[datetime] = None
    last_memory_consolidation: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    last_context_detection: Optional[datetime] = None
    reflection_count: int = 0
    planning_count: int = 0
    context_events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.context_events is None:
            self.context_events = []


class ContextDetector:
    """Detects local application context and system state."""
    
    def __init__(self, max_context_size: int = 1000):
        self.max_context_size = max_context_size
        self.last_active_window = None
        self.recent_commands = []
        
    def get_active_window(self) -> Optional[str]:
        """Get the currently active window title (Linux/WSL)."""
        try:
            # For WSL environments, we might not have direct window access
            # This is a placeholder implementation
            if os.name == 'posix':
                # Try to get active window on Linux
                result = subprocess.run(['xdotool', 'getactivewindow', 'getwindowname'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
    
    def get_shell_history_recent(self, lines: int = 10) -> List[str]:
        """Get recent shell command history."""
        try:
            history_file = os.path.expanduser("~/.bash_history")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    commands = f.readlines()
                    return [cmd.strip() for cmd in commands[-lines:] if cmd.strip()]
        except Exception:
            pass
        return []
    
    def detect_context_changes(self) -> Dict[str, Any]:
        """Detect changes in system context."""
        context_event = {
            'timestamp': datetime.now().isoformat(),
            'changes': []
        }
        
        # Check active window changes
        current_window = self.get_active_window()
        if current_window and current_window != self.last_active_window:
            context_event['changes'].append({
                'type': 'window_change',
                'from': self.last_active_window,
                'to': current_window
            })
            self.last_active_window = current_window
        
        # Check for new shell commands
        recent_commands = self.get_shell_history_recent(5)
        new_commands = [cmd for cmd in recent_commands if cmd not in self.recent_commands]
        if new_commands:
            context_event['changes'].append({
                'type': 'shell_commands',
                'commands': new_commands
            })
            self.recent_commands = recent_commands
        
        # Add system resource info
        context_event['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        return context_event if context_event['changes'] else None


class MeRNSTAIntegrationRunner:
    """
    Main integration runner that orchestrates the MeRNSTA cognitive system 
    as a persistent OS-level service.
    """
    
    def __init__(self, mode: str = "daemon"):
        self.mode = mode
        self.config = get_config()
        self.os_config = self.config.get('os_integration', {})
        self.intervals = self.os_config.get('intervals', {})
        
        # Initialize state
        self.state = IntegrationState(start_time=datetime.now())
        self.running = False
        self.shutdown_requested = False
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize context detector
        context_config = self.os_config.get('context', {})
        self.context_detector = ContextDetector(
            max_context_size=context_config.get('context_memory_size', 1000)
        ) if context_config.get('detect_active_window', True) else None
        
        # Initialize cognitive system
        self.logger.info("Initializing MeRNSTA cognitive system...")
        self.cognitive_system = None
        self._init_cognitive_system()
        
        # Background task tracking
        self.background_tasks = []
        
        self.logger.info(f"MeRNSTA Integration Runner initialized in {mode} mode")
    
    def _setup_logging(self):
        """Setup logging configuration for OS integration."""
        log_config = self.os_config.get('logging', {})
        
        if not log_config.get('enabled', True):
            logging.disable(logging.CRITICAL)
            return
        
        # Create output directory if it doesn't exist
        log_file = log_config.get('log_file', 'output/os_bridge.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup rotating file handler
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_size_mb', 50) * 1024 * 1024,
            backupCount=log_config.get('backup_count', 5)
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('mernsta.integration')
        self.logger.setLevel(getattr(logging, log_config.get('log_level', 'INFO')))
        self.logger.addHandler(handler)
        
        # Also log to console in interactive mode
        if self.mode == 'interactive':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _init_cognitive_system(self):
        """Initialize the Phase 2 cognitive system."""
        try:
            self.cognitive_system = Phase2AutonomousCognitiveSystem()
            self.logger.info("Successfully initialized Phase2AutonomousCognitiveSystem")
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive system: {e}")
            if self.mode != 'headless':
                raise
    
    def _should_run_task(self, task_name: str, last_run: Optional[datetime]) -> bool:
        """Check if a background task should be run based on its interval."""
        if last_run is None:
            return True
        
        interval = self.intervals.get(task_name, 3600)  # Default 1 hour
        return (datetime.now() - last_run).total_seconds() >= interval
    
    async def _run_reflection(self):
        """Run reflection process."""
        self.logger.info("Starting reflection process...")
        try:
            if self.cognitive_system:
                # Trigger reflection through the cognitive system
                result = self.cognitive_system.trigger_autonomous_reflection()
                self.state.last_reflection = datetime.now()
                self.state.reflection_count += 1
                self.logger.info(f"Reflection completed successfully (count: {self.state.reflection_count})")
                return result
        except Exception as e:
            self.logger.error(f"Reflection process failed: {e}")
            return None
    
    async def _run_planning(self):
        """Run planning process."""
        self.logger.info("Starting planning process...")
        try:
            if self.cognitive_system:
                # Trigger planning through the cognitive system
                result = self.cognitive_system.generate_autonomous_goals()
                self.state.last_planning = datetime.now()
                self.state.planning_count += 1
                self.logger.info(f"Planning completed successfully (count: {self.state.planning_count})")
                return result
        except Exception as e:
            self.logger.error(f"Planning process failed: {e}")
            return None
    
    async def _run_memory_consolidation(self):
        """Run memory consolidation process."""
        self.logger.info("Starting memory consolidation...")
        try:
            if self.cognitive_system:
                # Trigger memory consolidation
                result = self.cognitive_system.consolidate_memories()
                self.state.last_memory_consolidation = datetime.now()
                self.logger.info("Memory consolidation completed successfully")
                return result
        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {e}")
            return None
    
    async def _run_health_check(self):
        """Run system health check."""
        try:
            health_info = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.state.start_time).total_seconds(),
                'reflection_count': self.state.reflection_count,
                'planning_count': self.state.planning_count,
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'cognitive_system_status': 'active' if self.cognitive_system else 'inactive'
            }
            
            self.state.last_health_check = datetime.now()
            self.logger.debug(f"Health check: {health_info}")
            return health_info
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return None
    
    async def _run_context_detection(self):
        """Run context detection."""
        if not self.context_detector:
            return None
            
        try:
            context_event = self.context_detector.detect_context_changes()
            if context_event:
                # Add to context events (with size limit)
                self.state.context_events.append(context_event)
                if len(self.state.context_events) > self.context_detector.max_context_size:
                    self.state.context_events.pop(0)
                
                self.logger.debug(f"Context change detected: {context_event}")
                
                # Optionally feed context to cognitive system
                if self.cognitive_system and context_event['changes']:
                    context_summary = f"System context change: {json.dumps(context_event['changes'])}"
                    self.cognitive_system.process_input_with_full_cognition(
                        f"[CONTEXT] {context_summary}"
                    )
            
            self.state.last_context_detection = datetime.now()
            return context_event
        except Exception as e:
            self.logger.error(f"Context detection failed: {e}")
            return None
    
    def _save_state(self):
        """Save current state to disk."""
        try:
            persistence_config = self.os_config.get('persistence', {})
            state_file = persistence_config.get('state_file', 'output/os_integration_state.json')
            
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            # Convert state to JSON-serializable format
            state_dict = asdict(self.state)
            state_dict['start_time'] = self.state.start_time.isoformat()
            if self.state.last_reflection:
                state_dict['last_reflection'] = self.state.last_reflection.isoformat()
            if self.state.last_planning:
                state_dict['last_planning'] = self.state.last_planning.isoformat()
            if self.state.last_memory_consolidation:
                state_dict['last_memory_consolidation'] = self.state.last_memory_consolidation.isoformat()
            if self.state.last_health_check:
                state_dict['last_health_check'] = self.state.last_health_check.isoformat()
            if self.state.last_context_detection:
                state_dict['last_context_detection'] = self.state.last_context_detection.isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from disk if available."""
        try:
            persistence_config = self.os_config.get('persistence', {})
            if not persistence_config.get('resume_on_restart', True):
                return
                
            state_file = persistence_config.get('state_file', 'output/os_integration_state.json')
            
            if not os.path.exists(state_file):
                return
                
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Restore timestamps
            if 'last_reflection' in state_dict and state_dict['last_reflection']:
                self.state.last_reflection = datetime.fromisoformat(state_dict['last_reflection'])
            if 'last_planning' in state_dict and state_dict['last_planning']:
                self.state.last_planning = datetime.fromisoformat(state_dict['last_planning'])
            if 'last_memory_consolidation' in state_dict and state_dict['last_memory_consolidation']:
                self.state.last_memory_consolidation = datetime.fromisoformat(state_dict['last_memory_consolidation'])
            if 'last_health_check' in state_dict and state_dict['last_health_check']:
                self.state.last_health_check = datetime.fromisoformat(state_dict['last_health_check'])
            if 'last_context_detection' in state_dict and state_dict['last_context_detection']:
                self.state.last_context_detection = datetime.fromisoformat(state_dict['last_context_detection'])
            
            # Restore counts
            self.state.reflection_count = state_dict.get('reflection_count', 0)
            self.state.planning_count = state_dict.get('planning_count', 0)
            self.state.context_events = state_dict.get('context_events', [])
            
            self.logger.info(f"Restored state from {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    async def _background_task_loop(self):
        """Main background task loop."""
        save_interval = self.os_config.get('persistence', {}).get('save_interval', 300)
        last_save = datetime.now()
        
        while self.running and not self.shutdown_requested:
            try:
                # Check and run scheduled tasks
                tasks_to_run = []
                
                if self._should_run_task('reflection', self.state.last_reflection):
                    tasks_to_run.append(self._run_reflection())
                
                if self._should_run_task('planning', self.state.last_planning):
                    tasks_to_run.append(self._run_planning())
                
                if self._should_run_task('memory_consolidation', self.state.last_memory_consolidation):
                    tasks_to_run.append(self._run_memory_consolidation())
                
                if self._should_run_task('health_check', self.state.last_health_check):
                    tasks_to_run.append(self._run_health_check())
                
                if self._should_run_task('context_detection', self.state.last_context_detection):
                    tasks_to_run.append(self._run_context_detection())
                
                # Run tasks concurrently
                if tasks_to_run:
                    await asyncio.gather(*tasks_to_run, return_exceptions=True)
                
                # Save state periodically
                if (datetime.now() - last_save).total_seconds() >= save_interval:
                    self._save_state()
                    last_save = datetime.now()
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in background task loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """Start the integration runner."""
        self.logger.info(f"Starting MeRNSTA Integration Runner in {self.mode} mode")
        
        # Load previous state if available
        self._load_state()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.running = True
        
        try:
            if self.mode in ['daemon', 'headless']:
                # Run background task loop
                await self._background_task_loop()
            elif self.mode == 'interactive':
                # Run interactive mode with background tasks
                await self._run_interactive_mode()
            elif self.mode == 'bridge_only':
                # Just run background tasks without full cognitive loop
                await self._background_task_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error in runner: {e}")
        finally:
            await self.shutdown()
    
    async def _run_interactive_mode(self):
        """Run in interactive mode with console interface."""
        print("\n🧠 MeRNSTA OS Integration - Interactive Mode")
        print("="*50)
        print("Commands: status, reflection, planning, context, quit")
        print("Background tasks are running automatically.")
        print("-"*50)
        
        # Start background tasks
        background_task = asyncio.create_task(self._background_task_loop())
        
        try:
            while self.running and not self.shutdown_requested:
                try:
                    user_input = input("\nmernsta-os> ").strip().lower()
                    
                    if user_input in ['quit', 'exit', 'q']:
                        break
                    elif user_input == 'status':
                        await self._print_status()
                    elif user_input == 'reflection':
                        result = await self._run_reflection()
                        print(f"Reflection completed: {result is not None}")
                    elif user_input == 'planning':
                        result = await self._run_planning()
                        print(f"Planning completed: {result is not None}")
                    elif user_input == 'context':
                        self._print_context_info()
                    elif user_input == 'help':
                        print("Available commands: status, reflection, planning, context, quit")
                    else:
                        print(f"Unknown command: {user_input}")
                        
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
        finally:
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
    
    async def _print_status(self):
        """Print current system status."""
        uptime = datetime.now() - self.state.start_time
        print(f"\n📊 MeRNSTA OS Integration Status")
        print(f"Uptime: {uptime}")
        print(f"Mode: {self.mode}")
        print(f"Reflections: {self.state.reflection_count}")
        print(f"Planning cycles: {self.state.planning_count}")
        print(f"Last reflection: {self.state.last_reflection or 'Never'}")
        print(f"Last planning: {self.state.last_planning or 'Never'}")
        print(f"Context events: {len(self.state.context_events)}")
        print(f"Cognitive system: {'Active' if self.cognitive_system else 'Inactive'}")
    
    def _print_context_info(self):
        """Print recent context information."""
        print(f"\n🔍 Context Information")
        print(f"Recent context events: {len(self.state.context_events)}")
        
        if self.state.context_events:
            for event in self.state.context_events[-5:]:  # Show last 5 events
                print(f"  {event['timestamp']}: {len(event.get('changes', []))} changes")
    
    async def shutdown(self):
        """Graceful shutdown of the integration runner."""
        self.logger.info("Shutting down MeRNSTA Integration Runner...")
        
        self.running = False
        self.shutdown_requested = True
        
        # Save final state
        self._save_state()
        
        # Shutdown cognitive system
        if self.cognitive_system:
            try:
                # If the cognitive system has a shutdown method, call it
                if hasattr(self.cognitive_system, 'shutdown'):
                    self.cognitive_system.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down cognitive system: {e}")
        
        self.logger.info("MeRNSTA Integration Runner shutdown complete")


def main():
    """Main entry point for the integration runner."""
    parser = argparse.ArgumentParser(description='MeRNSTA OS Integration Runner')
    parser.add_argument('--mode', choices=['daemon', 'interactive', 'headless', 'bridge_only'],
                      default='daemon', help='Runtime mode')
    parser.add_argument('--config', help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    # Initialize and run the integration runner
    runner = MeRNSTAIntegrationRunner(mode=args.mode)
    
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()