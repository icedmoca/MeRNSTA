#!/usr/bin/env python3
"""
🚀 MeRNSTA Unified Runner - Full AGI Mode
Unified runner that starts all MeRNSTA components in one command for complete AGI system.

This module provides a single entry point that orchestrates:
- Web Chat Interface (FastAPI + HTML/JS frontend)
- REST API Server (for external integrations)
- Background Cognitive Tasks (reflection, planning, memory consolidation)
- Multi-Agent System (all 23 specialized agents)
- Autonomous OS Integration (context detection, persistent service)
- Enterprise Features (optional: Celery, Redis, monitoring)

The unified runner ensures proper initialization order, graceful shutdown,
and comprehensive health monitoring across all components.
"""

import asyncio
import signal
import sys
import os
import time
import json
import logging
import threading
import subprocess
import psutil
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uvicorn
from contextlib import asynccontextmanager

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem


@dataclass
class UnifiedSystemState:
    """State tracking for the unified system."""
    start_time: datetime
    web_server_started: bool = False
    api_server_started: bool = False
    cognitive_system_started: bool = False
    background_tasks_started: bool = False
    agents_initialized: bool = False
    enterprise_features_started: bool = False
    graceful_shutdown_requested: bool = False
    active_background_tasks: Set[str] = None
    
    def __post_init__(self):
        if self.active_background_tasks is None:
            self.active_background_tasks = set()


class MeRNSTAUnifiedRunner:
    """
    Unified runner that orchestrates all MeRNSTA components in a single process.
    
    Features:
    - Concurrent startup of web UI, API server, and background tasks
    - Proper initialization dependency management
    - Graceful shutdown with cleanup
    - Health monitoring and status reporting
    - Cross-platform compatibility
    - Enterprise feature integration (optional)
    """
    
    def __init__(self, 
                 web_port: int = 8000,
                 api_port: int = 8001,
                 enable_web: bool = True,
                 enable_api: bool = True,
                 enable_background: bool = True,
                 enable_agents: bool = True,
                 enable_enterprise: bool = False,
                 debug: bool = False):
        
        self.web_port = web_port
        self.api_port = api_port
        self.enable_web = enable_web
        self.enable_api = enable_api
        self.enable_background = enable_background
        self.enable_agents = enable_agents
        self.enable_enterprise = enable_enterprise
        self.debug = debug
        
        # Load configuration
        self.config = get_config()
        self.os_config = self.config.get('os_integration', {})
        
        # Initialize state
        self.state = UnifiedSystemState(start_time=datetime.now())
        self.running = False
        self.shutdown_requested = False
        
        # Component instances
        self.cognitive_system = None
        self.web_server = None
        self.api_server = None
        self.integration_runner = None
        self.background_tasks = []
        self.background_task_handles = []
        
        # Process tracking for cleanup
        self.child_processes: List[subprocess.Popen] = []
        self.process_group_id = None
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers
        self._setup_signal_handlers()
        
        # Register cleanup at exit
        atexit.register(self._emergency_cleanup)
        
        self.logger.info(f"MeRNSTAUnifiedRunner initialized with web_port={web_port}, api_port={api_port}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the unified system."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/unified_system.log'),
                logging.FileHandler('logs/mernsta.log')  # Compatibility with existing monitoring
            ]
        )
        
        self.logger = logging.getLogger('mernsta.unified')
        self.logger.setLevel(log_level)
        
        # Suppress some verbose loggers in production
        if not self.debug:
            logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
            logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
            logging.getLogger('asyncio').setLevel(logging.WARNING)

    async def _preflight_cleanup_existing_processes(self):
        """Terminate stale MeRNSTA processes from previous runs.

        We only target safe signatures:
        - python ... main.py run|web|api|integration
        - uvicorn serving web.main:app or api.system_bridge
        - celery workers for tasks.task_queue started by MeRNSTA (optional)
        """
        try:
            # Allow disabling via config (defaults to enabled to avoid port collisions)
            if not self.config.get('system', {}).get('kill_stale_processes', True):
                self.logger.info("Preflight: kill_stale_processes disabled by config")
                return

            project_root = str(Path(__file__).resolve().parents[1])  # repo root
            current_pid = os.getpid()
            killed: int = 0

            # First, terminate processes by signature and path match
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    pid = proc.info.get('pid')
                    if not pid or pid == current_pid:
                        continue
                    cmdline = proc.info.get('cmdline') or []
                    if not isinstance(cmdline, list):
                        continue

                    cmd = ' '.join(cmdline)
                    # Identify MeRNSTA processes
                    is_python = any(x.endswith('python') or 'python' in x for x in cmdline)
                    is_main = 'main.py' in cmd
                    is_modestr = any(m in cmd for m in [' run', ' web', ' api', ' integration'])
                    is_uvicorn_web = ('uvicorn' in cmd and 'web.main:app' in cmd) or ('uvicorn' in cmd and 'web.main' in cmd)
                    is_uvicorn_api = ('uvicorn' in cmd and 'api.system_bridge' in cmd) or ('SystemBridgeAPI' in cmd)
                    is_celery = ('celery' in cmd and 'tasks.task_queue' in cmd)

                    # Ensure the process is related to this repo (path match in cmd or cwd)
                    related = project_root in cmd
                    if not related:
                        try:
                            p_cwd = proc.cwd()
                            if p_cwd and project_root in p_cwd:
                                related = True
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            pass

                    if related and ((is_python and is_main and is_modestr) or is_uvicorn_web or is_uvicorn_api or is_celery):
                        # Double-check: don't kill our parent/children already tracked
                        if pid in [p.pid for p in self.child_processes if p and p.poll() is None]:
                            continue
                        try:
                            self.logger.info(f"Preflight: terminating stale MeRNSTA process PID={pid} CMD='{cmd[:200]}'")
                            proc.terminate()
                            proc.wait(timeout=3)
                        except (psutil.TimeoutExpired):
                            self.logger.info(f"Preflight: killing stubborn process PID={pid}")
                            proc.kill()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Second, free configured ports if still occupied (web/api) by uvicorn for our apps
            try:
                listening_pids: set[int] = set()
                for c in psutil.net_connections(kind='inet'):
                    if c.status == psutil.CONN_LISTEN and c.laddr:
                        if c.laddr.port in {self.web_port, self.api_port}:
                            if c.pid and c.pid != current_pid:
                                try:
                                    p = psutil.Process(c.pid)
                                    pcmd_list = p.cmdline() or []
                                    pcmd = ' '.join(pcmd_list)
                                    if 'uvicorn' in pcmd and ('web.main' in pcmd or 'api.system_bridge' in pcmd):
                                        listening_pids.add(c.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
                for pid in listening_pids:
                    try:
                        p = psutil.Process(pid)
                        # Avoid killing ourselves/children
                        if pid in [p_.pid for p_ in self.child_processes if p_ and p_.poll() is None]:
                            continue
                        self.logger.info(f"Preflight: freeing port by terminating PID={pid} listening on {self.web_port}/{self.api_port}")
                        p.terminate()
                        try:
                            p.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            p.kill()
                        killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                self.logger.debug(f"Preflight: port scan failed or not permitted: {e}")

            if killed:
                self.logger.info(f"Preflight: terminated {killed} stale MeRNSTA process(es)")
            else:
                self.logger.info("Preflight: no stale MeRNSTA processes found")
        except Exception as e:
            self.logger.warning(f"Preflight cleanup failed: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            
            # Create a new event loop for shutdown if we're not in the main thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule shutdown in the running loop
                    loop.create_task(self.shutdown())
                else:
                    # Run shutdown in new loop
                    asyncio.run(self.shutdown())
            except RuntimeError:
                # We're likely in a different thread, use thread-safe approach
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle SIGQUIT on Unix systems
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)
    
    def _emergency_cleanup(self):
        """Emergency cleanup function called at exit."""
        if not hasattr(self, 'logger'):
            return
            
        try:
            self.logger.info("🚨 Emergency cleanup initiated...")
            
            # Kill all child processes
            for proc in self.child_processes:
                try:
                    if proc.poll() is None:  # Process is still running
                        self.logger.info(f"Killing child process {proc.pid}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except:
                    pass
            
            # Kill any remaining MeRNSTA processes
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if (isinstance(cmdline, list) and 
                            any('main.py' in arg for arg in cmdline) and
                            any('python' in arg for arg in cmdline)):
                            self.logger.info(f"Killing remaining MeRNSTA process {proc.info['pid']}")
                            proc.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except:
                pass
                
            self.logger.info("✅ Emergency cleanup completed")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in emergency cleanup: {e}")
    
    async def _initialize_cognitive_system(self):
        """Initialize the core cognitive system."""
        self.logger.info("Initializing Phase 2 Autonomous Cognitive System...")
        
        try:
            self.cognitive_system = Phase2AutonomousCognitiveSystem()
            self.state.cognitive_system_started = True
            self.logger.info("✅ Cognitive system initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize cognitive system: {e}")
            return False
    
    async def _initialize_agents(self):
        """Initialize the multi-agent system."""
        if not self.enable_agents:
            self.logger.info("Agents disabled - skipping initialization")
            return True
            
        self.logger.info("Initializing multi-agent cognitive system...")
        
        try:
            from agents.registry import get_agent_registry
            
            registry = get_agent_registry()
            status = registry.get_system_status()
            
            self.state.agents_initialized = True
            self.logger.info(f"✅ Initialized {status['total_agents']} agents: {status['agent_names']}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agents: {e}")
            return False
    
    async def _start_web_server(self):
        """Start the web chat interface server."""
        if not self.enable_web:
            self.logger.info("Web server disabled - skipping startup")
            return True
            
        self.logger.info(f"Starting web server on port {self.web_port}...")
        
        try:
            from web.main import app as web_app
            
            # Create server config
            config = uvicorn.Config(
                web_app,
                host='0.0.0.0',
                port=self.web_port,
                log_level='info' if not self.debug else 'debug',
                access_log=self.debug,
                loop='asyncio'
            )
            
            # Start server in background
            server = uvicorn.Server(config)
            
            # Run server in a separate task
            web_task = asyncio.create_task(server.serve())
            self.background_task_handles.append(web_task)
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            self.web_server = server
            self.state.web_server_started = True
            self.logger.info(f"✅ Web server started: http://0.0.0.0:{self.web_port}")
            self.logger.info(f"   💬 Chat interface: http://localhost:{self.web_port}/chat")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start web server: {e}")
            return False
    
    async def _start_api_server(self):
        """Start the REST API server."""
        if not self.enable_api:
            self.logger.info("API server disabled - skipping startup")
            return True
            
        self.logger.info(f"Starting API server on port {self.api_port}...")
        
        try:
            from api.system_bridge import SystemBridgeAPI
            
            # Create API instance
            api = SystemBridgeAPI()
            
            # Create server config
            config = uvicorn.Config(
                api.app,
                host='127.0.0.1',  # API server on localhost only for security
                port=self.api_port,
                log_level='info' if not self.debug else 'debug',
                access_log=self.debug,
                loop='asyncio'
            )
            
            # Start server in background
            server = uvicorn.Server(config)
            
            # Run server in a separate task
            api_task = asyncio.create_task(server.serve())
            self.background_task_handles.append(api_task)
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            self.api_server = server
            self.state.api_server_started = True
            self.logger.info(f"✅ API server started: http://127.0.0.1:{self.api_port}")
            self.logger.info(f"   📖 API docs: http://127.0.0.1:{self.api_port}/docs")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start API server: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start all background cognitive tasks."""
        if not self.enable_background:
            self.logger.info("Background tasks disabled - skipping startup")
            return True
            
        self.logger.info("Starting background cognitive tasks...")
        
        try:
            from system.integration_runner import MeRNSTAIntegrationRunner
            
            # Create integration runner in headless mode for background tasks
            self.integration_runner = MeRNSTAIntegrationRunner(mode='headless')
            
            # Start background task loop
            integration_task = asyncio.create_task(self.integration_runner._background_task_loop())
            self.background_task_handles.append(integration_task)
            
            self.state.background_tasks_started = True
            self.state.active_background_tasks.update([
                'reflection', 'planning', 'memory_consolidation', 
                'health_check', 'context_detection'
            ])
            
            self.logger.info("✅ Background tasks started:")
            self.logger.info("   🔮 Reflection cycles (every 6 hours)")
            self.logger.info("   📋 Planning cycles (every 30 minutes)")  
            self.logger.info("   🧠 Memory consolidation (every hour)")
            self.logger.info("   💓 Health monitoring (every 5 minutes)")
            self.logger.info("   👁️ Context detection (every minute)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start background tasks: {e}")
            return False
    
    async def _start_enterprise_features(self):
        """Start enterprise features if enabled."""
        if not self.enable_enterprise:
            self.logger.info("Enterprise features disabled - skipping startup")
            return True
            
        self.logger.info("Starting enterprise features...")
        
        try:
            # Check if Redis is available for enterprise features
            try:
                import redis
                r = redis.from_url(self.config.get('network', {}).get('redis_url', 'redis://localhost:6379/0'))
                r.ping()
                self.logger.info("✅ Redis connection verified")
            except Exception as e:
                self.logger.warning(f"⚠️ Redis not available for enterprise features: {e}")
                return True  # Continue without enterprise features
            
            # Start Celery workers if available
            try:
                from tasks.task_queue import celery_app
                
                # Start background Celery worker and beat
                worker_cmd = [
                    sys.executable, "-m", "celery", "-A", "tasks.task_queue", 
                    "worker", "--loglevel=info", "--detach"
                ]
                beat_cmd = [
                    sys.executable, "-m", "celery", "-A", "tasks.task_queue", 
                    "beat", "--loglevel=info", "--detach"
                ]
                
                # Start worker and beat in background
                worker_proc = subprocess.Popen(worker_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                beat_proc = subprocess.Popen(beat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Track processes for cleanup
                self.child_processes.extend([worker_proc, beat_proc])
                
                self.state.enterprise_features_started = True
                self.logger.info("✅ Enterprise features started:")
                self.logger.info("   🔄 Celery task queue")
                self.logger.info("   ⏰ Celery beat scheduler")
                self.logger.info("   📊 Prometheus metrics")
                
            except ImportError:
                self.logger.warning("⚠️ Celery not available - continuing without task queue")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start enterprise features: {e}")
            return False
    
    async def _run_health_monitor(self):
        """Run continuous health monitoring."""
        while self.running and not self.shutdown_requested:
            try:
                # Basic health check
                health_info = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': (datetime.now() - self.state.start_time).total_seconds(),
                    'components': {
                        'cognitive_system': self.state.cognitive_system_started,
                        'web_server': self.state.web_server_started,
                        'api_server': self.state.api_server_started,
                        'background_tasks': self.state.background_tasks_started,
                        'agents': self.state.agents_initialized,
                        'enterprise': self.state.enterprise_features_started
                    },
                    'system': {
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
                    }
                }
                
                # Log health info periodically (every 5 minutes)
                if int(health_info['uptime_seconds']) % 300 == 0:
                    self.logger.debug(f"Health: {health_info}")
                
                # Check for any failed components
                failed_components = [
                    name for name, status in health_info['components'].items()
                    if not status and getattr(self, f'enable_{name.replace("_server", "").replace("_tasks", "_background").replace("agents", "agents")}', True)
                ]
                
                if failed_components:
                    self.logger.warning(f"⚠️ Failed components detected: {failed_components}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the unified MeRNSTA system."""
        self.logger.info("🚀 Starting MeRNSTA Unified Full AGI System")
        self.logger.info("=" * 60)
        
        self.running = True
        
        try:
            # Preflight: terminate any stale MeRNSTA processes from previous runs
            await self._preflight_cleanup_existing_processes()

            # Phase 1: Core system initialization
            self.logger.info("📋 Phase 1: Core System Initialization")
            if not await self._initialize_cognitive_system():
                raise RuntimeError("Failed to initialize cognitive system")
            
            if not await self._initialize_agents():
                raise RuntimeError("Failed to initialize agents")
            
            # Phase 2: Server startup
            self.logger.info("📋 Phase 2: Server Startup")
            
            # Start servers concurrently
            server_tasks = []
            if self.enable_web:
                server_tasks.append(self._start_web_server())
            if self.enable_api:
                server_tasks.append(self._start_api_server())
            
            if server_tasks:
                server_results = await asyncio.gather(*server_tasks, return_exceptions=True)
                for i, result in enumerate(server_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Server startup failed: {result}")
                        raise result
                    elif not result:
                        raise RuntimeError(f"Server {i} failed to start")
            
            # Phase 3: Background services
            self.logger.info("📋 Phase 3: Background Services")
            if not await self._start_background_tasks():
                self.logger.warning("⚠️ Background tasks failed to start - continuing anyway")
            
            # Phase 4: Enterprise features (optional)
            if self.enable_enterprise:
                self.logger.info("📋 Phase 4: Enterprise Features")
                if not await self._start_enterprise_features():
                    self.logger.warning("⚠️ Enterprise features failed to start - continuing anyway")
            
            # Phase 5: Health monitoring
            self.logger.info("📋 Phase 5: Health Monitoring")
            health_task = asyncio.create_task(self._run_health_monitor())
            self.background_task_handles.append(health_task)
            
            # System ready!
            self._print_startup_summary()
            
            # Keep running until shutdown
            await self._run_main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown requested by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"❌ Fatal error during startup: {e}")
            raise
        finally:
            await self.shutdown()
    
    def _print_startup_summary(self):
        """Print a comprehensive startup summary."""
        uptime = datetime.now() - self.state.start_time
        
        print("\n" + "=" * 70)
        print("🎉 MeRNSTA Unified Full AGI System - READY!")
        print("=" * 70)
        print(f"⏱️  Startup time: {uptime.total_seconds():.2f} seconds")
        print()
        
        if self.state.web_server_started:
            print(f"💬 Web Chat Interface:")
            print(f"   http://localhost:{self.web_port}/chat")
            print(f"   http://localhost:{self.web_port}/health")
        
        if self.state.api_server_started:
            print(f"🔌 REST API Server:")
            print(f"   http://127.0.0.1:{self.api_port}/docs")
            print(f"   http://127.0.0.1:{self.api_port}/health")
        
        if self.state.agents_initialized:
            try:
                from agents.registry import get_agent_registry
                registry = get_agent_registry()
                status = registry.get_system_status()
                print(f"🤖 Cognitive Agents: {status['total_agents']} active")
            except:
                print(f"🤖 Cognitive Agents: Active")
        
        if self.state.background_tasks_started:
            print(f"🔄 Background Tasks: {len(self.state.active_background_tasks)} active")
        
        if self.state.enterprise_features_started:
            print(f"🏢 Enterprise Features: Active")
        
        print()
        print("📊 System Health:")
        try:
            print(f"   CPU: {psutil.cpu_percent()}%")
            print(f"   Memory: {psutil.virtual_memory().percent}%")
            print(f"   Disk: {psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent}%")
        except:
            print("   System metrics unavailable")
        
        print()
        print("🛑 To stop the system: Press Ctrl+C")
        print("=" * 70)
        print()
    
    async def _run_main_loop(self):
        """Main loop - keep the system running."""
        while self.running and not self.shutdown_requested:
            try:
                await asyncio.sleep(1)
                
                # Check if any critical tasks have failed
                failed_tasks = [
                    task for task in self.background_task_handles 
                    if task.done() and task.exception()
                ]
                
                if failed_tasks:
                    self.logger.warning(f"⚠️ {len(failed_tasks)} background tasks have failed")
                    for task in failed_tasks:
                        self.logger.error(f"Task failure: {task.exception()}")
                        
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def shutdown(self):
        """Graceful shutdown of all components."""
        if self.state.graceful_shutdown_requested:
            self.logger.info("Shutdown already in progress...")
            return
            
        self.state.graceful_shutdown_requested = True
        self.running = False
        
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Cancel all background tasks
            self.logger.info("Stopping background tasks...")
            for task in self.background_task_handles:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish with timeout
            if self.background_task_handles:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.background_task_handles, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ Some background tasks did not shut down gracefully")
            
            # Stop integration runner
            if self.integration_runner:
                await self.integration_runner.shutdown()
            
            # Stop servers
            if self.web_server:
                self.logger.info("Stopping web server...")
                self.web_server.should_exit = True
            
            if self.api_server:
                self.logger.info("Stopping API server...")
                self.api_server.should_exit = True
            
            # Shutdown cognitive system
            if self.cognitive_system:
                self.logger.info("Shutting down cognitive system...")
                if hasattr(self.cognitive_system, 'shutdown'):
                    self.cognitive_system.shutdown()
            
            # Stop enterprise features
            if self.state.enterprise_features_started:
                self.logger.info("Stopping enterprise features...")
                # Send shutdown signal to Celery workers
                try:
                    subprocess.run([
                        sys.executable, "-m", "celery", "-A", "tasks.task_queue", 
                        "control", "shutdown"
                    ], timeout=5, capture_output=True)
                except:
                    pass  # Ignore errors during shutdown
            
            # Kill all child processes
            self.logger.info("Stopping child processes...")
            for proc in self.child_processes:
                try:
                    if proc.poll() is None:  # Process is still running
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except:
                    pass
            
            uptime = datetime.now() - self.state.start_time
            self.logger.info(f"✅ MeRNSTA Unified System shutdown complete (uptime: {uptime})")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'running': self.running,
            'uptime_seconds': (datetime.now() - self.state.start_time).total_seconds(),
            'components': {
                'cognitive_system': self.state.cognitive_system_started,
                'web_server': self.state.web_server_started,
                'api_server': self.state.api_server_started,
                'background_tasks': self.state.background_tasks_started,
                'agents': self.state.agents_initialized,
                'enterprise': self.state.enterprise_features_started
            },
            'configuration': {
                'web_port': self.web_port,
                'api_port': self.api_port,
                'enable_web': self.enable_web,
                'enable_api': self.enable_api,
                'enable_background': self.enable_background,
                'enable_agents': self.enable_agents,
                'enable_enterprise': self.enable_enterprise,
                'debug': self.debug
            },
            'active_background_tasks': list(self.state.active_background_tasks),
            'state': asdict(self.state)
        }


def main():
    """Main entry point for testing the unified runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MeRNSTA Unified Runner')
    parser.add_argument('--web-port', type=int, default=8000, help='Web server port')
    parser.add_argument('--api-port', type=int, default=8001, help='API server port')
    parser.add_argument('--no-web', action='store_true', help='Disable web server')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('--no-background', action='store_true', help='Disable background tasks')
    parser.add_argument('--no-agents', action='store_true', help='Disable agents')
    parser.add_argument('--enterprise', action='store_true', help='Enable enterprise features')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    runner = MeRNSTAUnifiedRunner(
        web_port=args.web_port,
        api_port=args.api_port,
        enable_web=not args.no_web,
        enable_api=not args.no_api,
        enable_background=not args.no_background,
        enable_agents=not args.no_agents,
        enable_enterprise=args.enterprise,
        debug=args.debug
    )
    
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
