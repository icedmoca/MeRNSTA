#!/usr/bin/env python3
"""
🖥️ MeRNSTA Interactive Shell - Phase 30
Interactive shell client that provides a REPL interface to the MeRNSTA cognitive system.

This module provides an interactive command-line interface similar to IPython or Bash
that communicates with the system bridge API endpoints.

Commands:
- ask <question>     - Ask the cognitive system a question
- status             - Show system status and health
- memory <type>      - Query memory (search, recent, facts, contradictions)  
- reflect            - Trigger reflection process
- goal <action>      - Goal management (list, add, remove)
- agent              - Show agent information
- exit/quit          - Exit the shell
"""

import os
import sys
import json
import argparse
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import readline  # For command history and editing

# Add rich console support if available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
# Sovereign Mode imports (Phase 35)
from system.sovereign_crypto import get_sovereign_crypto
from agents.sovereign_guardian_agent import get_sovereign_guardian
from storage.memory_encryption import get_memory_encryption_manager
from system.self_update_manager import get_self_update_manager


class MeRNSTAShell:
    """Interactive shell for MeRNSTA OS integration."""
    
    def __init__(self, api_host: str = "127.0.0.1", api_port: int = 8181):
        self.api_host = api_host
        self.api_port = api_port
        self.api_base_url = f"http://{api_host}:{api_port}"
        
        # Initialize rich console if available
        self.console = Console() if RICH_AVAILABLE else None
        
        # Command history
        self.command_history = []
        
        # Session info
        self.session_id = f"shell_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Available commands
        self.commands = {
            'ask': self.cmd_ask,
            'status': self.cmd_status,
            'memory': self.cmd_memory,
            'reflect': self.cmd_reflect,
            'goal': self.cmd_goal,
            'agent': self.cmd_agent,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'clear': self.cmd_clear,
            'history': self.cmd_history,
            'config': self.cmd_config,
            # Sovereign Mode Commands (Phase 35)
            'sovereign_status': self.cmd_sovereign_status,
            'seal': self.cmd_seal,
            'regen_identity': self.cmd_regen_identity,
            'self_update': self.cmd_self_update
        }
        
        # Setup readline history
        self._setup_readline()
    
    def _setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            # Setup command completion
            readline.set_completer(self._completer)
            readline.parse_and_bind('tab: complete')
            
            # Load history file if it exists
            history_file = os.path.expanduser("~/.mernsta_shell_history")
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
                
        except Exception:
            pass  # Readline not available or setup failed
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Command completion function."""
        matches = [cmd for cmd in self.commands.keys() if cmd.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None
    
    def _save_history(self):
        """Save command history to file."""
        try:
            history_file = os.path.expanduser("~/.mernsta_shell_history")
            readline.write_history_file(history_file)
        except Exception:
            pass
    
    # Sovereign Mode Commands (Phase 35)
    async def cmd_sovereign_status(self, args: List[str]):
        """Show sovereign mode status and security information."""
        try:
            # Get status from all sovereign components
            crypto = get_sovereign_crypto()
            guardian = get_sovereign_guardian()
            memory_manager = get_memory_encryption_manager()
            update_manager = get_self_update_manager()
            
            crypto_status = crypto.get_system_status()
            guardian_status = guardian.get_status()
            memory_status = memory_manager.get_encryption_status()
            update_status = update_manager.get_status()
            
            if RICH_AVAILABLE:
                # Create rich status display
                status_table = Table(title="🔒 Sovereign Mode Status")
                status_table.add_column("Component", style="cyan")
                status_table.add_column("Status", style="white")
                status_table.add_column("Details", style="dim")
                
                # Sovereign system
                sovereign_enabled = crypto_status.get("sovereign_mode_enabled", False)
                status_table.add_row(
                    "Sovereign Mode",
                    "[green]ENABLED[/green]" if sovereign_enabled else "[red]DISABLED[/red]",
                    f"Guardian: {'Active' if guardian_status.get('running') else 'Inactive'}"
                )
                
                # Identity
                identity = crypto_status.get("identity", {})
                status_table.add_row(
                    "Identity",
                    "[green]VALID[/green]" if identity.get("fingerprint") else "[red]INVALID[/red]",
                    f"Fingerprint: {identity.get('fingerprint', 'None')}"
                )
                
                # Memory encryption
                encryption_enabled = memory_status.get("encryption_enabled", False)
                encrypted_count = memory_status.get("encrypted_databases", 0)
                total_count = memory_status.get("total_databases", 0)
                status_table.add_row(
                    "Memory Encryption",
                    "[green]ACTIVE[/green]" if encryption_enabled else "[red]INACTIVE[/red]",
                    f"{encrypted_count}/{total_count} databases encrypted"
                )
                
                # Contract enforcement
                enforcement_mode = guardian_status.get("enforcement_mode", "off")
                active_contracts = guardian_status.get("metrics", {}).get("active_contracts", 0)
                status_table.add_row(
                    "Contract Enforcement",
                    f"[yellow]{enforcement_mode.upper()}[/yellow]",
                    f"{active_contracts} active contracts"
                )
                
                # Self-update system
                update_enabled = update_status.get("enabled", False)
                current_version = update_status.get("current_version", "unknown")
                status_table.add_row(
                    "Self-Update",
                    "[green]ENABLED[/green]" if update_enabled else "[red]DISABLED[/red]",
                    f"Version: {current_version}"
                )
                
                # Audit logging
                audit_enabled = crypto_status.get("audit_logs", {}).get("enabled", False)
                status_table.add_row(
                    "Audit Logging",
                    "[green]ACTIVE[/green]" if audit_enabled else "[red]INACTIVE[/red]",
                    "Immutable logs" if audit_enabled else "Disabled"
                )
                
                self.console.print(status_table)
                
                # Additional details panel
                details = f"""
[bold]Identity Details:[/bold]
• Created: {identity.get('created_at', 'Unknown')}
• Expires: {identity.get('expires_at', 'Never')}
• Days until rotation: {identity.get('days_until_rotation', 'N/A')}

[bold]Guardian Metrics:[/bold]
• Violations detected: {guardian_status.get('metrics', {}).get('violations_detected', 0)}
• Actions taken: {guardian_status.get('metrics', {}).get('actions_taken', 0)}
• Agents suspended: {guardian_status.get('metrics', {}).get('agents_suspended', 0)}

[bold]Memory Encryption:[/bold]
• Algorithm: {memory_status.get('encryption_algorithm', 'N/A')}
• System status: {memory_status.get('system_status', 'Unknown')}

[bold]Update System:[/bold]
• Last check: {update_status.get('last_check', 'Never')}
• Monitoring: {'Active' if update_status.get('monitoring') else 'Inactive'}
• Auto-apply patch: {'Yes' if update_status.get('auto_apply_patch') else 'No'}
                """
                
                details_panel = Panel(
                    details.strip(),
                    title="🔍 Detailed Status",
                    border_style="blue"
                )
                self.console.print(details_panel)
                
            else:
                # Plain text display
                print("\n🔒 Sovereign Mode Status:")
                print("="*50)
                print(f"Sovereign Mode: {'ENABLED' if crypto_status.get('sovereign_mode_enabled') else 'DISABLED'}")
                print(f"Identity: {identity.get('fingerprint', 'None')[:16]}...")
                print(f"Memory Encryption: {'ACTIVE' if memory_status.get('encryption_enabled') else 'INACTIVE'}")
                print(f"Contract Enforcement: {guardian_status.get('enforcement_mode', 'off').upper()}")
                print(f"Self-Update: {'ENABLED' if update_status.get('enabled') else 'DISABLED'}")
                print(f"Current Version: {update_status.get('current_version', 'unknown')}")
                
        except Exception as e:
            self.print_error(f"Failed to get sovereign status: {e}")
    
    async def cmd_seal(self, args: List[str]):
        """Seal the system by encrypting all memory and securing keys."""
        try:
            if RICH_AVAILABLE:
                if not Confirm.ask("🔒 Are you sure you want to seal the system? This will encrypt all databases."):
                    self.print_info("Seal operation cancelled")
                    return
            else:
                response = input("🔒 Are you sure you want to seal the system? This will encrypt all databases. (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    self.print_info("Seal operation cancelled")
                    return
            
            # Perform system sealing
            memory_manager = get_memory_encryption_manager()
            
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(description="Sealing system...", total=None)
                    result = memory_manager.seal_system()
            else:
                print("Sealing system...")
                result = memory_manager.seal_system()
            
            if result.get("sealed", False):
                self.print_success("System successfully sealed!")
                if RICH_AVAILABLE:
                    self.console.print("[dim]All databases encrypted and identity secured[/dim]")
                else:
                    print("All databases encrypted and identity secured")
            else:
                self.print_error(f"Sealing failed: {result.get('status', 'Unknown error')}")
                
        except Exception as e:
            self.print_error(f"Seal operation failed: {e}")
    
    async def cmd_regen_identity(self, args: List[str]):
        """Regenerate sovereign identity keypair."""
        try:
            if RICH_AVAILABLE:
                if not Confirm.ask("🔑 Regenerate identity? This will create new cryptographic keys."):
                    self.print_info("Identity regeneration cancelled")
                    return
            else:
                response = input("🔑 Regenerate identity? This will create new cryptographic keys. (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    self.print_info("Identity regeneration cancelled")
                    return
            
            # Regenerate identity
            crypto = get_sovereign_crypto()
            
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(description="Generating new identity...", total=None)
                    new_identity = crypto.generate_identity(force_new=True)
            else:
                print("Generating new identity...")
                new_identity = crypto.generate_identity(force_new=True)
            
            if new_identity:
                self.print_success("New identity generated successfully!")
                if RICH_AVAILABLE:
                    self.console.print(f"[dim]New fingerprint: {new_identity.fingerprint[:16]}...[/dim]")
                else:
                    print(f"New fingerprint: {new_identity.fingerprint[:16]}...")
            else:
                self.print_error("Failed to generate new identity")
                
        except Exception as e:
            self.print_error(f"Identity regeneration failed: {e}")
    
    async def cmd_self_update(self, args: List[str]):
        """Manage autonomous self-update system."""
        if not args:
            # Show update status
            try:
                update_manager = get_self_update_manager()
                status = update_manager.get_status()
                
                if RICH_AVAILABLE:
                    update_table = Table(title="🔄 Self-Update System")
                    update_table.add_column("Setting", style="cyan")
                    update_table.add_column("Value", style="white")
                    
                    update_table.add_row("Status", "[green]ENABLED[/green]" if status.get("enabled") else "[red]DISABLED[/red]")
                    update_table.add_row("Current Version", status.get("current_version", "unknown"))
                    update_table.add_row("Monitoring", "[green]ACTIVE[/green]" if status.get("monitoring") else "[red]INACTIVE[/red]")
                    update_table.add_row("Last Check", status.get("last_check", "Never"))
                    update_table.add_row("Auto-apply Minor", "Yes" if status.get("auto_apply_minor") else "No")
                    update_table.add_row("Auto-apply Patch", "Yes" if status.get("auto_apply_patch") else "No")
                    update_table.add_row("Signed Goals", str(status.get("signed_goals", 0)))
                    update_table.add_row("Update History", str(status.get("update_history", 0)))
                    
                    self.console.print(update_table)
                    
                    # Show recent updates
                    recent_updates = status.get("recent_updates", [])
                    if recent_updates:
                        self.console.print("\n[bold]Recent Updates:[/bold]")
                        for update in recent_updates[-3:]:
                            self.console.print(f"• {update.get('version')} - {update.get('applied_at')}")
                    
                else:
                    print("\n🔄 Self-Update System:")
                    print("="*30)
                    print(f"Status: {'ENABLED' if status.get('enabled') else 'DISABLED'}")
                    print(f"Current Version: {status.get('current_version', 'unknown')}")
                    print(f"Monitoring: {'ACTIVE' if status.get('monitoring') else 'INACTIVE'}")
                    print(f"Last Check: {status.get('last_check', 'Never')}")
                    
            except Exception as e:
                self.print_error(f"Failed to get update status: {e}")
            return
        
        # Handle subcommands
        subcommand = args[0].lower()
        
        if subcommand == "start":
            try:
                update_manager = get_self_update_manager()
                await update_manager.start_monitoring()
                self.print_success("Self-update monitoring started")
            except Exception as e:
                self.print_error(f"Failed to start monitoring: {e}")
        
        elif subcommand == "stop":
            try:
                update_manager = get_self_update_manager()
                await update_manager.stop_monitoring()
                self.print_success("Self-update monitoring stopped")
            except Exception as e:
                self.print_error(f"Failed to stop monitoring: {e}")
        
        elif subcommand == "check":
            try:
                update_manager = get_self_update_manager()
                self.print_info("Checking for updates...")
                # This would trigger a manual check - implementation would depend on the update manager API
                self.print_info("Update check completed")
            except Exception as e:
                self.print_error(f"Update check failed: {e}")
        
        else:
            self.print_error(f"Unknown subcommand: {subcommand}")
            self.print_info("Available subcommands: start, stop, check")
    
    def print_banner(self):
        """Print the shell banner."""
        if RICH_AVAILABLE:
            banner = Panel.fit(
                "[bold blue]🧠 MeRNSTA Interactive Shell[/bold blue]\n"
                "[dim]OS Integration - Phase 30[/dim]\n\n"
                f"[green]Connected to:[/green] {self.api_base_url}\n"
                f"[green]Session ID:[/green] {self.session_id}\n\n"
                "[yellow]Commands:[/yellow] ask, status, memory, reflect, goal, agent, help, exit\n"
                "[dim]Type 'help' for detailed command information[/dim]",
                title="MeRNSTA Shell",
                border_style="blue"
            )
            self.console.print(banner)
        else:
            print("\n" + "="*70)
            print("🧠 MeRNSTA Interactive Shell - OS Integration")
            print("="*70)
            print(f"Connected to: {self.api_base_url}")
            print(f"Session ID: {self.session_id}")
            print("\nCommands: ask, status, memory, reflect, goal, agent, help, exit")
            print("Type 'help' for detailed command information")
            print("-"*70)
    
    async def check_api_connection(self) -> bool:
        """Check if the API server is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass
        return False
    
    async def make_api_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make an API request to the system bridge."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}{endpoint}"
                
                if method.upper() == 'GET':
                    async with session.get(url, timeout=30) as response:
                        return await response.json()
                elif method.upper() == 'POST':
                    async with session.post(url, json=data, timeout=30) as response:
                        return await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
        except aiohttp.ClientError as e:
            return {"error": f"Connection error: {e}"}
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def print_error(self, message: str):
        """Print an error message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}")
    
    def print_success(self, message: str):
        """Print a success message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold green]Success:[/bold green] {message}")
        else:
            print(f"Success: {message}")
    
    def print_info(self, message: str):
        """Print an info message."""
        if RICH_AVAILABLE:
            self.console.print(f"[blue]Info:[/blue] {message}")
        else:
            print(f"Info: {message}")
    
    # Command implementations
    async def cmd_ask(self, args: List[str]):
        """Ask the cognitive system a question."""
        if not args:
            self.print_error("Please provide a question. Usage: ask <your question>")
            return
        
        question = " ".join(args)
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(description="Processing question...", total=None)
                
                result = await self.make_api_request('POST', '/ask', {
                    'query': question,
                    'session_id': self.session_id
                })
        else:
            print("Processing question...")
            result = await self.make_api_request('POST', '/ask', {
                'query': question,
                'session_id': self.session_id
            })
        
        if 'error' in result:
            self.print_error(result['error'])
            return
        
        # Display response
        if RICH_AVAILABLE:
            response_panel = Panel(
                f"[bold]Question:[/bold] {question}\n\n"
                f"[bold]Response:[/bold] {result.get('response', 'No response')}\n\n"
                f"[dim]Confidence: {result.get('confidence', 'N/A')} | "
                f"Time: {result.get('timestamp', 'N/A')}[/dim]",
                title="🧠 MeRNSTA Response",
                border_style="green"
            )
            self.console.print(response_panel)
        else:
            print(f"\n🧠 MeRNSTA Response:")
            print(f"Question: {question}")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    
    async def cmd_status(self, args: List[str]):
        """Show system status."""
        result = await self.make_api_request('GET', '/status')
        
        if 'error' in result:
            self.print_error(result['error'])
            return
        
        if RICH_AVAILABLE:
            # Create status table
            table = Table(title="🔋 System Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("System Status", result.get('system_status', 'Unknown'))
            table.add_row("Uptime", f"{result.get('uptime_seconds', 0):.0f} seconds")
            table.add_row("Cognitive System", "Active" if result.get('cognitive_system_active') else "Inactive")
            table.add_row("Last Activity", result.get('last_activity', 'Unknown'))
            
            self.console.print(table)
            
            # Background tasks
            bg_tasks = result.get('background_tasks', {})
            if bg_tasks:
                bg_table = Table(title="Background Tasks")
                bg_table.add_column("Task", style="cyan")
                bg_table.add_column("Status", style="green")
                
                for task, status in bg_tasks.items():
                    bg_table.add_row(task, status)
                
                self.console.print(bg_table)
        else:
            print("\n🔋 System Status:")
            print(f"System Status: {result.get('system_status', 'Unknown')}")
            print(f"Uptime: {result.get('uptime_seconds', 0):.0f} seconds")
            print(f"Cognitive System: {'Active' if result.get('cognitive_system_active') else 'Inactive'}")
            print(f"Last Activity: {result.get('last_activity', 'Unknown')}")
            
            bg_tasks = result.get('background_tasks', {})
            if bg_tasks:
                print("\nBackground Tasks:")
                for task, status in bg_tasks.items():
                    print(f"  {task}: {status}")
    
    async def cmd_memory(self, args: List[str]):
        """Query memory system."""
        if not args:
            self.print_error("Please specify memory query type. Usage: memory <search|recent|facts|contradictions> [query]")
            return
        
        query_type = args[0].lower()
        query_text = " ".join(args[1:]) if len(args) > 1 else None
        
        if query_type not in ['search', 'recent', 'facts', 'contradictions']:
            self.print_error(f"Unknown memory query type: {query_type}")
            return
        
        request_data = {
            'query_type': query_type,
            'limit': 10
        }
        
        if query_text:
            request_data['query'] = query_text
        
        result = await self.make_api_request('POST', '/memory', request_data)
        
        if 'error' in result:
            self.print_error(result['error'])
            return
        
        results = result.get('results', [])
        
        if RICH_AVAILABLE:
            table = Table(title=f"💾 Memory Query: {query_type}")
            table.add_column("Item", style="cyan")
            table.add_column("Details", style="white")
            
            for i, item in enumerate(results[:10]):  # Limit display
                item_str = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                table.add_row(f"#{i+1}", item_str[:200] + "..." if len(item_str) > 200 else item_str)
            
            self.console.print(table)
            self.print_info(f"Found {result.get('total_count', 0)} total results")
        else:
            print(f"\n💾 Memory Query: {query_type}")
            for i, item in enumerate(results[:10]):
                print(f"#{i+1}: {item}")
            print(f"\nFound {result.get('total_count', 0)} total results")
    
    async def cmd_reflect(self, args: List[str]):
        """Trigger reflection process."""
        focus_area = " ".join(args) if args else None
        
        request_data = {'trigger_type': 'manual'}
        if focus_area:
            request_data['focus_area'] = focus_area
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(description="Running reflection...", total=None)
                result = await self.make_api_request('POST', '/reflect', request_data)
        else:
            print("Running reflection...")
            result = await self.make_api_request('POST', '/reflect', request_data)
        
        if 'error' in result:
            self.print_error(result['error'])
            return
        
        if result.get('success'):
            self.print_success(f"Reflection completed in {result.get('duration_seconds', 0):.2f} seconds")
            
            insights = result.get('insights', [])
            if insights and RICH_AVAILABLE:
                insights_panel = Panel(
                    "\n".join([f"• {insight}" for insight in insights]),
                    title="🔍 Reflection Insights",
                    border_style="yellow"
                )
                self.console.print(insights_panel)
            elif insights:
                print("\n🔍 Reflection Insights:")
                for insight in insights:
                    print(f"• {insight}")
        else:
            self.print_error("Reflection failed")
    
    async def cmd_goal(self, args: List[str]):
        """Goal management."""
        if not args:
            # Default to list goals
            action = "list"
            goal_args = []
        else:
            action = args[0].lower()
            goal_args = args[1:]
        
        if action not in ['list', 'add', 'remove', 'update']:
            self.print_error(f"Unknown goal action: {action}. Use: list, add, remove, update")
            return
        
        request_data = {'action': action}
        
        if action == 'add' and goal_args:
            request_data['goal_text'] = " ".join(goal_args)
        elif action in ['remove', 'update'] and goal_args:
            request_data['goal_id'] = goal_args[0]
            if action == 'update' and len(goal_args) > 1:
                request_data['goal_text'] = " ".join(goal_args[1:])
        
        result = await self.make_api_request('POST', '/goal', request_data)
        
        if 'error' in result:
            self.print_error(result['error'])
            return
        
        if result.get('success'):
            self.print_success(result.get('message', 'Goal operation completed'))
            
            goals = result.get('goals', [])
            if goals and RICH_AVAILABLE:
                table = Table(title="🎯 Current Goals")
                table.add_column("ID", style="cyan")
                table.add_column("Goal", style="white")
                table.add_column("Priority", style="magenta")
                
                for goal in goals:
                    goal_id = goal.get('id', 'N/A')
                    goal_text = goal.get('text', goal.get('goal', 'N/A'))
                    priority = goal.get('priority', 'N/A')
                    table.add_row(str(goal_id), goal_text, str(priority))
                
                self.console.print(table)
            elif goals:
                print("\n🎯 Current Goals:")
                for goal in goals:
                    print(f"• {goal}")
        else:
            self.print_error(result.get('message', 'Goal operation failed'))
    
    async def cmd_agent(self, args: List[str]):
        """Show agent information."""
        # This is a placeholder - could be extended to show agent status
        self.print_info("Agent information endpoint - placeholder for future implementation")
        
        # Show basic system info for now
        await self.cmd_status([])
    
    async def cmd_help(self, args: List[str]):
        """Show help information."""
        if RICH_AVAILABLE:
            help_text = """
[bold blue]Available Commands:[/bold blue]

[cyan]ask <question>[/cyan]     - Ask the cognitive system a question
[cyan]status[/cyan]             - Show system status and health information
[cyan]memory <type> [query][/cyan] - Query memory system
    [dim]Types: search, recent, facts, contradictions[/dim]
[cyan]reflect [focus][/cyan]    - Trigger reflection process
[cyan]goal <action> [args][/cyan] - Goal management
    [dim]Actions: list, add <text>, remove <id>, update <id> <text>[/dim]
[cyan]agent[/cyan]              - Show agent information
[cyan]config[/cyan]             - Show configuration information
[cyan]history[/cyan]            - Show command history
[cyan]clear[/cyan]              - Clear screen
[cyan]help[/cyan]               - Show this help message
[cyan]exit/quit[/cyan]          - Exit the shell

[bold yellow]🔒 Sovereign Mode Commands (Phase 35):[/bold yellow]
[cyan]sovereign_status[/cyan]   - Show sovereign mode security status
[cyan]seal[/cyan]               - Seal system (encrypt all memory)
[cyan]regen_identity[/cyan]     - Regenerate cryptographic identity
[cyan]self_update[/cyan]        - Manage autonomous self-update system

[yellow]Examples:[/yellow]
  ask What are my current beliefs about AI?
  memory search artificial intelligence
  memory recent
  goal add Learn more about machine learning
  goal list
  reflect cognitive biases
            """
            
            help_panel = Panel(
                help_text,
                title="📖 MeRNSTA Shell Help",
                border_style="blue"
            )
            self.console.print(help_panel)
        else:
            print("\n📖 MeRNSTA Shell Help:")
            print("="*50)
            print("ask <question>     - Ask the cognitive system a question")
            print("status             - Show system status and health")
            print("memory <type>      - Query memory (search, recent, facts, contradictions)")
            print("reflect [focus]    - Trigger reflection process")
            print("goal <action>      - Goal management (list, add, remove, update)")
            print("agent              - Show agent information")
            print("config             - Show configuration")
            print("history            - Show command history")
            print("clear              - Clear screen")
            print("help               - Show this help")
            print("exit/quit          - Exit the shell")
            print("")
            print("🔒 Sovereign Mode Commands (Phase 35):")
            print("sovereign_status   - Show sovereign mode security status")
            print("seal               - Seal system (encrypt all memory)")
            print("regen_identity     - Regenerate cryptographic identity")
            print("self_update        - Manage autonomous self-update system")
    
    async def cmd_config(self, args: List[str]):
        """Show configuration information."""
        try:
            config = get_config()
            os_config = config.get('os_integration', {})
            
            if RICH_AVAILABLE:
                config_text = json.dumps(os_config, indent=2)
                syntax = Syntax(config_text, "json", theme="monokai", line_numbers=True)
                
                config_panel = Panel(
                    syntax,
                    title="⚙️ OS Integration Configuration",
                    border_style="cyan"
                )
                self.console.print(config_panel)
            else:
                print("\n⚙️ OS Integration Configuration:")
                print(json.dumps(os_config, indent=2))
                
        except Exception as e:
            self.print_error(f"Failed to load configuration: {e}")
    
    async def cmd_history(self, args: List[str]):
        """Show command history."""
        if not self.command_history:
            self.print_info("No command history available")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="📜 Command History")
            table.add_column("#", style="cyan")
            table.add_column("Command", style="white")
            table.add_column("Time", style="dim")
            
            for i, (cmd, timestamp) in enumerate(self.command_history[-20:]):  # Last 20
                table.add_row(str(i+1), cmd, timestamp.strftime("%H:%M:%S"))
            
            self.console.print(table)
        else:
            print("\n📜 Command History:")
            for i, (cmd, timestamp) in enumerate(self.command_history[-20:]):
                print(f"{i+1:2d}. {cmd} ({timestamp.strftime('%H:%M:%S')})")
    
    async def cmd_clear(self, args: List[str]):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.print_banner()
    
    async def cmd_exit(self, args: List[str]):
        """Exit the shell."""
        if RICH_AVAILABLE:
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
        else:
            print("Goodbye! 👋")
        self._save_history()
        return True  # Signal to exit
    
    def get_prompt(self) -> str:
        """Get the shell prompt."""
        return "mernsta> "
    
    async def run(self):
        """Run the interactive shell."""
        # Check API connection
        if not await self.check_api_connection():
            self.print_error(f"Cannot connect to MeRNSTA API at {self.api_base_url}")
            self.print_info("Please ensure the system bridge API is running:")
            self.print_info("  python api/system_bridge.py")
            return
        
        self.print_banner()
        self.print_success("Connected to MeRNSTA API")
        
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = Prompt.ask(self.get_prompt()).strip()
                else:
                    user_input = input(self.get_prompt()).strip()
                
                if not user_input:
                    continue
                
                # Add to history
                self.command_history.append((user_input, datetime.now()))
                
                # Parse command
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]
                
                # Execute command
                if command in self.commands:
                    should_exit = await self.commands[command](args)
                    if should_exit:
                        break
                else:
                    self.print_error(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to leave the shell.")
            except EOFError:
                break
            except Exception as e:
                self.print_error(f"Unexpected error: {e}")
        
        self._save_history()


def main():
    """Main entry point for the shell."""
    parser = argparse.ArgumentParser(description='MeRNSTA Interactive Shell')
    parser.add_argument('--host', default='127.0.0.1', help='API server host')
    parser.add_argument('--port', type=int, default=8181, help='API server port')
    parser.add_argument('--no-rich', action='store_true', help='Disable rich console features')
    
    args = parser.parse_args()
    
    # Disable rich if requested
    if args.no_rich:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False
    
    # Create and run shell
    shell = MeRNSTAShell(api_host=args.host, api_port=args.port)
    
    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        print("\nShell interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()