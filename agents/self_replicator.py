#!/usr/bin/env python3
"""
Recursive Self-Replication Agent for MeRNSTA Phase 22

Handles cloning, mutation, and execution of forked agents. This is different from
the genetic evolution system - it works directly with agent source code files.
"""

import os
import sys
import json
import uuid
import time
import shutil
import logging
import subprocess
import importlib.util
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from .base import BaseAgent
from config.settings import get_config


class AgentReplicator(BaseAgent):
    """
    Agent Replicator for recursive self-replication and agent genesis.
    
    Creates actual forked agent files, mutates their source code, tests them,
    and decides whether to keep or prune them based on performance.
    """
    
    def __init__(self):
        super().__init__("agent_replicator")
        
        self.config = get_config().get('self_replication', {})
        
        # Fork management settings
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        self.max_forks = self.config.get('max_forks', 10)
        self.survival_threshold = self.config.get('survival_threshold', 0.75)
        
        # Directory setup
        self.agents_dir = Path("agents")
        self.forks_dir = Path("agent_forks")
        self.logs_dir = Path("output")
        
        # Ensure directories exist
        self.forks_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Fork tracking and registry
        self.active_forks = {}
        self.fork_logs_path = self.logs_dir / "fork_logs.jsonl"
        self.fork_registry_path = self.logs_dir / "fork_registry.json"
        self.fork_metadata = {}
        
        # Load existing fork registry
        self._load_fork_registry()
        
        logging.info(f"[AgentReplicator] Initialized with mutation_rate={self.mutation_rate}, max_forks={self.max_forks}")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the agent replicator"""
        return """You are the Agent Replicator for MeRNSTA's Phase 22 recursive self-replication system.

Your primary responsibilities are:
1. Clone and fork existing agent source code with unique UUIDs
2. Apply mutations to agent code (rename functions, tweak logic, adjust prompts)
3. Test and evaluate forked agents in isolation
4. Score agent performance and implement reintegration policies
5. Prune underperforming forks and maintain optimal agent population

Key capabilities:
- Source code level agent forking and mutation
- Isolated testing and evaluation of agent variants
- Performance-based survival and reintegration
- Autonomous agent population management
- CLI interface for manual fork management

Use your replication capabilities to create diverse agent variants that can
improve overall system performance through evolutionary selection."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process replication-related messages"""
        
        message_lower = message.lower()
        
        try:
            if any(word in message_lower for word in ['fork', 'clone', 'replicate']):
                return self._handle_fork_request(message, context)
            
            elif any(word in message_lower for word in ['mutate', 'mutation', 'change']):
                return self._handle_mutation_request(message, context)
            
            elif any(word in message_lower for word in ['test', 'evaluate', 'score']):
                return self._handle_test_request(message, context)
            
            elif any(word in message_lower for word in ['prune', 'cleanup', 'remove']):
                return self._handle_prune_request(message, context)
            
            elif 'status' in message_lower or 'list' in message_lower:
                return self._get_fork_status()
            
            else:
                # General replicator info
                response = f"Agent Replicator active. "
                response += f"Active forks: {len(self.active_forks)}/{self.max_forks}. "
                response += f"Mutation rate: {self.mutation_rate}. "
                response += f"Use 'fork', 'mutate', 'test', or 'prune' for specific operations."
                
                return {
                    'response': response,
                    'agent': 'agent_replicator'
                }
        
        except Exception as e:
            logging.error(f"[AgentReplicator] Error processing message: {e}")
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'error': str(e),
                'agent': 'agent_replicator'
            }
    
    def fork_agent(self, agent_name: str) -> Optional[str]:
        """
        Clone agent source code to a new file with a UUID
        
        Args:
            agent_name: Name of the agent to fork (e.g., 'critic', 'planner')
            
        Returns:
            Fork ID (UUID) if successful, None if failed
        """
        
        if len(self.active_forks) >= self.max_forks:
            logging.warning(f"[AgentReplicator] Maximum forks ({self.max_forks}) reached")
            return None
        
        # Find source agent file
        source_file = self.agents_dir / f"{agent_name}.py"
        if not source_file.exists():
            logging.error(f"[AgentReplicator] Source agent file not found: {source_file}")
            return None
        
        # Generate fork ID and create fork directory
        fork_id = str(uuid.uuid4())
        fork_dir = self.forks_dir / fork_id
        fork_dir.mkdir(exist_ok=True)
        
        # Copy source code to fork directory
        fork_file = fork_dir / f"{agent_name}_{fork_id[:8]}.py"
        try:
            shutil.copy2(source_file, fork_file)
            
            # Track the fork with enhanced metadata
            fork_metadata = {
                'agent_name': agent_name,
                'fork_file': str(fork_file),
                'created': time.time(),
                'created_iso': datetime.now().isoformat(),
                'mutations': 0,
                'tested': False,
                'score': None,
                'status': 'created',
                'mutation_history': [],
                'test_history': [],
                'performance_metrics': {}
            }
            
            self.active_forks[fork_id] = fork_metadata
            self.fork_metadata[fork_id] = fork_metadata.copy()
            
            # Save registry
            self._save_fork_registry()
            
            # Log fork creation
            self._log_fork_event({
                'event': 'fork_created',
                'fork_id': fork_id,
                'agent_name': agent_name,
                'timestamp': time.time()
            })
            
            logging.info(f"[AgentReplicator] Forked {agent_name} -> {fork_id[:8]}")
            return fork_id
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to fork {agent_name}: {e}")
            return None
    
    def mutate_agent(self, file_path: str) -> bool:
        """
        Apply mutation operations to agent source code
        
        Args:
            file_path: Path to the agent file to mutate
            
        Returns:
            True if mutation successful, False otherwise
        """
        
        if not Path(file_path).exists():
            logging.error(f"[AgentReplicator] File not found for mutation: {file_path}")
            return False
        
        try:
            # Load mutation utilities
            from .mutation_utils import MutationEngine
            mutation_engine = MutationEngine(self.mutation_rate)
            
            # Apply mutations
            success = mutation_engine.mutate_file(file_path)
            
            if success:
                # Update fork tracking with detailed history
                fork_id = self._get_fork_id_from_path(file_path)
                if fork_id and fork_id in self.active_forks:
                    self.active_forks[fork_id]['mutations'] += 1
                    self.active_forks[fork_id]['status'] = 'mutated'
                    self.active_forks[fork_id]['last_mutation'] = time.time()
                    
                    # Track mutation history
                    mutation_event = {
                        'timestamp': time.time(),
                        'timestamp_iso': datetime.now().isoformat(),
                        'mutation_count': self.active_forks[fork_id]['mutations']
                    }
                    
                    if 'mutation_history' not in self.active_forks[fork_id]:
                        self.active_forks[fork_id]['mutation_history'] = []
                    
                    self.active_forks[fork_id]['mutation_history'].append(mutation_event)
                    
                    # Update metadata
                    self.fork_metadata[fork_id] = self.active_forks[fork_id].copy()
                    self._save_fork_registry()
                
                # Log mutation
                self._log_fork_event({
                    'event': 'mutation_applied',
                    'fork_id': fork_id,
                    'file_path': file_path,
                    'timestamp': time.time()
                })
                
                logging.info(f"[AgentReplicator] Successfully mutated {file_path}")
            
            return success
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to mutate {file_path}: {e}")
            return False
    
    def test_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Run test suite or evaluation loop for a forked agent
        
        Args:
            agent_name: Name of the forked agent to test
            
        Returns:
            Test results dictionary
        """
        
        fork_id = self._get_fork_id_from_agent_name(agent_name)
        if not fork_id:
            return {'error': f'Fork not found for agent: {agent_name}'}
        
        fork_info = self.active_forks[fork_id]
        fork_file = Path(fork_info['fork_file'])
        
        if not fork_file.exists():
            return {'error': f'Fork file not found: {fork_file}'}
        
        try:
            # Test syntax validity
            syntax_valid = self._validate_syntax(fork_file)
            if not syntax_valid:
                return {
                    'fork_id': fork_id,
                    'syntax_valid': False,
                    'score': 0.0,
                    'error': 'Syntax validation failed'
                }
            
            # Run basic functionality test
            functionality_score = self._test_functionality(fork_file)
            
            # Calculate overall score
            overall_score = functionality_score * (1.0 if syntax_valid else 0.0)
            
            # Update fork tracking with detailed test history
            self.active_forks[fork_id]['tested'] = True
            self.active_forks[fork_id]['score'] = overall_score
            self.active_forks[fork_id]['status'] = 'tested'
            self.active_forks[fork_id]['last_tested'] = time.time()
            
            # Track test history
            test_event = {
                'timestamp': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                'syntax_valid': syntax_valid,
                'functionality_score': functionality_score,
                'overall_score': overall_score
            }
            
            if 'test_history' not in self.active_forks[fork_id]:
                self.active_forks[fork_id]['test_history'] = []
            
            self.active_forks[fork_id]['test_history'].append(test_event)
            
            # Update metadata
            self.fork_metadata[fork_id] = self.active_forks[fork_id].copy()
            self._save_fork_registry()
            
            # Log test results
            self._log_fork_event({
                'event': 'agent_tested',
                'fork_id': fork_id,
                'score': overall_score,
                'syntax_valid': syntax_valid,
                'timestamp': time.time()
            })
            
            return {
                'fork_id': fork_id,
                'syntax_valid': syntax_valid,
                'functionality_score': functionality_score,
                'overall_score': overall_score,
                'agent_name': agent_name
            }
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to test {agent_name}: {e}")
            return {
                'fork_id': fork_id,
                'error': str(e),
                'score': 0.0
            }
    
    def evaluate_performance(self, log_path: str) -> Dict[str, Any]:
        """
        Read logs and score agent effectiveness
        
        Args:
            log_path: Path to the log file to analyze
            
        Returns:
            Performance evaluation results
        """
        
        if not Path(log_path).exists():
            return {'error': f'Log file not found: {log_path}'}
        
        try:
            scores = {}
            total_entries = 0
            
            with open(log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        total_entries += 1
                        
                        fork_id = entry.get('fork_id')
                        if fork_id and entry.get('event') == 'agent_tested':
                            score = entry.get('score', 0.0)
                            if fork_id not in scores:
                                scores[fork_id] = []
                            scores[fork_id].append(score)
            
            # Calculate average scores
            avg_scores = {}
            for fork_id, score_list in scores.items():
                avg_scores[fork_id] = sum(score_list) / len(score_list)
            
            # Sort by performance
            ranked_forks = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'total_log_entries': total_entries,
                'tested_forks': len(scores),
                'average_scores': avg_scores,
                'ranked_performance': ranked_forks,
                'top_performer': ranked_forks[0] if ranked_forks else None
            }
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to evaluate performance: {e}")
            return {'error': str(e)}
    
    def reintegration_policy(self) -> Dict[str, Any]:
        """
        Decide whether to keep forks based on performance
        
        Returns:
            Reintegration decisions for all tested forks
        """
        
        decisions = {}
        survivors = []
        pruned = []
        
        for fork_id, fork_info in self.active_forks.items():
            if not fork_info.get('tested', False):
                decisions[fork_id] = 'not_tested'
                continue
            
            score = fork_info.get('score', 0.0)
            
            if score >= self.survival_threshold:
                decisions[fork_id] = 'keep'
                survivors.append(fork_id)
            else:
                decisions[fork_id] = 'prune'
                pruned.append(fork_id)
        
        # Log reintegration decisions
        self._log_fork_event({
            'event': 'reintegration_policy',
            'decisions': decisions,
            'survivors': len(survivors),
            'pruned': len(pruned),
            'threshold': self.survival_threshold,
            'timestamp': time.time()
        })
        
        return {
            'decisions': decisions,
            'survivors': survivors,
            'pruned': pruned,
            'survival_threshold': self.survival_threshold,
            'summary': f'{len(survivors)} survivors, {len(pruned)} to prune'
        }
    
    def prune_forks(self, fork_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Remove low-performing forks
        
        Args:
            fork_ids: Specific fork IDs to prune, or None for auto-prune
            
        Returns:
            Pruning results
        """
        
        if fork_ids is None:
            # Auto-prune based on reintegration policy
            policy_result = self.reintegration_policy()
            fork_ids = policy_result['pruned']
        
        pruned_count = 0
        errors = []
        
        for fork_id in fork_ids:
            if fork_id not in self.active_forks:
                errors.append(f'Fork not found: {fork_id}')
                continue
            
            try:
                # Remove fork directory and files
                fork_dir = self.forks_dir / fork_id
                if fork_dir.exists():
                    shutil.rmtree(fork_dir)
                
                # Remove from tracking
                del self.active_forks[fork_id]
                pruned_count += 1
                
                # Log pruning
                self._log_fork_event({
                    'event': 'fork_pruned',
                    'fork_id': fork_id,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                error_msg = f'Failed to prune {fork_id}: {str(e)}'
                errors.append(error_msg)
                logging.error(f"[AgentReplicator] {error_msg}")
        
        return {
            'pruned_count': pruned_count,
            'errors': errors,
            'remaining_forks': len(self.active_forks)
        }
    
    def _handle_fork_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle fork creation requests"""
        
        # Extract agent name from message
        words = message.lower().split()
        agent_name = None
        
        # Look for agent names in the message
        known_agents = ['critic', 'planner', 'debater', 'reflector', 'self_prompter']
        for word in words:
            if word in known_agents:
                agent_name = word
                break
        
        if not agent_name:
            return {
                'response': 'Please specify an agent to fork (e.g., "fork critic")',
                'agent': 'agent_replicator'
            }
        
        fork_id = self.fork_agent(agent_name)
        
        if fork_id:
            response = f"Successfully forked {agent_name} -> {fork_id[:8]}\n"
            response += f"Fork location: agent_forks/{fork_id}/\n"
            response += f"Use '/mutate_agent {fork_id}' to apply mutations"
            
            return {
                'response': response,
                'fork_id': fork_id,
                'agent_name': agent_name,
                'agent': 'agent_replicator'
            }
        else:
            return {
                'response': f"Failed to fork {agent_name}. Check logs for details.",
                'agent': 'agent_replicator'
            }
    
    def _handle_mutation_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle mutation requests"""
        
        # Extract fork ID from message
        words = message.split()
        fork_id = None
        
        for word in words:
            if len(word) >= 8 and word in self.active_forks:
                fork_id = word
                break
        
        if not fork_id:
            return {
                'response': 'Please specify a fork ID to mutate',
                'agent': 'agent_replicator'
            }
        
        fork_info = self.active_forks[fork_id]
        success = self.mutate_agent(fork_info['fork_file'])
        
        if success:
            response = f"Successfully mutated fork {fork_id[:8]}\n"
            response += f"Mutations applied: {fork_info['mutations']}\n"
            response += f"Use '/test_agent {fork_id}' to test the mutated agent"
            
            return {
                'response': response,
                'fork_id': fork_id,
                'mutations': fork_info['mutations'],
                'agent': 'agent_replicator'
            }
        else:
            return {
                'response': f"Failed to mutate fork {fork_id[:8]}",
                'agent': 'agent_replicator'
            }
    
    def _handle_test_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle testing requests"""
        
        # Extract agent name or fork ID from message
        words = message.split()
        target = None
        
        for word in words:
            if len(word) >= 8 and word in self.active_forks:
                target = word
                break
        
        if not target:
            return {
                'response': 'Please specify a fork ID to test',
                'agent': 'agent_replicator'
            }
        
        # Get agent name for the fork
        fork_info = self.active_forks[target]
        agent_name = f"{fork_info['agent_name']}_{target[:8]}"
        
        results = self.test_agent(agent_name)
        
        if 'error' in results:
            return {
                'response': f"Testing failed: {results['error']}",
                'agent': 'agent_replicator'
            }
        
        response = f"Test results for {agent_name}:\n"
        response += f"• Syntax valid: {results['syntax_valid']}\n"
        response += f"• Functionality score: {results['functionality_score']:.2f}\n"
        response += f"• Overall score: {results['overall_score']:.2f}\n"
        
        if results['overall_score'] >= self.survival_threshold:
            response += f"• Status: ✅ SURVIVOR (above {self.survival_threshold})"
        else:
            response += f"• Status: ❌ CANDIDATE FOR PRUNING (below {self.survival_threshold})"
        
        return {
            'response': response,
            'test_results': results,
            'agent': 'agent_replicator'
        }
    
    def _handle_prune_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle pruning requests"""
        
        results = self.prune_forks()
        
        response = f"Pruning completed:\n"
        response += f"• Forks pruned: {results['pruned_count']}\n"
        response += f"• Remaining forks: {results['remaining_forks']}\n"
        
        if results['errors']:
            response += f"• Errors: {len(results['errors'])}\n"
            for error in results['errors'][:3]:  # Show first 3 errors
                response += f"  - {error}\n"
        
        return {
            'response': response,
            'prune_results': results,
            'agent': 'agent_replicator'
        }
    
    def _get_fork_status(self) -> Dict[str, Any]:
        """Get status of all active forks"""
        
        if not self.active_forks:
            return {
                'response': 'No active forks. Use "/fork_agent <agent_name>" to create one.',
                'agent': 'agent_replicator'
            }
        
        response = f"Active forks ({len(self.active_forks)}/{self.max_forks}):\n\n"
        
        for fork_id, fork_info in self.active_forks.items():
            response += f"🔀 {fork_id[:8]} ({fork_info['agent_name']})\n"
            response += f"   Status: {fork_info['status']}\n"
            response += f"   Mutations: {fork_info['mutations']}\n"
            
            if fork_info.get('tested', False):
                score = fork_info.get('score', 0.0)
                status_icon = "✅" if score >= self.survival_threshold else "❌"
                response += f"   Score: {score:.2f} {status_icon}\n"
            else:
                response += f"   Score: Not tested\n"
            
            response += "\n"
        
        return {
            'response': response,
            'active_forks': self.active_forks,
            'agent': 'agent_replicator'
        }
    
    def _validate_syntax(self, file_path: Path) -> bool:
        """Validate Python syntax of a file"""
        
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            compile(code, str(file_path), 'exec')
            return True
            
        except SyntaxError as e:
            logging.error(f"[AgentReplicator] Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logging.error(f"[AgentReplicator] Error validating {file_path}: {e}")
            return False
    
    def _test_functionality(self, file_path: Path) -> float:
        """Test basic functionality of an agent file"""
        
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec is None or spec.loader is None:
                return 0.0
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Basic functionality score (can be enhanced)
            score = 0.5  # Base score for successful import
            
            # Check for required methods/classes
            if hasattr(module, 'BaseAgent'):
                score += 0.2
            
            # Check for agent classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.endswith('Agent'):
                    score += 0.3
                    break
            
            return min(1.0, score)
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Functionality test failed for {file_path}: {e}")
            return 0.0
    
    def _get_fork_id_from_path(self, file_path: str) -> Optional[str]:
        """Extract fork ID from file path"""
        
        path = Path(file_path)
        if len(path.parts) >= 2 and path.parts[-2] in self.active_forks:
            return path.parts[-2]
        return None
    
    def _get_fork_id_from_agent_name(self, agent_name: str) -> Optional[str]:
        """Get fork ID from agent name"""
        
        # Extract fork ID from agent names like "critic_a1b2c3d4"
        if '_' in agent_name:
            parts = agent_name.split('_')
            if len(parts) >= 2:
                fork_id_part = parts[-1]
                # Find full fork ID that starts with this part
                for fork_id in self.active_forks:
                    if fork_id.startswith(fork_id_part):
                        return fork_id
        return None
    
    def _log_fork_event(self, event_data: Dict[str, Any]):
        """Log fork events to JSONL file"""
        
        try:
            with open(self.fork_logs_path, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to log event: {e}")
    
    def _load_fork_registry(self):
        """Load fork registry from disk"""
        try:
            if self.fork_registry_path.exists():
                with open(self.fork_registry_path, 'r') as f:
                    registry_data = json.load(f)
                    self.fork_metadata = registry_data.get('metadata', {})
                    
                    # Restore active forks that still exist
                    saved_forks = registry_data.get('active_forks', {})
                    for fork_id, fork_info in saved_forks.items():
                        fork_file = Path(fork_info['fork_file'])
                        if fork_file.exists():
                            self.active_forks[fork_id] = fork_info
                    
                    logging.info(f"[AgentReplicator] Loaded {len(self.active_forks)} active forks from registry")
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to load fork registry: {e}")
    
    def _save_fork_registry(self):
        """Save fork registry to disk"""
        try:
            registry_data = {
                'metadata': self.fork_metadata,
                'active_forks': self.active_forks,
                'last_updated': time.time(),
                'version': '1.0'
            }
            
            with open(self.fork_registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"[AgentReplicator] Failed to save fork registry: {e}")
    
    def get_fork_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fork statistics"""
        stats = {
            'total_forks_created': len(self.fork_metadata),
            'active_forks': len(self.active_forks),
            'max_forks': self.max_forks,
            'fork_capacity_used': len(self.active_forks) / self.max_forks,
            'mutation_rate': self.mutation_rate,
            'survival_threshold': self.survival_threshold
        }
        
        # Analyze active forks
        if self.active_forks:
            statuses = [fork['status'] for fork in self.active_forks.values()]
            stats['fork_statuses'] = {
                status: statuses.count(status) for status in set(statuses)
            }
            
            # Score statistics
            scores = [fork['score'] for fork in self.active_forks.values() if fork.get('score') is not None]
            if scores:
                stats['score_statistics'] = {
                    'average': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'survivors': len([s for s in scores if s >= self.survival_threshold])
                }
            
            # Mutation statistics
            mutations = [fork['mutations'] for fork in self.active_forks.values()]
            stats['mutation_statistics'] = {
                'total_mutations': sum(mutations),
                'average_mutations_per_fork': sum(mutations) / len(mutations),
                'max_mutations': max(mutations),
                'forks_with_mutations': len([m for m in mutations if m > 0])
            }
        
        return stats
    
    def promote_agent(self, agent_name: str, reason: str = "High performance detected") -> Dict[str, Any]:
        """
        Promote an agent by boosting its contract confidence values.
        
        Args:
            agent_name: Name of the agent to promote
            reason: Reason for promotion
            
        Returns:
            Dict with promotion result
        """
        try:
            # Import registry to get agent
            from .registry import get_agent_registry
            
            registry = get_agent_registry()
            agent = registry.get_agent(agent_name)
            
            if not agent:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found in registry',
                    'agent_name': agent_name
                }
            
            # Check if agent has a contract
            if not agent.contract:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} has no contract to promote',
                    'agent_name': agent_name
                }
            
            # Boost confidence values by 10%
            original_confidence = agent.contract.confidence_vector.copy()
            boosted_confidence = {}
            
            for capability, confidence in original_confidence.items():
                # Boost by 10% but cap at 1.0
                boosted_value = min(1.0, confidence * 1.1)
                boosted_confidence[capability] = boosted_value
            
            # Update contract
            agent.contract.confidence_vector = boosted_confidence
            agent.contract.last_updated = datetime.now()
            
            # Record lifecycle event
            agent.update_lifecycle_event('promotion', {
                'reason': reason,
                'original_confidence': original_confidence,
                'boosted_confidence': boosted_confidence,
                'boost_factor': 1.1
            })
            
            # Log to lifecycle.jsonl
            lifecycle_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'promote',
                'agent_name': agent_name,
                'reason': reason,
                'original_confidence': original_confidence,
                'new_confidence': boosted_confidence,
                'initiated_by': 'agent_replicator'
            }
            
            with open(self.logs_dir / "lifecycle.jsonl", 'a') as f:
                f.write(json.dumps(lifecycle_entry) + '\n')
            
            logging.info(f"[AgentReplicator] Promoted agent {agent_name}: {reason}")
            
            return {
                'success': True,
                'agent_name': agent_name,
                'action': 'promote',
                'reason': reason,
                'confidence_boost': boosted_confidence,
                'original_confidence': original_confidence
            }
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Error promoting agent {agent_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': agent_name
            }
    
    def retire_agent(self, agent_name: str, reason: str = "Performance below threshold") -> Dict[str, Any]:
        """
        Retire an agent by marking it as inactive and unregistering it.
        
        Args:
            agent_name: Name of the agent to retire
            reason: Reason for retirement
            
        Returns:
            Dict with retirement result
        """
        try:
            # Import registry to get agent
            from .registry import get_agent_registry
            
            registry = get_agent_registry()
            agent = registry.get_agent(agent_name)
            
            if not agent:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found in registry',
                    'agent_name': agent_name
                }
            
            # Record lifecycle event
            agent.update_lifecycle_event('retirement', {
                'reason': reason,
                'final_metrics': agent.get_lifecycle_metrics() if hasattr(agent, 'get_lifecycle_metrics') else {}
            })
            
            # Archive agent contract if it exists
            if agent.contract:
                archive_path = self.logs_dir / f"retired_contracts"
                archive_path.mkdir(exist_ok=True)
                
                contract_data = {
                    'agent_name': agent_name,
                    'retirement_timestamp': datetime.now().isoformat(),
                    'reason': reason,
                    'final_contract': {
                        'purpose': agent.contract.purpose,
                        'capabilities': agent.contract.capabilities,
                        'confidence_vector': agent.contract.confidence_vector,
                        'version': agent.contract.version
                    }
                }
                
                with open(archive_path / f"{agent_name}_contract.json", 'w') as f:
                    json.dump(contract_data, f, indent=2)
            
            # Mark agent as inactive in registry (don't remove completely)
            # This preserves the agent for potential reactivation
            agent.enabled = False
            
            # Log to lifecycle.jsonl
            lifecycle_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'retire',
                'agent_name': agent_name,
                'reason': reason,
                'final_metrics': agent.get_lifecycle_metrics() if hasattr(agent, 'get_lifecycle_metrics') else {},
                'initiated_by': 'agent_replicator'
            }
            
            with open(self.logs_dir / "lifecycle.jsonl", 'a') as f:
                f.write(json.dumps(lifecycle_entry) + '\n')
            
            logging.info(f"[AgentReplicator] Retired agent {agent_name}: {reason}")
            
            return {
                'success': True,
                'agent_name': agent_name,
                'action': 'retire',
                'reason': reason,
                'archived': agent.contract is not None
            }
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Error retiring agent {agent_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': agent_name
            }
    
    def mutate_agent(self, agent_name: str, reason: str = "Drift or performance issues detected") -> Dict[str, Any]:
        """
        Mutate an agent by calling the existing fork/mutation pipeline.
        
        Args:
            agent_name: Name of the agent to mutate
            reason: Reason for mutation
            
        Returns:
            Dict with mutation result
        """
        try:
            # Import registry to get agent
            from .registry import get_agent_registry
            
            registry = get_agent_registry()
            agent = registry.get_agent(agent_name)
            
            if not agent:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found in registry',
                    'agent_name': agent_name
                }
            
            # Record lifecycle event
            agent.update_lifecycle_event('mutation', {
                'reason': reason,
                'pre_mutation_metrics': agent.get_lifecycle_metrics() if hasattr(agent, 'get_lifecycle_metrics') else {}
            })
            
            # Use existing fork_agent method with lifecycle-specific context
            fork_result = self.fork_agent(
                agent_name=agent_name,
                replication_reason=f"Lifecycle mutation: {reason}",
                triggered_by_reflection=True  # Mark as automated lifecycle action
            )
            
            if fork_result.get('success'):
                fork_id = fork_result.get('fork_id')
                
                # Apply mutation to the fork
                mutation_result = self.mutate_fork(fork_id, mutation_intensity=0.3)
                
                if mutation_result.get('success'):
                    # Log to lifecycle.jsonl
                    lifecycle_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'mutate',
                        'agent_name': agent_name,
                        'reason': reason,
                        'fork_id': fork_id,
                        'mutation_details': mutation_result,
                        'initiated_by': 'agent_replicator'
                    }
                    
                    with open(self.logs_dir / "lifecycle.jsonl", 'a') as f:
                        f.write(json.dumps(lifecycle_entry) + '\n')
                    
                    logging.info(f"[AgentReplicator] Mutated agent {agent_name}: {reason} -> fork {fork_id}")
                    
                    return {
                        'success': True,
                        'agent_name': agent_name,
                        'action': 'mutate',
                        'reason': reason,
                        'fork_id': fork_id,
                        'mutation_result': mutation_result
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Fork created but mutation failed: {mutation_result.get("error", "Unknown error")}',
                        'agent_name': agent_name,
                        'fork_id': fork_id
                    }
            else:
                return {
                    'success': False,
                    'error': f'Failed to create fork: {fork_result.get("error", "Unknown error")}',
                    'agent_name': agent_name
                }
            
        except Exception as e:
            logging.error(f"[AgentReplicator] Error mutating agent {agent_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': agent_name
            }
    
    def get_fork_lineage(self, fork_id: str) -> Dict[str, Any]:
        """Get lineage information for a specific fork"""
        if fork_id not in self.active_forks and fork_id not in self.fork_metadata:
            return {'error': f'Fork {fork_id} not found'}
        
        fork_info = self.active_forks.get(fork_id) or self.fork_metadata.get(fork_id, {})
        
        lineage = {
            'fork_id': fork_id,
            'agent_name': fork_info.get('agent_name'),
            'created': fork_info.get('created'),
            'mutations': fork_info.get('mutations', 0),
            'score': fork_info.get('score'),
            'status': fork_info.get('status'),
            'replication_reason': fork_info.get('replication_reason'),
            'triggered_by_reflection': fork_info.get('triggered_by_reflection', False)
        }
        
        # Add performance timeline if available
        performance_events = []
        try:
            if self.fork_logs_path.exists():
                with open(self.fork_logs_path, 'r') as f:
                    for line in f:
                        event = json.loads(line)
                        if event.get('fork_id') == fork_id:
                            performance_events.append(event)
            
            lineage['performance_timeline'] = performance_events
        except Exception as e:
            logging.error(f"[AgentReplicator] Error getting performance timeline: {e}")
        
        return lineage


# Global instance
_agent_replicator_instance = None

def get_agent_replicator() -> AgentReplicator:
    """Get global agent replicator instance"""
    global _agent_replicator_instance
    if _agent_replicator_instance is None:
        _agent_replicator_instance = AgentReplicator()
    return _agent_replicator_instance