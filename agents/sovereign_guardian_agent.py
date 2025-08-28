#!/usr/bin/env python3
"""
Sovereign Guardian Agent for MeRNSTA Phase 35
Enforces cryptographic contracts and monitors system integrity.
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path
import queue

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from system.sovereign_crypto import get_sovereign_crypto, AgentContract
from storage.audit_logger import AuditLogger  # We'll create this

logger = logging.getLogger(__name__)


@dataclass
class ContractViolation:
    """Represents a contract violation event."""
    agent_id: str
    contract_id: str
    violation_type: str
    description: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    action_taken: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'contract_id': self.contract_id,
            'violation_type': self.violation_type,
            'description': self.description,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'action_taken': self.action_taken,
            'evidence': self.evidence
        }


@dataclass
class GuardianMetrics:
    """Guardian operational metrics."""
    contracts_monitored: int = 0
    violations_detected: int = 0
    actions_taken: int = 0
    agents_suspended: int = 0
    system_interventions: int = 0
    last_check_time: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.now)


class SovereignGuardianAgent(BaseAgent):
    """
    Guardian agent that enforces cryptographic contracts and maintains system sovereignty.
    
    Responsibilities:
    - Monitor all agent actions against their contracts
    - Detect contract violations and security breaches
    - Enforce contract compliance with escalating actions
    - Maintain audit logs of all enforcement actions
    - Protect system integrity and sovereignty
    """
    
    def __init__(self, agent_id: str = "sovereign_guardian"):
        super().__init__(agent_id)
        self.crypto = get_sovereign_crypto()
        self.config = self._load_guardian_config()
        
        # Internal state
        self.active_contracts: Dict[str, AgentContract] = {}
        self.suspended_agents: Set[str] = set()
        self.violation_history: List[ContractViolation] = []
        self.metrics = GuardianMetrics()
        
        # Monitoring and enforcement
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._action_queue = queue.Queue()
        self._enforcement_callbacks: Dict[str, Callable] = {}
        
        # Rate limiting and caching
        self._check_cache: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=30)
        
        # Initialize audit logger
        self.audit_logger = AuditLogger("sovereign_guardian")
        
        logger.info(f"Sovereign Guardian Agent initialized: {agent_id}")
    
    def _load_guardian_config(self) -> Dict[str, Any]:
        """Load guardian-specific configuration."""
        sovereign_config = self.crypto.sovereign_config
        return sovereign_config.get("guardian", {})
    
    async def start_monitoring(self):
        """Start the guardian monitoring system."""
        if self._running:
            logger.warning("Guardian monitoring already running")
            return
        
        self._running = True
        self.metrics.uptime_start = datetime.now()
        
        # Start background monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        # Start action processing
        asyncio.create_task(self._action_processor())
        
        await self.audit_logger.log_event({
            "event_type": "guardian_started",
            "agent_id": self.agent_id,
            "config": self.config
        })
        
        logger.info("Sovereign Guardian monitoring started")
    
    async def stop_monitoring(self):
        """Stop the guardian monitoring system."""
        self._running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        await self.audit_logger.log_event({
            "event_type": "guardian_stopped",
            "agent_id": self.agent_id,
            "uptime_seconds": (datetime.now() - self.metrics.uptime_start).total_seconds()
        })
        
        logger.info("Sovereign Guardian monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        check_interval = self.config.get("check_interval_seconds", 30)
        
        while self._running:
            try:
                self._perform_monitoring_cycle()
                self.metrics.last_check_time = datetime.now()
            except Exception as e:
                logger.error(f"Guardian monitoring cycle failed: {e}")
            
            time.sleep(check_interval)
    
    def _perform_monitoring_cycle(self):
        """Perform one complete monitoring cycle."""
        # Check contract expiries
        self._check_contract_expiries()
        
        # Validate active contracts
        self._validate_contracts()
        
        # Check system integrity
        self._check_system_integrity()
        
        # Process violation history for patterns
        self._analyze_violation_patterns()
        
        # Clean up old cache entries
        self._cleanup_cache()
    
    def register_contract(self, contract: AgentContract) -> bool:
        """Register an agent contract for monitoring."""
        try:
            # Verify contract signature
            if not self.crypto.verify_contract(contract):
                logger.error(f"Invalid contract signature for agent {contract.agent_id}")
                return False
            
            # Check if contract is expired
            if contract.is_expired():
                logger.error(f"Attempted to register expired contract for agent {contract.agent_id}")
                return False
            
            # Store contract
            self.active_contracts[contract.agent_id] = contract
            self.metrics.contracts_monitored += 1
            
            # Remove from suspended agents if present
            self.suspended_agents.discard(contract.agent_id)
            
            # Log contract registration
            asyncio.create_task(self.audit_logger.log_event({
                "event_type": "contract_registered",
                "agent_id": contract.agent_id,
                "contract": contract.to_dict(),
                "guardian_id": self.agent_id
            }))
            
            logger.info(f"Registered contract for agent {contract.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register contract for agent {contract.agent_id}: {e}")
            return False
    
    def check_action_authorization(self, agent_id: str, action: str, 
                                 resource: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if an agent action is authorized by its contract.
        This is the main enforcement point for all agent actions.
        """
        # Quick cache check to avoid repeated validations
        cache_key = f"{agent_id}:{action}:{resource}"
        if cache_key in self._check_cache:
            if datetime.now() - self._check_cache[cache_key] < self._cache_ttl:
                return True  # Recently validated
        
        try:
            # Check if agent is suspended
            if agent_id in self.suspended_agents:
                self._record_violation(
                    agent_id, "suspended_agent_action", 
                    f"Suspended agent {agent_id} attempted action: {action}",
                    "high"
                )
                return False
            
            # Check if agent has valid contract
            if agent_id not in self.active_contracts:
                self._record_violation(
                    agent_id, "no_contract",
                    f"Agent {agent_id} attempted action without contract: {action}",
                    "high"
                )
                return False
            
            contract = self.active_contracts[agent_id]
            
            # Check contract expiry
            if contract.is_expired():
                self._record_violation(
                    agent_id, "expired_contract",
                    f"Agent {agent_id} action with expired contract: {action}",
                    "medium"
                )
                return False
            
            # Check capability authorization
            if not self._check_capability_authorization(contract, action, resource, context):
                return False
            
            # Check resource limits
            if not self._check_resource_limits(contract, action, context):
                return False
            
            # Cache successful authorization
            self._check_cache[cache_key] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Authorization check failed for {agent_id}: {e}")
            self._record_violation(
                agent_id, "authorization_error",
                f"Authorization check error for action {action}: {str(e)}",
                "medium"
            )
            return False
    
    def _check_capability_authorization(self, contract: AgentContract, action: str,
                                      resource: Optional[str] = None,
                                      context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if action is authorized by contract capabilities."""
        
        # Map actions to required capabilities
        action_capability_map = {
            "memory_read": "memory_read",
            "memory_write": "memory_write",
            "file_read": "file_read",
            "file_write": "file_write",
            "network_request": "network_request",
            "system_command": "system_modify",
            "crypto_operation": "crypto_operations",
            "identity_access": "identity_access",
            "sovereign_control": "sovereign_control"
        }
        
        required_capability = action_capability_map.get(action, action)
        
        if not contract.has_capability(required_capability):
            self._record_violation(
                contract.agent_id, "capability_violation",
                f"Agent lacks required capability '{required_capability}' for action '{action}'",
                "high"
            )
            return False
        
        # Check for restricted capabilities
        restricted = self.crypto.sovereign_config.get("contracts", {}).get("restricted_capabilities", [])
        if required_capability in restricted:
            # Restricted capabilities require special authorization
            if not self._check_restricted_capability_auth(contract, required_capability, context):
                return False
        
        return True
    
    def _check_restricted_capability_auth(self, contract: AgentContract, capability: str,
                                        context: Optional[Dict[str, Any]]) -> bool:
        """Check authorization for restricted capabilities."""
        
        # Restricted capabilities need additional validation
        if capability == "sovereign_control":
            # Only allow sovereign control to the guardian itself
            if contract.agent_id != self.agent_id:
                self._record_violation(
                    contract.agent_id, "unauthorized_sovereign_control",
                    f"Non-guardian agent attempted sovereign control",
                    "critical"
                )
                return False
        
        elif capability == "crypto_operations":
            # Crypto operations need special context validation
            if not self._validate_crypto_operation_context(contract, context):
                return False
        
        elif capability == "identity_access":
            # Identity access is highly restricted
            if not self._validate_identity_access(contract, context):
                return False
        
        return True
    
    def _validate_crypto_operation_context(self, contract: AgentContract, 
                                         context: Optional[Dict[str, Any]]) -> bool:
        """Validate crypto operation context."""
        if not context:
            self._record_violation(
                contract.agent_id, "crypto_no_context",
                "Crypto operation without required context",
                "high"
            )
            return False
        
        # Check operation type
        operation = context.get("operation")
        if operation not in ["encrypt", "decrypt", "sign", "verify"]:
            self._record_violation(
                contract.agent_id, "invalid_crypto_operation",
                f"Invalid crypto operation: {operation}",
                "high"
            )
            return False
        
        return True
    
    def _validate_identity_access(self, contract: AgentContract,
                                context: Optional[Dict[str, Any]]) -> bool:
        """Validate identity access authorization."""
        # Identity access is only allowed for the guardian
        if contract.agent_id != self.agent_id:
            self._record_violation(
                contract.agent_id, "unauthorized_identity_access",
                "Non-guardian agent attempted identity access",
                "critical"
            )
            return False
        
        return True
    
    def _check_resource_limits(self, contract: AgentContract, action: str,
                             context: Optional[Dict[str, Any]]) -> bool:
        """Check if action violates resource limits."""
        limits = contract.resource_limits
        
        # TODO: Implement actual resource monitoring
        # For now, just check basic limits
        
        if context:
            # Check memory usage
            memory_usage = context.get("memory_usage_mb", 0)
            max_memory = limits.get("max_memory_mb", 1024)
            if memory_usage > max_memory:
                self._record_violation(
                    contract.agent_id, "memory_limit_exceeded",
                    f"Memory usage {memory_usage}MB exceeds limit {max_memory}MB",
                    "medium"
                )
                return False
            
            # Check file operations
            if action.startswith("file_"):
                file_ops = context.get("file_operations_count", 0)
                max_file_ops = limits.get("max_file_operations_per_hour", 500)
                if file_ops > max_file_ops:
                    self._record_violation(
                        contract.agent_id, "file_operations_limit_exceeded",
                        f"File operations {file_ops} exceeds hourly limit {max_file_ops}",
                        "medium"
                    )
                    return False
        
        return True
    
    def _record_violation(self, agent_id: str, violation_type: str, 
                         description: str, severity: str):
        """Record a contract violation and take appropriate action."""
        
        contract_id = self.active_contracts.get(agent_id, {}).get('nonce', 'unknown')
        
        violation = ContractViolation(
            agent_id=agent_id,
            contract_id=contract_id,
            violation_type=violation_type,
            description=description,
            severity=severity,
            timestamp=datetime.now(),
            action_taken=""  # Will be filled by enforcement action
        )
        
        self.violation_history.append(violation)
        self.metrics.violations_detected += 1
        
        # Take enforcement action
        action_taken = self._take_enforcement_action(violation)
        violation.action_taken = action_taken
        
        # Log to audit system
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "contract_violation",
            "violation": violation.to_dict(),
            "guardian_id": self.agent_id
        }))
        
        logger.warning(f"Contract violation: {violation_type} by {agent_id} - {description}")
    
    def _take_enforcement_action(self, violation: ContractViolation) -> str:
        """Take appropriate enforcement action based on violation."""
        enforcement_mode = self.config.get("enforcement_mode", "advisory")
        
        if enforcement_mode == "off":
            return "logged_only"
        
        elif enforcement_mode == "advisory":
            # Advisory mode: log and notify only
            self._notify_violation(violation)
            return "advisory_notification"
        
        elif enforcement_mode == "strict":
            # Strict mode: take progressively stronger action
            return self._escalated_enforcement(violation)
        
        return "unknown"
    
    def _escalated_enforcement(self, violation: ContractViolation) -> str:
        """Apply escalated enforcement based on violation severity and history."""
        agent_id = violation.agent_id
        severity = violation.severity
        
        # Count recent violations for this agent
        recent_violations = [
            v for v in self.violation_history[-100:]  # Check last 100 violations
            if v.agent_id == agent_id and 
            (datetime.now() - v.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        violation_count = len(recent_violations)
        
        if severity == "critical" or violation_count >= 5:
            # Suspend agent immediately
            self.suspended_agents.add(agent_id)
            self.metrics.agents_suspended += 1
            return "agent_suspended"
        
        elif severity == "high" or violation_count >= 3:
            # Force contract renewal
            if agent_id in self.active_contracts:
                del self.active_contracts[agent_id]
            return "forced_recontract"
        
        elif severity == "medium" or violation_count >= 2:
            # Warning and rate limiting
            self._apply_rate_limiting(agent_id)
            return "rate_limited"
        
        else:
            # Low severity: just log and notify
            self._notify_violation(violation)
            return "warning_issued"
    
    def _notify_violation(self, violation: ContractViolation):
        """Notify about contract violation."""
        # Add to action queue for async processing
        self._action_queue.put(("notify", violation))
    
    def _apply_rate_limiting(self, agent_id: str):
        """Apply rate limiting to an agent."""
        # Add to action queue for async processing
        self._action_queue.put(("rate_limit", agent_id))
    
    async def _action_processor(self):
        """Process enforcement actions asynchronously."""
        while self._running:
            try:
                if not self._action_queue.empty():
                    action_type, data = self._action_queue.get_nowait()
                    
                    if action_type == "notify":
                        await self._process_notification(data)
                    elif action_type == "rate_limit":
                        await self._process_rate_limiting(data)
                    
                    self.metrics.actions_taken += 1
                
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Action processor error: {e}")
    
    async def _process_notification(self, violation: ContractViolation):
        """Process violation notification."""
        # TODO: Implement notification system (email, webhook, etc.)
        logger.info(f"Notification sent for violation: {violation.violation_type}")
    
    async def _process_rate_limiting(self, agent_id: str):
        """Process rate limiting enforcement."""
        # TODO: Implement actual rate limiting mechanism
        logger.info(f"Rate limiting applied to agent: {agent_id}")
    
    def _check_contract_expiries(self):
        """Check for and handle contract expiries."""
        expired_agents = []
        
        for agent_id, contract in self.active_contracts.items():
            if contract.is_expired():
                expired_agents.append(agent_id)
                
                auto_renewal = self.crypto.sovereign_config.get("contracts", {}).get("auto_renewal", True)
                if auto_renewal:
                    # Attempt auto-renewal
                    try:
                        new_contract = self.crypto.create_agent_contract(
                            agent_id, contract.capabilities, contract.resource_limits
                        )
                        self.active_contracts[agent_id] = new_contract
                        
                        asyncio.create_task(self.audit_logger.log_event({
                            "event_type": "contract_auto_renewed",
                            "agent_id": agent_id,
                            "old_contract_id": contract.nonce,
                            "new_contract_id": new_contract.nonce
                        }))
                        
                        logger.info(f"Auto-renewed contract for agent {agent_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to auto-renew contract for {agent_id}: {e}")
                        expired_agents.append(agent_id)
        
        # Remove expired contracts that couldn't be renewed
        for agent_id in expired_agents:
            if agent_id in self.active_contracts:
                del self.active_contracts[agent_id]
                logger.warning(f"Removed expired contract for agent {agent_id}")
    
    def _validate_contracts(self):
        """Validate all active contracts."""
        invalid_agents = []
        
        for agent_id, contract in self.active_contracts.items():
            if not self.crypto.verify_contract(contract):
                invalid_agents.append(agent_id)
                self._record_violation(
                    agent_id, "invalid_contract_signature",
                    "Contract signature validation failed",
                    "critical"
                )
        
        # Remove invalid contracts
        for agent_id in invalid_agents:
            del self.active_contracts[agent_id]
            self.suspended_agents.add(agent_id)
    
    def _check_system_integrity(self):
        """Check overall system integrity."""
        try:
            # Check if sovereign identity is still valid
            identity = self.crypto.generate_identity()
            if not identity:
                logger.critical("Sovereign identity validation failed")
                self.metrics.system_interventions += 1
            
            # Check encrypted databases
            encrypted_dbs = self.crypto.sovereign_config.get("memory_encryption", {}).get("encrypted_databases", [])
            for db_path in encrypted_dbs:
                if os.path.exists(db_path):
                    # TODO: Implement integrity checking
                    pass
            
            # Check for suspicious system changes
            current_fingerprint = self.crypto._generate_os_fingerprint()
            # TODO: Compare with stored fingerprint and alert on changes
            
        except Exception as e:
            logger.error(f"System integrity check failed: {e}")
            self.metrics.system_interventions += 1
    
    def _analyze_violation_patterns(self):
        """Analyze violation patterns for threat detection."""
        if len(self.violation_history) < 10:
            return  # Not enough data
        
        # Check for patterns in recent violations
        recent_violations = [
            v for v in self.violation_history[-50:]  # Last 50 violations
            if (datetime.now() - v.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Group by agent
        agent_violations = {}
        for violation in recent_violations:
            if violation.agent_id not in agent_violations:
                agent_violations[violation.agent_id] = []
            agent_violations[violation.agent_id].append(violation)
        
        # Check for suspicious patterns
        for agent_id, violations in agent_violations.items():
            if len(violations) >= 5:  # High violation frequency
                logger.warning(f"High violation frequency detected for agent {agent_id}")
                self.suspended_agents.add(agent_id)
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self._check_cache.items()
            if now - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._check_cache[key]
    
    def get_status(self) -> Dict[str, Any]:
        """Get guardian status and metrics."""
        return {
            "guardian_id": self.agent_id,
            "running": self._running,
            "uptime_seconds": (datetime.now() - self.metrics.uptime_start).total_seconds(),
            "enforcement_mode": self.config.get("enforcement_mode", "advisory"),
            "metrics": {
                "contracts_monitored": self.metrics.contracts_monitored,
                "active_contracts": len(self.active_contracts),
                "violations_detected": self.metrics.violations_detected,
                "actions_taken": self.metrics.actions_taken,
                "agents_suspended": self.metrics.agents_suspended,
                "suspended_agents": list(self.suspended_agents),
                "system_interventions": self.metrics.system_interventions,
                "last_check_time": self.metrics.last_check_time.isoformat() if self.metrics.last_check_time else None
            },
            "recent_violations": [
                v.to_dict() for v in self.violation_history[-10:]  # Last 10 violations
            ]
        }
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown of all monitored agents."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Suspend all agents
        for agent_id in list(self.active_contracts.keys()):
            self.suspended_agents.add(agent_id)
        
        # Clear all contracts
        self.active_contracts.clear()
        
        # Log emergency action
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "emergency_shutdown",
            "reason": reason,
            "guardian_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        self.metrics.system_interventions += 1


# Global guardian instance
_guardian_instance = None

def get_sovereign_guardian() -> SovereignGuardianAgent:
    """Get or create global guardian instance."""
    global _guardian_instance
    if _guardian_instance is None:
        _guardian_instance = SovereignGuardianAgent()
    return _guardian_instance


# Contract enforcement decorator
def requires_contract(capabilities: List[str], resource_limits: Optional[Dict[str, Any]] = None):
    """Decorator to enforce contract requirements for agent methods."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            guardian = get_sovereign_guardian()
            
            # Check authorization
            if hasattr(self, 'agent_id'):
                agent_id = self.agent_id
            else:
                agent_id = getattr(self, 'id', 'unknown')
            
            action = func.__name__
            context = {
                "method": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            if not guardian.check_action_authorization(agent_id, action, context=context):
                raise PermissionError(f"Contract violation: {agent_id} not authorized for {action}")
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator