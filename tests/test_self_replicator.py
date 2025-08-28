#!/usr/bin/env python3
"""
Tests for Phase 22: Recursive Self-Replication Agent System

Tests fork creation, mutation validation, testing, and reintegration policies.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.self_replicator import AgentReplicator
from agents.mutation_utils import MutationEngine


class TestAgentReplicator:
    """Test suite for the AgentReplicator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / "agents"
        self.forks_dir = self.temp_dir / "agent_forks"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.agents_dir.mkdir(exist_ok=True)
        self.forks_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a test agent file
        self.test_agent_code = '''#!/usr/bin/env python3
"""Test Agent for Self-Replication Testing"""

from .base import BaseAgent

class TestAgent(BaseAgent):
    def __init__(self):
        super().__init__("test_agent")
    
    def get_agent_instructions(self):
        return "Test agent instructions"
    
    def respond(self, message, context=None):
        return {"response": "Test response", "agent": "test_agent"}
    
    def test_function(self):
        """A test function for mutation"""
        return "original_result"
'''
        
        self.test_agent_file = self.agents_dir / "test_agent.py"
        with open(self.test_agent_file, 'w') as f:
            f.write(self.test_agent_code)
        
        # Mock the config
        self.mock_config = {
            'mutation_rate': 0.3,
            'max_forks': 5,
            'survival_threshold': 0.7
        }
        
        # Create replicator with mocked paths
        with patch('agents.self_replicator.get_config') as mock_get_config:
            mock_get_config.return_value = {'self_replication': self.mock_config}
            self.replicator = AgentReplicator()
            
            # Override paths to use temp directory
            self.replicator.agents_dir = self.agents_dir
            self.replicator.forks_dir = self.forks_dir
            self.replicator.logs_dir = self.output_dir
            self.replicator.fork_logs_path = self.output_dir / "fork_logs.jsonl"
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_fork_agent_success(self):
        """Test successful agent forking"""
        fork_id = self.replicator.fork_agent("test_agent")
        
        assert fork_id is not None
        assert len(fork_id) == 36  # UUID length
        assert fork_id in self.replicator.active_forks
        
        # Check fork directory and file exist
        fork_dir = self.forks_dir / fork_id
        assert fork_dir.exists()
        
        fork_file = fork_dir / f"test_agent_{fork_id[:8]}.py"
        assert fork_file.exists()
        
        # Verify fork tracking
        fork_info = self.replicator.active_forks[fork_id]
        assert fork_info['agent_name'] == 'test_agent'
        assert fork_info['mutations'] == 0
        assert fork_info['tested'] == False
        assert fork_info['status'] == 'created'
    
    def test_fork_agent_nonexistent(self):
        """Test forking a nonexistent agent"""
        fork_id = self.replicator.fork_agent("nonexistent_agent")
        assert fork_id is None
    
    def test_fork_agent_max_limit(self):
        """Test fork creation at maximum limit"""
        # Create forks up to the limit
        fork_ids = []
        for i in range(self.mock_config['max_forks']):
            fork_id = self.replicator.fork_agent("test_agent")
            assert fork_id is not None
            fork_ids.append(fork_id)
        
        # Next fork should fail
        fork_id = self.replicator.fork_agent("test_agent")
        assert fork_id is None
        
        assert len(self.replicator.active_forks) == self.mock_config['max_forks']
    
    def test_mutate_agent_success(self):
        """Test successful agent mutation"""
        # First fork an agent
        fork_id = self.replicator.fork_agent("test_agent")
        assert fork_id is not None
        
        fork_info = self.replicator.active_forks[fork_id]
        original_mutations = fork_info['mutations']
        
        # Mock the mutation engine
        with patch('agents.self_replicator.MutationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.mutate_file.return_value = True
            mock_engine_class.return_value = mock_engine
            
            success = self.replicator.mutate_agent(fork_info['fork_file'])
            
            assert success == True
            assert fork_info['mutations'] == original_mutations + 1
            assert fork_info['status'] == 'mutated'
    
    def test_mutate_agent_nonexistent_file(self):
        """Test mutation of nonexistent file"""
        success = self.replicator.mutate_agent("/nonexistent/file.py")
        assert success == False
    
    def test_test_agent_syntax_validation(self):
        """Test agent testing with syntax validation"""
        # Fork and create a test agent
        fork_id = self.replicator.fork_agent("test_agent")
        fork_info = self.replicator.active_forks[fork_id]
        agent_name = f"test_agent_{fork_id[:8]}"
        
        # Test with valid syntax
        results = self.replicator.test_agent(agent_name)
        
        assert 'error' not in results
        assert results['fork_id'] == fork_id
        assert results['syntax_valid'] == True
        assert 'functionality_score' in results
        assert 'overall_score' in results
        
        # Verify fork tracking is updated
        assert fork_info['tested'] == True
        assert fork_info['score'] is not None
        assert fork_info['status'] == 'tested'
    
    def test_test_agent_invalid_syntax(self):
        """Test agent testing with invalid syntax"""
        # Fork an agent
        fork_id = self.replicator.fork_agent("test_agent")
        fork_info = self.replicator.active_forks[fork_id]
        
        # Corrupt the fork file to create syntax error
        fork_file = Path(fork_info['fork_file'])
        with open(fork_file, 'w') as f:
            f.write("invalid python syntax $$$ !!!")
        
        agent_name = f"test_agent_{fork_id[:8]}"
        results = self.replicator.test_agent(agent_name)
        
        assert results['syntax_valid'] == False
        assert results['overall_score'] == 0.0
    
    def test_evaluate_performance(self):
        """Test performance evaluation from logs"""
        # Create test log entries
        import json
        import time
        
        test_logs = [
            {
                'event': 'agent_tested',
                'fork_id': 'test-fork-1',
                'score': 0.8,
                'timestamp': time.time()
            },
            {
                'event': 'agent_tested', 
                'fork_id': 'test-fork-2',
                'score': 0.6,
                'timestamp': time.time()
            },
            {
                'event': 'agent_tested',
                'fork_id': 'test-fork-1',  # Second test of same fork
                'score': 0.85,
                'timestamp': time.time()
            }
        ]
        
        # Write test logs
        with open(self.replicator.fork_logs_path, 'w') as f:
            for log in test_logs:
                f.write(json.dumps(log) + '\n')
        
        results = self.replicator.evaluate_performance(str(self.replicator.fork_logs_path))
        
        assert 'error' not in results
        assert results['total_log_entries'] == 3
        assert results['tested_forks'] == 2
        assert 'test-fork-1' in results['average_scores']
        assert 'test-fork-2' in results['average_scores']
        
        # Check that fork-1 has higher average (0.825) than fork-2 (0.6)
        assert results['average_scores']['test-fork-1'] > results['average_scores']['test-fork-2']
        
        # Check ranking
        assert results['ranked_performance'][0][0] == 'test-fork-1'  # Top performer
    
    def test_reintegration_policy(self):
        """Test reintegration policy decisions"""
        # Create test forks with different scores
        fork_id_1 = self.replicator.fork_agent("test_agent")
        fork_id_2 = self.replicator.fork_agent("test_agent")
        
        # Set test scores
        self.replicator.active_forks[fork_id_1]['tested'] = True
        self.replicator.active_forks[fork_id_1]['score'] = 0.8  # Above threshold
        
        self.replicator.active_forks[fork_id_2]['tested'] = True
        self.replicator.active_forks[fork_id_2]['score'] = 0.5  # Below threshold
        
        results = self.replicator.reintegration_policy()
        
        assert results['decisions'][fork_id_1] == 'keep'
        assert results['decisions'][fork_id_2] == 'prune'
        assert fork_id_1 in results['survivors']
        assert fork_id_2 in results['pruned']
        assert results['survival_threshold'] == self.mock_config['survival_threshold']
    
    def test_prune_forks(self):
        """Test fork pruning functionality"""
        # Create test forks
        fork_id_1 = self.replicator.fork_agent("test_agent")
        fork_id_2 = self.replicator.fork_agent("test_agent")
        
        # Set scores for pruning
        self.replicator.active_forks[fork_id_1]['tested'] = True
        self.replicator.active_forks[fork_id_1]['score'] = 0.8  # Keep
        
        self.replicator.active_forks[fork_id_2]['tested'] = True
        self.replicator.active_forks[fork_id_2]['score'] = 0.5  # Prune
        
        initial_fork_count = len(self.replicator.active_forks)
        results = self.replicator.prune_forks()
        
        assert results['pruned_count'] == 1
        assert results['remaining_forks'] == initial_fork_count - 1
        assert fork_id_1 in self.replicator.active_forks
        assert fork_id_2 not in self.replicator.active_forks
        
        # Check that fork directory was removed
        fork_dir_2 = self.forks_dir / fork_id_2
        assert not fork_dir_2.exists()
    
    def test_response_handling(self):
        """Test agent response handling for different message types"""
        # Test fork request
        response = self.replicator.respond("fork test_agent")
        assert response['agent'] == 'agent_replicator'
        assert 'fork_id' in response or 'response' in response
        
        # Test status request
        response = self.replicator.respond("status")
        assert response['agent'] == 'agent_replicator'
        assert 'response' in response
        
        # Test general info
        response = self.replicator.respond("hello")
        assert response['agent'] == 'agent_replicator'
        assert 'Agent Replicator active' in response['response']


class TestMutationEngine:
    """Test suite for the MutationEngine class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mutation_engine = MutationEngine(mutation_rate=0.5)
        
        self.test_code = '''#!/usr/bin/env python3
"""Test code for mutation testing"""

def test_function():
    """A test function"""
    result = "hello world"
    return result

class TestClass:
    def respond(self, message):
        return {"response": "test"}
'''
        
        self.test_file = self.temp_dir / "test_code.py"
        with open(self.test_file, 'w') as f:
            f.write(self.test_code)
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_syntax_validation_valid(self):
        """Test syntax validation with valid code"""
        assert self.mutation_engine.validate_syntax(self.test_code) == True
    
    def test_syntax_validation_invalid(self):
        """Test syntax validation with invalid code"""
        invalid_code = "def invalid_function(:\n    return 'broken'"
        assert self.mutation_engine.validate_syntax(invalid_code) == False
    
    def test_mutate_file_preserves_syntax(self):
        """Test that file mutation preserves valid syntax"""
        original_content = self.test_code
        
        # Apply mutations multiple times
        for _ in range(3):
            success = self.mutation_engine.mutate_file(str(self.test_file))
            
            # Read mutated content
            with open(self.test_file, 'r') as f:
                mutated_content = f.read()
            
            # Verify syntax is still valid
            assert self.mutation_engine.validate_syntax(mutated_content) == True
    
    def test_function_name_mutation(self):
        """Test function name mutation strategy"""
        original_content = self.test_code
        mutated_content = self.mutation_engine._mutate_function_names(original_content)
        
        # Mutations might or might not occur due to randomness
        # Just verify syntax is preserved
        assert self.mutation_engine.validate_syntax(mutated_content) == True
    
    def test_string_literal_mutation(self):
        """Test string literal mutation strategy"""
        original_content = self.test_code
        mutated_content = self.mutation_engine._mutate_string_literals(original_content)
        
        # Verify syntax is preserved
        assert self.mutation_engine.validate_syntax(mutated_content) == True
    
    def test_mutation_summary(self):
        """Test mutation summary generation"""
        original = "def old_function():\n    return 'old'"
        mutated = "def new_function():\n    return 'new'"
        
        summary = self.mutation_engine.get_mutation_summary(original, mutated)
        
        assert 'total_lines' in summary
        assert 'changed_lines' in summary
        assert 'mutation_rate' in summary
        assert 'changes' in summary
        assert summary['changed_lines'] == 2  # Both lines changed


class TestIntegration:
    """Integration tests for the complete replication system"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / "agents"
        self.forks_dir = self.temp_dir / "agent_forks"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.agents_dir.mkdir(exist_ok=True)
        self.forks_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create test agent
        test_agent_code = '''#!/usr/bin/env python3
from .base import BaseAgent

class TestIntegrationAgent(BaseAgent):
    def __init__(self):
        super().__init__("test_integration")
    
    def get_agent_instructions(self):
        return "Integration test agent"
    
    def respond(self, message, context=None):
        return {"response": f"Processed: {message}", "agent": "test_integration"}
'''
        
        with open(self.agents_dir / "test_integration.py", 'w') as f:
            f.write(test_agent_code)
        
        # Create replicator
        mock_config = {
            'mutation_rate': 0.2,
            'max_forks': 3,
            'survival_threshold': 0.6
        }
        
        with patch('agents.self_replicator.get_config') as mock_get_config:
            mock_get_config.return_value = {'self_replication': mock_config}
            self.replicator = AgentReplicator()
            
            # Override paths
            self.replicator.agents_dir = self.agents_dir
            self.replicator.forks_dir = self.forks_dir
            self.replicator.logs_dir = self.output_dir
            self.replicator.fork_logs_path = self.output_dir / "fork_logs.jsonl"
    
    def teardown_method(self):
        """Clean up integration test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_fork_lifecycle(self):
        """Test complete fork lifecycle: create, mutate, test, evaluate, prune"""
        # 1. Fork agent
        fork_id = self.replicator.fork_agent("test_integration")
        assert fork_id is not None
        
        # 2. Mutate agent
        fork_info = self.replicator.active_forks[fork_id]
        
        with patch('agents.self_replicator.MutationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.mutate_file.return_value = True
            mock_engine_class.return_value = mock_engine
            
            success = self.replicator.mutate_agent(fork_info['fork_file'])
            assert success == True
        
        # 3. Test agent
        agent_name = f"test_integration_{fork_id[:8]}"
        results = self.replicator.test_agent(agent_name)
        assert 'error' not in results
        assert results['syntax_valid'] == True
        
        # 4. Evaluate performance
        performance = self.replicator.evaluate_performance(str(self.replicator.fork_logs_path))
        assert 'error' not in performance
        assert performance['tested_forks'] >= 1
        
        # 5. Apply reintegration policy
        policy = self.replicator.reintegration_policy()
        assert fork_id in policy['decisions']
        
        # 6. Prune if necessary
        prune_results = self.replicator.prune_forks()
        assert 'pruned_count' in prune_results
        assert 'remaining_forks' in prune_results
    
    def test_fork_isolation(self):
        """Test that forks are properly isolated from each other"""
        # Create multiple forks
        fork_id_1 = self.replicator.fork_agent("test_integration")
        fork_id_2 = self.replicator.fork_agent("test_integration")
        
        assert fork_id_1 != fork_id_2
        assert fork_id_1 in self.replicator.active_forks
        assert fork_id_2 in self.replicator.active_forks
        
        # Check that each fork has its own directory
        fork_dir_1 = self.forks_dir / fork_id_1
        fork_dir_2 = self.forks_dir / fork_id_2
        
        assert fork_dir_1.exists()
        assert fork_dir_2.exists()
        assert fork_dir_1 != fork_dir_2
        
        # Check that mutations to one don't affect the other
        fork_info_1 = self.replicator.active_forks[fork_id_1]
        fork_info_2 = self.replicator.active_forks[fork_id_2]
        
        with patch('agents.self_replicator.MutationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.mutate_file.return_value = True
            mock_engine_class.return_value = mock_engine
            
            # Mutate only first fork
            self.replicator.mutate_agent(fork_info_1['fork_file'])
            
            assert fork_info_1['mutations'] == 1
            assert fork_info_2['mutations'] == 0  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])