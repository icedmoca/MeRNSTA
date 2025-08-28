#!/usr/bin/env python3
"""
Phase 22: Recursive Self-Replication System Demo

This demonstrates the complete self-replication capabilities of MeRNSTA,
including agent forking, mutation, testing, and evolution cycles.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.self_replicator import AgentReplicator
from agents.mutation_utils import MutationEngine
from agents.registry import get_agent_registry


def demo_basic_fork_and_mutate():
    """Demonstrate basic fork and mutation functionality"""
    print("🔬 Phase 22 Self-Replication Demo")
    print("=" * 50)
    
    # Get the agent replicator
    replicator = AgentReplicator()
    
    print(f"📊 Initial State:")
    print(f"   Active forks: {len(replicator.active_forks)}")
    print(f"   Max forks: {replicator.max_forks}")
    print(f"   Mutation rate: {replicator.mutation_rate}")
    print(f"   Survival threshold: {replicator.survival_threshold}")
    print()
    
    # Fork a critic agent
    print("🍴 Forking critic agent...")
    fork_id = replicator.fork_agent("critic")
    
    if fork_id:
        print(f"   ✅ Successfully created fork: {fork_id[:8]}")
        
        # Get fork info
        fork_info = replicator.active_forks[fork_id]
        print(f"   📁 Fork file: {fork_info['fork_file']}")
        print(f"   📅 Created: {fork_info['created_iso']}")
        print()
        
        # Apply mutations
        print("🧬 Applying mutations...")
        success = replicator.mutate_agent(fork_info['fork_file'])
        
        if success:
            print(f"   ✅ Mutations applied successfully")
            print(f"   🔄 Mutation count: {fork_info['mutations']}")
            print()
            
            # Test the mutated agent
            print("🧪 Testing mutated agent...")
            agent_name = f"critic_{fork_id[:8]}"
            test_results = replicator.test_agent(agent_name)
            
            if 'error' not in test_results:
                print(f"   ✅ Testing completed")
                print(f"   🎯 Overall score: {test_results['overall_score']:.2f}")
                print(f"   ✓ Syntax valid: {test_results['syntax_valid']}")
                print()
                
                # Show fork statistics
                print("📈 Fork Statistics:")
                stats = replicator.get_fork_statistics()
                print(f"   Total forks created: {stats['total_forks_created']}")
                print(f"   Active forks: {stats['active_forks']}")
                print(f"   Capacity used: {stats['fork_capacity_used']:.1%}")
                
                if 'score_statistics' in stats:
                    score_stats = stats['score_statistics']
                    print(f"   Average score: {score_stats['average']:.2f}")
                    print(f"   Survivors: {score_stats['survivors']}")
                print()
                
            else:
                print(f"   ❌ Testing failed: {test_results['error']}")
        else:
            print(f"   ❌ Mutation failed")
    else:
        print(f"   ❌ Fork creation failed")
    
    return replicator


def demo_automated_replication_cycle():
    """Demonstrate automated replication cycle"""
    print("🔄 Automated Replication Cycle Demo")
    print("=" * 50)
    
    # Get the reflection orchestrator
    registry = get_agent_registry()
    orchestrator = registry.get_agent('reflection_orchestrator')
    
    if not orchestrator or not hasattr(orchestrator, 'process_automated_replication_cycle'):
        print("❌ Reflection orchestrator with replication support not available")
        return
    
    # Simulate some contradictions to trigger replication
    orchestrator.contradiction_count = 6  # Above threshold of 5
    orchestrator.uncertainty_events = 4   # Above threshold of 3
    
    print(f"📊 Pre-cycle state:")
    print(f"   Contradictions: {orchestrator.contradiction_count}")
    print(f"   Uncertainty events: {orchestrator.uncertainty_events}")
    print()
    
    # Run automated cycle
    print("🚀 Running automated replication cycle...")
    results = orchestrator.process_automated_replication_cycle()
    
    print(f"📋 Cycle Results:")
    print(f"   Status: {results['status']}")
    print(f"   Replications triggered: {results['replications_triggered']}")
    print(f"   Tests completed: {results['tests_completed']}")
    print(f"   Prunes performed: {results['prunes_performed']}")
    print()
    
    if results['actions']:
        print("📝 Actions taken:")
        for action in results['actions']:
            print(f"   • {action}")
    else:
        print("   No actions required")
    print()


def demo_mutation_engine():
    """Demonstrate mutation engine capabilities"""
    print("🧬 Mutation Engine Demo")
    print("=" * 50)
    
    # Create test code
    test_code = '''#!/usr/bin/env python3
"""Test agent for mutation demonstration"""

class TestAgent:
    def __init__(self):
        self.name = "test_agent"
        self.timeout = 30
    
    def respond(self, message):
        """Process a message and return response"""
        if message == "hello":
            return "Hello world"
        elif message == "error":
            raise Exception("Test error occurred")
        return "Unknown message"
    
    def analyze(self, data):
        result = self.process_data(data)
        return result
'''
    
    # Create mutation engine
    mutation_engine = MutationEngine(mutation_rate=0.5)
    
    print("📝 Original code:")
    print(test_code[:200] + "..." if len(test_code) > 200 else test_code)
    print()
    
    # Test individual mutation strategies
    strategies = [
        ('Function names', mutation_engine._mutate_function_names),
        ('String literals', mutation_engine._mutate_string_literals),
        ('Numeric constants', mutation_engine._mutate_numeric_constants),
        ('Error messages', mutation_engine._mutate_error_messages)
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"🔧 Testing {strategy_name} mutation...")
        mutated = strategy_func(test_code)
        
        if mutated != test_code:
            print(f"   ✅ Mutation applied")
            # Show first difference
            original_lines = test_code.split('\n')
            mutated_lines = mutated.split('\n')
            
            for i, (orig, mut) in enumerate(zip(original_lines, mutated_lines)):
                if orig != mut:
                    print(f"   Line {i+1}: '{orig.strip()}' → '{mut.strip()}'")
                    break
        else:
            print(f"   ➖ No mutations applied (random chance)")
        print()
    
    # Test syntax validation
    print("✅ Testing syntax validation...")
    valid = mutation_engine.validate_syntax(test_code)
    print(f"   Original code valid: {valid}")
    
    invalid_code = "def broken_syntax(:\n    return 'error'"
    valid = mutation_engine.validate_syntax(invalid_code)
    print(f"   Invalid code valid: {valid}")
    print()


def demo_configuration_tuning():
    """Demonstrate configuration tuning capabilities"""
    print("⚙️ Configuration Tuning Demo")
    print("=" * 50)
    
    replicator = AgentReplicator()
    
    print("📋 Current configuration:")
    print(f"   Mutation rate: {replicator.mutation_rate}")
    print(f"   Survival threshold: {replicator.survival_threshold}")
    print(f"   Max forks: {replicator.max_forks}")
    print()
    
    # Simulate tuning commands
    print("🔧 Tuning parameters...")
    
    # Increase mutation rate for more aggressive evolution
    old_rate = replicator.mutation_rate
    replicator.mutation_rate = 0.4
    print(f"   Mutation rate: {old_rate:.2f} → {replicator.mutation_rate:.2f}")
    
    # Lower survival threshold to keep more variants
    old_threshold = replicator.survival_threshold
    replicator.survival_threshold = 0.6
    print(f"   Survival threshold: {old_threshold:.2f} → {replicator.survival_threshold:.2f}")
    
    # Increase max forks for larger population
    old_max = replicator.max_forks
    replicator.max_forks = 15
    print(f"   Max forks: {old_max} → {replicator.max_forks}")
    print()
    
    print("✅ Configuration updated successfully!")
    print("💡 Use '/tune_replication <parameter> <value>' in CLI for real-time tuning")
    print()


def main():
    """Run the complete Phase 22 demo"""
    print("🚀 MeRNSTA Phase 22: Recursive Self-Replication System")
    print("=" * 60)
    print()
    
    try:
        # Demo 1: Basic fork and mutate
        replicator = demo_basic_fork_and_mutate()
        time.sleep(1)
        
        # Demo 2: Mutation engine
        demo_mutation_engine()
        time.sleep(1)
        
        # Demo 3: Configuration tuning
        demo_configuration_tuning()
        time.sleep(1)
        
        # Demo 4: Automated replication cycle
        demo_automated_replication_cycle()
        
        print("🎉 Phase 22 demo completed successfully!")
        print()
        print("💡 Available CLI commands:")
        print("   /fork_agent <agent_name>     - Create agent fork")
        print("   /mutate_agent <fork_id>      - Apply mutations to fork")
        print("   /run_fork <fork_id>          - Test fork performance")
        print("   /score_forks                 - Show fork scores")
        print("   /prune_forks                 - Remove underperformers")
        print("   /fork_status                 - Show all fork status")
        print("   /replication_cycle           - Run automated evolution")
        print("   /tune_replication <p> <v>    - Tune parameters")
        print()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()