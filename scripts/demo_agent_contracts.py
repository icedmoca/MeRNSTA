#!/usr/bin/env python3
"""
Demonstration script for Phase 26: Declarative Agent Contracts & Role Specialization

This script showcases the contract-based agent system functionality:
- Agent contract loading and inspection
- Task alignment scoring
- Contract-based routing
- Performance feedback and learning
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_contract import load_or_create_contract, score_all_agents_for_task
from agents.meta_router import MetaRouter
from agents.planner import PlannerAgent


def demo_contract_loading():
    """Demonstrate loading and inspecting agent contracts."""
    print("🏷️  PHASE 26 DEMO: Agent Contract Loading & Inspection")
    print("=" * 60)
    
    # Load some agent contracts
    agent_names = ["planner", "critic", "debater", "reflector"]
    
    for agent_name in agent_names:
        print(f"\n📋 Loading contract for {agent_name.title()}Agent:")
        contract = load_or_create_contract(agent_name)
        
        if contract:
            print(f"   Purpose: {contract.purpose}")
            print(f"   Version: {contract.version}")
            print(f"   Capabilities: {len(contract.capabilities)} defined")
            
            # Show top 3 confidence areas
            top_confidence = sorted(contract.confidence_vector.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top Skills: {', '.join(f'{skill}({conf:.2f})' for skill, conf in top_confidence)}")
            
            if contract.weaknesses:
                print(f"   Known Weaknesses: {', '.join(contract.weaknesses[:2])}{'...' if len(contract.weaknesses) > 2 else ''}")
        else:
            print(f"   ❌ Failed to load contract")
    
    print()


def demo_task_alignment_scoring():
    """Demonstrate task alignment scoring with different tasks."""
    print("🎯 DEMO: Task Alignment Scoring")
    print("=" * 40)
    
    test_tasks = [
        "Create a comprehensive project plan for a new mobile app",
        "Analyze system performance bottlenecks and optimization opportunities", 
        "Debug memory leaks causing application crashes",
        "Facilitate a design debate between competing architectural approaches",
        "Reflect on recent decision-making processes and identify improvements"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🔍 Task {i}: {task}")
        print("-" * 50)
        
        # Score all agents for this task
        agent_scores = score_all_agents_for_task(task)
        
        if agent_scores:
            print(f"📊 Top 3 Best-Aligned Agents:")
            for j, agent_result in enumerate(agent_scores[:3], 1):
                score = agent_result['alignment_score']
                name = agent_result['agent_name']
                
                # Visual score representation
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                confidence = "High" if score > 0.7 else "Med" if score > 0.4 else "Low"
                print(f"   {j}. {name:<15} [{bar}] {score:.3f} ({confidence})")
        else:
            print("   ❌ No agents available for scoring")
    
    print()


def demo_contract_based_routing():
    """Demonstrate contract-based routing through MetaRouter."""
    print("🧭 DEMO: Contract-Based Routing")
    print("=" * 35)
    
    try:
        # Initialize MetaRouter
        router = MetaRouter()
        
        # Test routing with different types of tasks
        routing_tasks = [
            {
                "task": "Design system architecture for scalable microservices",
                "context": {"urgency": 0.6, "complexity": 0.8}
            },
            {
                "task": "Fix critical bug in payment processing",
                "context": {"urgency": 0.9, "complexity": 0.5}
            },
            {
                "task": "Plan sprint goals and user story prioritization",
                "context": {"urgency": 0.4, "complexity": 0.6}
            }
        ]
        
        for i, task_info in enumerate(routing_tasks, 1):
            task = task_info["task"]
            context = task_info["context"]
            
            print(f"\n🚀 Routing Task {i}: {task}")
            print(f"   Context: Urgency={context['urgency']}, Complexity={context['complexity']}")
            
            # Use contract-based routing
            routing_result = router.route_task_with_contracts(task, context)
            
            if routing_result['success']:
                selected_agent = routing_result['selected_agent']
                confidence = routing_result['routing_confidence']
                
                print(f"   ✅ Selected: {selected_agent['agent_name']} (Score: {confidence:.3f})")
                print(f"   📝 Purpose: {selected_agent['purpose'][:60]}...")
                
                # Show alternatives
                alternatives = routing_result.get('alternatives', [])
                if alternatives:
                    print(f"   🔄 Alternatives: {', '.join(alt['agent_name'] for alt in alternatives[:2])}")
            else:
                print(f"   ❌ Routing failed: {routing_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"   ❌ Router demo failed: {e}")
    
    print()


def demo_performance_feedback():
    """Demonstrate performance feedback and learning."""
    print("📈 DEMO: Performance Feedback & Learning")
    print("=" * 42)
    
    # Load a contract for demonstration
    contract = load_or_create_contract("planner")
    
    if not contract:
        print("❌ Could not load planner contract for demo")
        return
    
    print(f"📋 Agent: {contract.agent_name}")
    print(f"   Initial planning confidence: {contract.confidence_vector.get('planning', 0.5):.3f}")
    
    # Simulate some performance feedback
    feedback_scenarios = [
        {
            "scenario": "Excellent project breakdown",
            "feedback": {
                "task_type": "planning",
                "success": True,
                "performance_score": 0.9,
                "quality_rating": 0.95,
                "notes": "Created very detailed and actionable project plan"
            }
        },
        {
            "scenario": "Good analysis work",
            "feedback": {
                "task_type": "analysis", 
                "success": True,
                "performance_score": 0.8,
                "quality_rating": 0.75,
                "notes": "Thorough analysis of system requirements"
            }
        },
        {
            "scenario": "Struggled with debugging",
            "feedback": {
                "task_type": "debugging",
                "success": False,
                "performance_score": 0.3,
                "quality_rating": 0.4,
                "notes": "Had difficulty identifying root cause of the issue"
            }
        }
    ]
    
    for i, scenario in enumerate(feedback_scenarios, 1):
        print(f"\n   📝 Feedback {i}: {scenario['scenario']}")
        
        # Get confidence before update
        task_type = scenario['feedback']['task_type']
        confidence_key = {'planning': 'planning', 'analysis': 'analysis', 'debugging': 'debugging'}.get(task_type, 'problem_solving')
        before_confidence = contract.confidence_vector.get(confidence_key, 0.5)
        
        # Apply feedback
        contract.update_from_performance_feedback(scenario['feedback'])
        
        # Get confidence after update
        after_confidence = contract.confidence_vector.get(confidence_key, 0.5)
        
        # Show change
        change = after_confidence - before_confidence
        change_str = f"{change:+.3f}" if change != 0 else "no change"
        trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"      {trend} {task_type} confidence: {before_confidence:.3f} → {after_confidence:.3f} ({change_str})")
    
    print(f"\n   📊 Performance history: {len(contract.performance_history)} entries")
    
    # Show specialization trends if available
    if contract.specialization_drift:
        print(f"   🔄 Specialization trends:")
        for task_type, trend_data in contract.specialization_drift.items():
            if len(trend_data) >= 3:
                trend = contract.get_specialization_trend(task_type)
                if trend is not None:
                    trend_icon = "📈" if trend > 0.05 else "📉" if trend < -0.05 else "➡️"
                    print(f"      {task_type}: {trend_icon} {trend:+.3f}")
    
    print()


def demo_agent_instantiation():
    """Demonstrate that existing agents automatically load contracts."""
    print("🤖 DEMO: Agent Contract Integration")
    print("=" * 38)
    
    try:
        # Create an agent instance - should automatically load contract
        planner = PlannerAgent()
        
        print(f"🧠 Created {planner.name}Agent instance")
        
        if hasattr(planner, 'contract') and planner.contract:
            contract = planner.contract
            print(f"   ✅ Contract loaded automatically: {contract.agent_name}")
            print(f"   📝 Purpose: {contract.purpose}")
            print(f"   📊 Confidence vector: {len(contract.confidence_vector)} skills defined")
            
            # Test task alignment through agent
            test_task = "Break down a complex software project into manageable tasks"
            if hasattr(planner, 'score_task_alignment'):
                alignment_score = planner.score_task_alignment(test_task)
                print(f"   🎯 Task alignment score: {alignment_score:.3f}")
            
            # Show agent capabilities including contract info
            capabilities = planner.get_capabilities()
            print(f"   🔧 Enhanced capabilities:")
            print(f"      - Contract loaded: {capabilities.get('contract_loaded', False)}")
            print(f"      - Contract version: {capabilities.get('contract_version', 'N/A')}")
            print(f"      - Last updated: {capabilities.get('last_contract_update', 'N/A')[:10]}")
            
        else:
            print(f"   ❌ No contract loaded for {planner.name}")
    
    except Exception as e:
        print(f"   ❌ Agent instantiation demo failed: {e}")
    
    print()


def main():
    """Run all demonstration functions."""
    print("🚀 MeRNSTA Phase 26: Declarative Agent Contracts Demo")
    print("=" * 60)
    print("This demo showcases the new contract-based agent system.")
    print()
    
    try:
        demo_contract_loading()
        demo_task_alignment_scoring()
        demo_contract_based_routing()
        demo_performance_feedback()
        demo_agent_instantiation()
        
        print("✅ DEMO COMPLETE!")
        print("\nKey Phase 26 Features Demonstrated:")
        print("  📋 Agent contracts with declarative role specifications")
        print("  🎯 Intelligent task alignment scoring")
        print("  🧭 Contract-based routing for optimal agent selection")
        print("  📈 Performance feedback and adaptive learning")
        print("  🔗 Seamless integration with existing agent infrastructure")
        print("\nTry these CLI commands:")
        print("  • list_contracts - See all available agent contracts")
        print("  • contract planner - View detailed contract information")
        print("  • score_task 'your task description' - Find the best agent")
        print("  • update_contract planner success=true score=0.9 - Provide feedback")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()