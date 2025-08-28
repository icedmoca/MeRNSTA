"""
CLI commands for MeRNSTA
"""
import sys
import time
import logging
from typing import List, Dict
from storage.memory_log import MemoryLog
from storage.memory_utils import detect_contradictions, get_contradiction_summary_by_clusters, _normalize_subject, calculate_contradiction_score, TripletFact, calculate_agreement_score
from storage.auto_reconciliation import AutoReconciliationEngine
from storage.memory_compression import MemoryCompressionEngine
from config.settings import CONTRADICTION_SCORE_THRESHOLD, VERBOSITY_LEVEL, enable_compression
from .response_generation import generate_response
from .memory_ops import process_user_input  # If needed
from .meta_goals import execute_meta_goals  # If needed
# Optional imports for dashboard functionality
try:
    import uvicorn
    from config.settings import dashboard_port
    from api.main import app as fastapi_app
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Import enhanced components if available
try:
    from storage.enhanced_memory_system import EnhancedMemorySystem
    ENHANCED_MODE = True
except ImportError:
    ENHANCED_MODE = False

# Import Phase 13 Command Router components
try:
    from agents.command_router import get_command_router, route_command
    from agents.registry import get_tool_registry, get_agent_registry
    from storage.tool_use_log import get_tool_logger
    COMMAND_ROUTER_MODE = True
except ImportError:
    COMMAND_ROUTER_MODE = False

# Import Phase 7 components
try:
    from storage.drive_system import get_drive_system
    from agents.intent_modeler import get_intent_modeler
    from storage.self_model import get_self_aware_model
    PHASE7_MODE = True
except ImportError:
    PHASE7_MODE = False

# Import Phase 11 Recursive Planning components
try:
    from agents.recursive_planner import RecursivePlanner
    from storage.plan_memory import PlanMemory
    from storage.intention_model import IntentionModel
    from agents.self_prompter import SelfPromptGenerator
    RECURSIVE_PLANNING_MODE = True
except ImportError:
    RECURSIVE_PLANNING_MODE = False

# Import Phase 12 Self-Repair components
try:
    from agents.self_healer import SelfHealer
    from storage.self_repair_log import SelfRepairLog
    SELF_REPAIR_MODE = True
except ImportError:
    SELF_REPAIR_MODE = False

# Import Phase 14 Recursive Execution components
try:
    from agents.file_writer import get_file_writer, write_and_execute
    from agents.execution_monitor import get_execution_monitor
    from agents.edit_loop import get_edit_loop, write_and_run_with_loop
    RECURSIVE_EXECUTION_MODE = True
except ImportError:
    RECURSIVE_EXECUTION_MODE = False

# Assuming global memory_log, current_memory_mode, current_personality, etc.
# These should be passed as parameters in a more robust setup

AVAILABLE_COMMANDS = [
    "quit", "exit", "help", "list_facts", "delete_fact", "show_contradictions", 
    "resolve_contradiction", "resolve_contradiction_interactive", "list_facts_decayed", "reinforce_fact", "summarize_all",
    "summarize_subject", "show_summaries", "memory_mode", "set_memory_mode",
    "list_episodes", "show_episode", "delete_episode", "prune_memory", "forget_subject",
    "personality", "set_personality", "personality_status", "adjust_personality", "evolve_personality", "personality_summary", "summarize_contradictions", "highlight_conflicts",
    "personality_evolve", "personality_history",
    "contradiction_clusters", "evolve_file_with_context", "generate_meta_goals",
    "health_check", "test_patterns", "show_emotive_facts", "reconcile",
    "embedding_cache_stats", "clear_embedding_cache", "test_agreement", "test_vectorizer",
    "execute_meta_goals", "dump_memory", "sanitize_facts",
    "sentiment_trajectory", "opinion_summary", "volatility_report", "belief_context",
    "resolve_from_trajectory", "apply_decay", "trajectory_reflect", "list_profiles", "merge_profiles",
    "beliefs", "reflex_templates", "memory_clean_log", "belief_trace",
    "hypotheses", "confirm_hypothesis", "reject_hypothesis", "causal_dashboard",
    "self_model", "simulate_repair", "compare_strategies", "route_subgoals",
    "self_drives", "intent_model", "goal_pressure", "drive_spawn",
    "plan_goal", "show_plan", "execute_plan", "why_am_i_doing_this", "self_prompt",
    "list_plans", "delete_plan", "plan_stats", "intention_stats",
    "self_diagnose", "self_repair", "show_flaws", "repair_log",
    "write_and_run", "recursive_run", "list_generated_files", "show_file", "delete_file",
    "world_state", "predict_event", "belief",
    "list_constraints", "evaluate_action", "add_constraint", "remove_constraint",
    "replicate_genome", "score_genome", "lineage_tree", "list_genomes", "activate_genome",
    "debate", "critique", "simulate_debate", "reflect_on",
    "consolidate_memory", "list_clusters", "prioritize_memory", "memory_stats",
    "fork_agent", "run_fork", "score_forks", "prune_forks", "mutate_agent", "fork_status", "replication_cycle", "tune_replication",
    "dissonance_report", "resolve_dissonance", "dissonance_history",
    "promote_agent", "retire_agent", "lifecycle_status", "evaluate_lifecycle", "list_agents_with_drift",
    "self_summary", "self_journal", "self_reflect", "self_sync",
    "contract", "score_task", "update_contract", "list_contracts",
    "meta_status", "meta_reflect", "meta_goals",
    "plan_status", "plan_next", "plan_eval",
    "reflex_mode", "reflex_status",
    "mesh_status", "mesh_sync", "mesh_agents", "mesh_enable", "mesh_disable",
    "arbiter_trace"
]

def handle_command(user_input: str, memory_log: MemoryLog, current_memory_mode: str, current_personality: str) -> str:
    """Dispatcher for commands. Returns 'continue' if handled, 'exit' if quit, 'not_handled' otherwise."""
    lower_input = user_input.lower()
    parts = user_input.split()
    cmd = parts[0].lower() if parts else ''
    
    if cmd in ['quit', 'exit']:
        print("👋 Goodbye!")
        return 'exit'
    
    elif cmd == 'list_profiles':
        # List all user_profile_id with fact counts
        from storage.db_utils import get_conn
        conn = get_conn(memory_log.db_path)
        rows = conn.execute("SELECT user_profile_id, COUNT(*) FROM facts WHERE user_profile_id IS NOT NULL GROUP BY user_profile_id").fetchall()
        print("\n👤 User Profiles:")
        print("-" * 40)
        for row in rows:
            print(f"{row[0]}: {row[1]} facts")
        return 'continue'
    elif cmd == 'merge_profiles' and len(parts) == 3:
        profile1, profile2 = parts[1], parts[2]
        from storage.db_utils import get_conn
        conn = get_conn(memory_log.db_path)
        # Update all facts from profile2 to profile1
        conn.execute("UPDATE facts SET user_profile_id = ? WHERE user_profile_id = ?", (profile1, profile2))
        conn.commit()
        print(f"✅ Merged facts from {profile2} into {profile1}")
        return 'continue'
    
    elif cmd == 'list_facts':
        # Use enhanced system if available
        if ENHANCED_MODE and hasattr(memory_log, 'enhanced_memory'):
            from config.settings import get_user_profile_id
            user_profile_id = get_user_profile_id()
            session_id = memory_log.current_session_id if hasattr(memory_log, 'current_session_id') else None
            result = memory_log.enhanced_memory._handle_command("/list_facts", user_profile_id, session_id or "default")
            print(result["response"])
            return 'continue'
        
        # Legacy code
        print("\n📚 All Facts (with IDs):")
        print("-" * 60)
        facts_with_ids = memory_log.list_facts_with_ids()
        if not facts_with_ids:
            print("No facts found.")
            return 'continue'
        
        for fact_id, fact in facts_with_ids:
            contra_icon = "⚠️" if getattr(fact, 'contradiction_score', 0) > 0.7 else ""
            vol_icon = "🔥" if getattr(fact, 'volatility_score', 0) > 0.5 else ""
            icons = f"{contra_icon}{vol_icon}".strip()
            ts = getattr(fact, 'timestamp', '')
            if isinstance(ts, str) and ts:
                ts_disp = ts.split()[0]
            else:
                ts_disp = str(ts)
            print(f"{fact_id:3d}. {icons} {fact.subject} {fact.predicate} {fact.object} (seen {fact.frequency}×, last: {ts_disp})")
        return 'continue'
    
    # Add similar handlers for all other commands...
    # For brevity, I'll add a few; in full, copy all elif blocks here as functions
    
    elif cmd == 'delete_fact':
        try:
            fact_id = int(parts[1])
            if memory_log.delete_fact(fact_id):
                print(f"✅ Deleted fact {fact_id}")
            else:
                print(f"❌ Fact {fact_id} not found")
            return 'continue'
        except (ValueError, IndexError):
            print("❌ Usage: delete_fact <ID>")
            return 'continue'
    
    # ... Add all other handlers similarly ...
    
    # For example, handle_show_contradictions:
    elif cmd == 'show_contradictions':
        # Use enhanced system if available
        if ENHANCED_MODE and hasattr(memory_log, 'enhanced_memory'):
            from config.settings import get_user_profile_id
            user_profile_id = get_user_profile_id()
            session_id = memory_log.current_session_id if hasattr(memory_log, 'current_session_id') else None
            result = memory_log.enhanced_memory._handle_command("/show_contradictions", user_profile_id, session_id or "default")
            print(result["response"])
            return 'continue'
        
        # Legacy code
        print("\n⚠️ Contradiction History:")
        print("-" * 60)
        contradictions = memory_log.get_contradictions()
        if not contradictions:
            print("No contradictions logged.")
            return 'continue'
        for contra in contradictions:
            status = "✅ RESOLVED" if contra['resolved'] else "❌ UNRESOLVED"
            print(f"{contra['id']:3d}. {status} (confidence: {contra['confidence']:.2f})")
            print(f"     A: {contra['fact_a_text']}")
            print(f"     B: {contra['fact_b_text']}")
            print(f"     Time: {contra['timestamp']}")
            if contra['resolved'] and contra['resolution_notes']:
                print(f"     Resolution: {contra['resolution_notes']}")
            print()
        return 'continue'
    
    elif cmd == 'summarize' or cmd == 'summarize_all':
        # Use enhanced system if available
        if ENHANCED_MODE and hasattr(memory_log, 'enhanced_memory'):
            from config.settings import get_user_profile_id
            user_profile_id = get_user_profile_id()
            session_id = memory_log.current_session_id if hasattr(memory_log, 'current_session_id') else None
            result = memory_log.enhanced_memory._handle_command("/summarize", user_profile_id, session_id or "default")
            print(result["response"])
            return 'continue'
        
        # Legacy summarization would go here
        print("Summarization not available in legacy mode")
        return 'continue'
    
    elif cmd == 'generate_meta_goals':
        # Use enhanced system if available
        if ENHANCED_MODE and hasattr(memory_log, 'enhanced_memory'):
            from config.settings import get_user_profile_id
            user_profile_id = get_user_profile_id()
            session_id = memory_log.current_session_id if hasattr(memory_log, 'current_session_id') else None
            result = memory_log.enhanced_memory._handle_command("/generate_meta_goals", user_profile_id, session_id or "default")
            print(result["response"])
            return 'continue'
        
        # Legacy would use execute_meta_goals
        return 'continue'
    
    # New enhanced CLI commands for Belief Abstraction + Reflex Compression + Memory Autoclean
    
    elif cmd == 'beliefs':
        """Show current abstracted beliefs per cluster"""
        try:
            from storage.enhanced_memory import BeliefAbstractionLayer
            
            belief_system = BeliefAbstractionLayer()
            beliefs = belief_system.get_all_beliefs(limit=20)
            stats = belief_system.get_belief_statistics()
            
            print("\n🧠 Abstract Beliefs:")
            print("-" * 80)
            print(f"Total beliefs: {stats['total_beliefs']}")
            print(f"Average confidence: {stats['average_confidence']:.2f}")
            print(f"Average coherence: {stats['average_coherence']:.2f}")
            print()
            
            if beliefs:
                for belief in beliefs:
                    confidence_icon = "🟢" if belief.confidence >= 0.8 else "🟡" if belief.confidence >= 0.6 else "🔴"
                    print(f"{confidence_icon} {belief.belief_id}")
                    print(f"   Cluster: {belief.cluster_id}")
                    print(f"   Statement: {belief.abstract_statement}")
                    print(f"   Confidence: {belief.confidence:.2f} | Coherence: {belief.coherence_score:.2f}")
                    print(f"   Usage: {belief.usage_count} | Facts: {len(belief.supporting_facts)}")
                    print()
            else:
                print("No abstract beliefs found. Beliefs are created automatically from consistent fact clusters.")
            
            return 'continue'
        except ImportError:
            print("❌ Belief abstraction system not available")
            return 'continue'
    
    elif cmd == 'reflex_templates':
        """View reusable repair patterns"""
        try:
            from storage.reflex_compression import ReflexCompressor
            
            compressor = ReflexCompressor()
            templates = compressor.get_all_templates(limit=10)
            stats = compressor.get_template_statistics()
            
            print("\n🔄 Reflex Templates:")
            print("-" * 80)
            print(f"Total templates: {stats['total_templates']}")
            print(f"Average success rate: {stats['average_success_rate']:.2f}")
            print(f"Average score: {stats['average_score']:.2f}")
            print()
            
            if templates:
                for template in templates:
                    success_icon = "🟢" if template.success_rate >= 0.8 else "🟡" if template.success_rate >= 0.6 else "🔴"
                    print(f"{success_icon} {template.template_id}")
                    print(f"   Strategy: {template.strategy}")
                    print(f"   Goal Pattern: {template.goal_pattern}")
                    print(f"   Success Rate: {template.success_rate:.2f} | Avg Score: {template.avg_score:.2f}")
                    print(f"   Usage: {template.usage_count} | Source Cycles: {len(template.source_cycles)}")
                    print(f"   Execution: {', '.join(template.execution_pattern[:3])}...")
                    print()
            else:
                print("No reflex templates found. Templates are created automatically from similar reflex cycles.")
            
            return 'continue'
        except ImportError:
            print("❌ Reflex compression system not available")
            return 'continue'
    
    elif cmd == 'memory_clean_log':
        """View memory autocleaner results"""
        try:
            from storage.memory_autocleaner import MemoryCleaner
            
            cleaner = MemoryCleaner()
            cleanup_log = cleaner.get_cleanup_log(limit=20)
            stats = cleaner.get_cleanup_statistics()
            memory_stats = cleaner.get_memory_usage_stats()
            
            print("\n🧹 Memory Cleanup Log:")
            print("-" * 80)
            print(f"Total cleanup actions: {stats['total_actions']}")
            print(f"Total memory saved: {stats['total_memory_saved']} bytes")
            print(f"Total confidence impact: {stats['total_confidence_impact']:.2f}")
            print(f"Recent actions (24h): {stats['recent_actions_24h']}")
            print()
            
            print("Current Memory Usage:")
            print(f"  Total facts: {memory_stats['total_facts']}")
            print(f"  Contradictions: {memory_stats['total_contradictions']}")
            print(f"  Avg confidence: {memory_stats['average_confidence']:.2f}")
            print(f"  Avg volatility: {memory_stats['average_volatility']:.2f}")
            print()
            
            if cleanup_log:
                print("Recent Cleanup Actions:")
                for action in cleanup_log[:10]:
                    action_icon = "🗑️" if action.action_type == "remove" else "🗜️" if action.action_type == "compress" else "⚡"
                    print(f"{action_icon} {action.action_type.upper()} {action.target_type}: {action.target_id}")
                    print(f"   Reason: {action.reason}")
                    print(f"   Memory saved: {action.memory_saved} bytes | Impact: {action.confidence_impact:.2f}")
                    print()
            else:
                print("No cleanup actions logged yet.")
            
            return 'continue'
        except ImportError:
            print("❌ Memory autocleaner system not available")
            return 'continue'
    
    elif cmd == 'belief_trace':
        """Show beliefs that informed a decision"""
        try:
            from storage.enhanced_reasoning import EnhancedReasoningEngine
            
            reasoning_engine = EnhancedReasoningEngine()
            
            if len(parts) > 1:
                # Show trace for specific goal or token
                target = parts[1]
                if target.isdigit():
                    # Token ID
                    token_id = int(target)
                    traces = reasoning_engine.get_belief_traces_by_token(token_id, limit=5)
                    print(f"\n🧠 Belief Traces for Token {token_id}:")
                else:
                    # Goal ID
                    trace = reasoning_engine.get_belief_trace(target)
                    traces = [trace] if trace else []
                    print(f"\n🧠 Belief Trace for Goal {target}:")
            else:
                # Show recent traces
                traces = reasoning_engine.get_recent_belief_traces(limit=10)
                print("\n🧠 Recent Belief Traces:")
            
            print("-" * 80)
            
            if traces:
                for trace in traces:
                    confidence_icon = "🟢" if trace.strategy_confidence >= 0.8 else "🟡" if trace.strategy_confidence >= 0.6 else "🔴"
                    print(f"{confidence_icon} {trace.trace_id}")
                    print(f"   Goal: {trace.goal_id}")
                    print(f"   Strategy: {trace.final_strategy} (confidence: {trace.strategy_confidence:.2f})")
                    print(f"   Beliefs considered: {len(trace.considered_beliefs)}")
                    if trace.belief_influences:
                        top_influences = sorted(trace.belief_influences.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top influences: {', '.join([f'{bid}({inf:.2f})' for bid, inf in top_influences])}")
                    print(f"   Reasoning: {trace.reasoning_notes}")
                    print()
            else:
                print("No belief traces found.")
            
            return 'continue'
        except ImportError:
            print("❌ Enhanced reasoning system not available")
            return 'continue'
    
    elif cmd == 'hypotheses':
        """Show open hypotheses"""
        try:
            from agents.hypothesis_generator import HypothesisGeneratorAgent
            
            hypothesis_agent = HypothesisGeneratorAgent()
            open_hypotheses = hypothesis_agent.get_open_hypotheses(limit=10)
            stats = hypothesis_agent.get_hypothesis_statistics()
            
            print("\n🔬 Open Hypotheses:")
            print("-" * 80)
            print(f"Total hypotheses: {stats['total_hypotheses']}")
            print(f"Confirmation rate: {stats['confirmation_rate']:.2%}")
            print(f"Recent hypotheses (24h): {stats['recent_hypotheses_24h']}")
            print()
            
            if open_hypotheses:
                for hypothesis in open_hypotheses:
                    confidence_icon = "🟢" if hypothesis.confidence_score >= 0.8 else "🟡" if hypothesis.confidence_score >= 0.6 else "🔴"
                    print(f"{confidence_icon} {hypothesis.hypothesis_id}")
                    print(f"   Cause: {hypothesis.cause_token}")
                    print(f"   Prediction: {hypothesis.predicted_outcome}")
                    print(f"   Probability: {hypothesis.probability:.2f} | Confidence: {hypothesis.confidence_score:.2f}")
                    print(f"   Type: {hypothesis.hypothesis_type}")
                    print(f"   Evidence: {', '.join(hypothesis.supporting_evidence)}")
                    print(f"   Created: {datetime.fromtimestamp(hypothesis.created_at)}")
                    print()
            else:
                print("No open hypotheses found.")
            
            return 'continue'
        except ImportError:
            print("❌ Hypothesis generator system not available")
            return 'continue'
    
    elif cmd.startswith('confirm_hypothesis'):
        """Confirm a hypothesis"""
        try:
            from agents.hypothesis_generator import HypothesisGeneratorAgent
            
            if len(parts) < 2:
                print("❌ Usage: confirm_hypothesis <hypothesis_id>")
                return 'continue'
            
            hypothesis_id = parts[1]
            hypothesis_agent = HypothesisGeneratorAgent()
            hypothesis_agent.confirm_hypothesis(hypothesis_id)
            
            print(f"✅ Hypothesis {hypothesis_id} confirmed")
            return 'continue'
        except ImportError:
            print("❌ Hypothesis generator system not available")
            return 'continue'
    
    elif cmd.startswith('reject_hypothesis'):
        """Reject a hypothesis"""
        try:
            from agents.hypothesis_generator import HypothesisGeneratorAgent
            
            if len(parts) < 2:
                print("❌ Usage: reject_hypothesis <hypothesis_id>")
                return 'continue'
            
            hypothesis_id = parts[1]
            hypothesis_agent = HypothesisGeneratorAgent()
            hypothesis_agent.reject_hypothesis(hypothesis_id)
            
            print(f"❌ Hypothesis {hypothesis_id} rejected")
            return 'continue'
        except ImportError:
            print("❌ Hypothesis generator system not available")
            return 'continue'
    
    elif cmd == 'causal_dashboard':
        """Show causal audit dashboard"""
        try:
            from storage.causal_audit_dashboard import CausalAuditDashboard
            from datetime import datetime
            
            dashboard = CausalAuditDashboard()
            dashboard_data = dashboard.generate_dashboard_data()
            
            print("\n" + "=" * 80)
            print("🔮 CAUSAL AUDIT DASHBOARD")
            print("=" * 80)
            print(f"Generated: {datetime.fromtimestamp(dashboard_data['timestamp'])}")
            print()
            
            # Current metrics
            metrics = dashboard_data['current_metrics']
            print("📊 CURRENT METRICS:")
            print(f"  Total Predictions: {metrics.total_predictions}")
            print(f"  Pending Predictions: {metrics.pending_predictions}")
            print(f"  Total Hypotheses: {metrics.total_hypotheses}")
            print(f"  Open Hypotheses: {metrics.open_hypotheses}")
            print(f"  Anticipatory Reflexes: {metrics.total_anticipatory_reflexes}")
            print(f"  Successful Reflexes: {metrics.successful_reflexes}")
            print(f"  Prediction Accuracy: {metrics.prediction_accuracy:.2%}")
            print(f"  Hypothesis Confirmation Rate: {metrics.hypothesis_confirmation_rate:.2%}")
            print(f"  Reflex Success Rate: {metrics.reflex_success_rate:.2%}")
            print()
            
            # Upcoming drifts
            print("🔮 UPCOMING PREDICTED DRIFTS:")
            for drift in dashboard_data['upcoming_drifts'][:5]:
                urgency_icon = "🔴" if drift['urgency'] > 0.8 else "🟡" if drift['urgency'] > 0.5 else "🟢"
                print(f"{urgency_icon} {drift['prediction_type']}: {drift['predicted_outcome']}")
                print(f"    Probability: {drift['probability']:.2%} | Urgency: {drift['urgency']:.2f}")
            print()
            
            # System health
            health = dashboard_data['system_statistics']['system_health']
            health_icon = "🟢" if health['status'] == 'excellent' else "🟡" if health['status'] == 'good' else "🔴"
            print(f"{health_icon} SYSTEM HEALTH: {health['status'].upper()} ({health['overall_score']:.2%})")
            print()
            
            return 'continue'
        except ImportError:
            print("❌ Causal audit dashboard not available")
            return 'continue'
    
    # Continue adding all handlers from the elif chain
    
    elif cmd == 'self_model':
        """Show symbolic self-representation"""
        try:
            from storage.self_model import CognitiveSelfModel
            
            self_model = CognitiveSelfModel()
            representation = self_model.get_symbolic_self_representation()
            
            print("\n" + "=" * 80)
            print("🧠 SYMBOLIC SELF-REPRESENTATION")
            print("=" * 80)
            
            # Core beliefs
            print("\n💭 CORE BELIEFS:")
            for belief, confidence in representation.get('core_beliefs', {}).items():
                confidence_icon = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                print(f"{confidence_icon} {belief}: {confidence:.2f}")
            
            # Active strategies
            print("\n🎯 ACTIVE STRATEGIES:")
            for strategy, effectiveness in representation.get('active_strategies', {}).items():
                effectiveness_icon = "🟢" if effectiveness > 0.8 else "🟡" if effectiveness > 0.5 else "🔴"
                print(f"{effectiveness_icon} {strategy}: {effectiveness:.2f}")
            
            # Symbolic rules
            print("\n🔗 SYMBOLIC RULES:")
            for rule in representation.get('symbolic_rules', [])[:10]:  # Show top 10
                confidence_icon = "🟢" if rule['confidence'] > 0.8 else "🟡" if rule['confidence'] > 0.5 else "🔴"
                print(f"{confidence_icon} {rule['antecedent']} → {rule['consequent']} (conf: {rule['confidence']:.2f})")
            
            # Cognitive patterns
            print("\n🧩 COGNITIVE PATTERNS:")
            patterns = representation.get('cognitive_patterns', {})
            for pattern, description in patterns.items():
                print(f"• {pattern}: {description}")
            
            print()
            return 'continue'
        except ImportError:
            print("❌ Self-model system not available")
            return 'continue'
    
    elif cmd.startswith('simulate_repair'):
        """Simulate repair options for a goal"""
        try:
            from agents.repair_simulator import get_repair_simulator
            import uuid
            
            # Parse goal_id from command
            goal_id = None
            if len(parts) > 1:
                goal_id = parts[1]
            else:
                goal_id = f"sim_goal_{uuid.uuid4().hex[:8]}"
            
            # Create sample current state
            current_state = {
                'drift_type': 'contradiction',
                'drift_score': 0.6,
                'cluster_id': 'sample_cluster',
                'volatility_score': 0.4,
                'contradiction_count': 3,
                'coherence_score': 0.7,
                'token_id': 123
            }
            
            repair_simulator = get_repair_simulator()
            simulation_result = repair_simulator.simulate_repair_paths(goal_id, current_state)
            
            print("\n" + "=" * 80)
            print(f"🔮 REPAIR SIMULATION FOR GOAL: {goal_id}")
            print("=" * 80)
            
            print(f"\n📊 SIMULATION RESULTS:")
            print(f"Total strategies simulated: {len(simulation_result.simulated_repairs)}")
            
            if simulation_result.best_repair:
                print(f"\n🏆 BEST STRATEGY: {simulation_result.best_repair.strategy}")
                print(f"Predicted Score: {simulation_result.best_repair.predicted_score:.3f}")
                print(f"Confidence: {simulation_result.best_repair.confidence:.3f}")
                print(f"Success Probability: {simulation_result.best_repair.success_probability:.3f}")
                print(f"Estimated Duration: {simulation_result.best_repair.estimated_duration:.1f}s")
                
                if simulation_result.best_repair.risk_factors:
                    print(f"\n⚠️ RISK FACTORS:")
                    for risk in simulation_result.best_repair.risk_factors:
                        print(f"  • {risk}")
            
            print(f"\n📋 ALL STRATEGIES:")
            for i, repair in enumerate(simulation_result.simulated_repairs[:5], 1):
                score_icon = "🟢" if repair.predicted_score > 0.7 else "🟡" if repair.predicted_score > 0.5 else "🔴"
                print(f"{i}. {score_icon} {repair.strategy}")
                print(f"   Score: {repair.predicted_score:.3f} | Confidence: {repair.confidence:.3f}")
                print(f"   Duration: {repair.estimated_duration:.1f}s | Success: {repair.success_probability:.3f}")
            
            print(f"\n💭 REASONING SUMMARY:")
            print(simulation_result.reasoning_summary)
            
            print()
            return 'continue'
        except ImportError:
            print("❌ Repair simulator not available")
            return 'continue'
    
    elif cmd.startswith('compare_strategies'):
        """Compare two strategies using symbolic logic"""
        try:
            from storage.reflex_log import compare_strategies_logically
            
            if len(parts) < 3:
                print("❌ Usage: compare_strategies <strategy1> <strategy2>")
                return 'continue'
            
            strategy1 = parts[1]
            strategy2 = parts[2]
            
            # Optional context
            context = {}
            if len(parts) > 3:
                context['drift_type'] = parts[3]
            
            comparison = compare_strategies_logically(strategy1, strategy2, context)
            
            print("\n" + "=" * 80)
            print(f"⚖️ STRATEGY COMPARISON: {strategy1} vs {strategy2}")
            print("=" * 80)
            
            if 'error' in comparison:
                print(f"❌ Error: {comparison['error']}")
                return 'continue'
            
            # Metrics comparison
            print(f"\n📊 METRICS COMPARISON:")
            metrics1 = comparison['metrics1']
            metrics2 = comparison['metrics2']
            
            print(f"{strategy1}:")
            print(f"  Avg Score: {metrics1['avg_score']:.3f}")
            print(f"  Success Rate: {metrics1['success_rate']:.3f}")
            print(f"  Sample Count: {metrics1['sample_count']}")
            
            print(f"\n{strategy2}:")
            print(f"  Avg Score: {metrics2['avg_score']:.3f}")
            print(f"  Success Rate: {metrics2['success_rate']:.3f}")
            print(f"  Sample Count: {metrics2['sample_count']}")
            
            # Symbolic rules
            print(f"\n🔗 SYMBOLIC RULES:")
            for rule in comparison['symbolic_rules']:
                print(f"  • {rule}")
            
            # Conclusion
            print(f"\n🏆 CONCLUSION:")
            print(f"{comparison['conclusion']}")
            if comparison['recommended_strategy']:
                print(f"Recommended: {comparison['recommended_strategy']}")
            
            print()
            return 'continue'
        except ImportError:
            print("❌ Strategy comparison system not available")
            return 'continue'
    
    elif cmd.startswith('route_subgoals'):
        """Show routed plan for subgoals"""
        try:
            from agents.meta_router import get_meta_router
            import uuid
            
            # Parse goal_id from command
            goal_id = None
            if len(parts) > 1:
                goal_id = parts[1]
            else:
                goal_id = f"route_goal_{uuid.uuid4().hex[:8]}"
            
            # Create sample current state
            current_state = {
                'drift_type': 'volatility',
                'drift_score': 0.5,
                'cluster_id': 'sample_cluster',
                'volatility_score': 0.6,
                'contradiction_count': 2,
                'coherence_score': 0.8,
                'token_id': 456
            }
            
            meta_router = get_meta_router()
            routing_plan = meta_router.route_subgoals(
                goal_id, current_state, 
                goal_description="Sample goal for demonstration",
                tags=["demo", "volatility_repair"]
            )
            
            print("\n" + "=" * 80)
            print(f"🛣️ SUBGOAL ROUTING PLAN FOR GOAL: {goal_id}")
            print("=" * 80)
            
            print(f"\n📊 ROUTING STATISTICS:")
            print(f"Total subgoals: {len(routing_plan.subgoals)}")
            print(f"Total estimated duration: {routing_plan.total_estimated_duration:.1f}s")
            print(f"Routing confidence: {routing_plan.routing_confidence:.3f}")
            
            if routing_plan.subgoals:
                print(f"\n🎯 ROUTED SUBGOALS:")
                for i, subgoal in enumerate(routing_plan.subgoals, 1):
                    confidence_icon = "🟢" if subgoal.confidence > 0.7 else "🟡" if subgoal.confidence > 0.5 else "🔴"
                    print(f"{i}. {confidence_icon} {subgoal.subgoal_id}")
                    print(f"   Agent: {subgoal.agent_type}")
                    print(f"   Priority: {subgoal.priority:.3f} | Confidence: {subgoal.confidence:.3f}")
                    print(f"   Duration: {subgoal.estimated_duration:.1f}s")
                    if subgoal.dependencies:
                        print(f"   Dependencies: {', '.join(subgoal.dependencies)}")
                    if subgoal.tags:
                        print(f"   Tags: {', '.join(subgoal.tags)}")
                    print()
            
            print(f"💭 ROUTING REASONING:")
            print(routing_plan.reasoning_summary)
            
            print()
            return 'continue'
        except ImportError:
            print("❌ Meta router system not available")
            return 'continue'
    
    # Phase 7: Motivational Drives & Intent Engine Commands
    elif cmd == 'self_drives':
        if not PHASE7_MODE:
            print("❌ Phase 7 drive system not available")
            return 'continue'
        
        try:
            drive_system = get_drive_system(memory_log.db_path)
            self_model = get_self_aware_model(memory_log.db_path)
            
            print("\n🧠 **Current Motivational Drive Signals**")
            print("=" * 50)
            
            # Get current drives
            dominant_drives = drive_system.get_current_dominant_drives()
            active_drives = self_model.get_dominant_motives()
            
            if not dominant_drives:
                print("No active drive signals detected.")
                return 'continue'
            
            # Sort drives by strength
            sorted_drives = sorted(dominant_drives.items(), key=lambda x: x[1], reverse=True)
            
            for drive, strength in sorted_drives:
                intensity = "🔥" if strength > 0.8 else "🟡" if strength > 0.5 else "🔵"
                print(f"{intensity} **{drive.title()}**: {strength:.3f}")
            
            # Show drive trends
            trends = drive_system.analyze_drive_trends()
            if not trends.get("error"):
                print(f"\n📊 **Analysis** ({trends.get('samples_analyzed', 0)} samples):")
                for drive, trend_data in trends.get("drive_trends", {}).items():
                    direction = trend_data.get("trend", "stable")
                    volatility = trend_data.get("volatility", 0.0)
                    print(f"  • {drive}: {direction} (volatility: {volatility:.2f})")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error accessing drive system: {e}")
            return 'continue'
    
    elif cmd == 'intent_model':
        if not PHASE7_MODE:
            print("❌ Phase 7 intent modeling not available")
            return 'continue'
        
        try:
            intent_modeler = get_intent_modeler(memory_log, get_drive_system(memory_log.db_path))
            
            print("\n🎯 **Current Intent Model & Meta-Goals**")
            print("=" * 50)
            
            # Get current motivational summary
            summary = intent_modeler.summarize_current_motives()
            print(summary)
            
            # Show recent evolved goals
            recent_goals = [
                goal for goal in intent_modeler.evolved_goals 
                if (time.time() - goal.created_at) < 86400  # Last 24 hours
            ]
            
            if recent_goals:
                print(f"\n⚡ **Recent Evolved Goals** ({len(recent_goals)} goals):")
                for goal in recent_goals[:5]:  # Show top 5
                    age_hours = (time.time() - goal.created_at) / 3600
                    print(f"  • {goal.description}")
                    print(f"    Priority: {goal.adaptive_priority:.2f}, Age: {age_hours:.1f}h")
                    if hasattr(goal, 'driving_motives'):
                        top_motive = max(goal.driving_motives.items(), key=lambda x: x[1])
                        print(f"    Driven by: {top_motive[0]} ({top_motive[1]:.2f})")
            else:
                print("\n⚡ **Recent Evolved Goals**: None in the last 24 hours")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error accessing intent modeler: {e}")
            return 'continue'
    
    elif cmd == 'goal_pressure':
        if not PHASE7_MODE:
            print("❌ Phase 7 drive system not available")
            return 'continue'
        
        try:
            drive_system = get_drive_system(memory_log.db_path)
            
            print("\n⚡ **Token Drive Pressure Rankings**")
            print("=" * 50)
            
            # Get tokens ranked by drive pressure
            pressured_tokens = drive_system.rank_tokens_by_drive_pressure(15)
            
            if not pressured_tokens:
                print("No tokens with significant drive pressure detected.")
                return 'continue'
            
            for i, (token_id, pressure) in enumerate(pressured_tokens, 1):
                # Get token details
                drives = drive_system.evaluate_token_state(token_id)
                top_drive = max(drives.items(), key=lambda x: x[1]) if drives else ("unknown", 0.0)
                
                urgency = "🚨" if pressure > 0.8 else "⚠️" if pressure > 0.6 else "📊"
                print(f"{urgency} **#{i} Token {token_id}**: Pressure {pressure:.3f}")
                print(f"    Dominant drive: {top_drive[0]} ({top_drive[1]:.2f})")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error accessing drive pressure data: {e}")
            return 'continue'
    
    elif cmd == 'drive_spawn':
        if not PHASE7_MODE:
            print("❌ Phase 7 drive system not available")
            return 'continue'
        
        try:
            drive_system = get_drive_system(memory_log.db_path)
            
            print("\n🌱 **Manual Drive-Based Goal Generation**")
            print("=" * 50)
            
            # Get high-pressure tokens
            pressured_tokens = drive_system.rank_tokens_by_drive_pressure(10)
            
            if not pressured_tokens:
                print("No tokens with sufficient drive pressure for goal generation.")
                return 'continue'
            
            spawned_count = 0
            for token_id, pressure in pressured_tokens[:5]:  # Try top 5 tokens
                goal = drive_system.spawn_goal_if_needed(token_id, force=True)
                if goal:
                    print(f"✅ Spawned goal for token {token_id}:")
                    print(f"   Goal: {goal.description}")
                    print(f"   Strategy: {goal.strategy}")
                    print(f"   Priority: {goal.priority:.2f}")
                    if hasattr(goal, 'driving_motives'):
                        top_motive = max(goal.driving_motives.items(), key=lambda x: x[1])
                        print(f"   Driven by: {top_motive[0]} ({top_motive[1]:.2f})")
                    spawned_count += 1
                    print()
            
            if spawned_count == 0:
                print("⚠️ No goals could be spawned at this time.")
            else:
                print(f"🎯 Successfully spawned {spawned_count} autonomous goals.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error spawning drive goals: {e}")
            return 'continue'
    
    elif cmd == 'mood_state':
        """Show current emotional/mood state"""
        try:
            from storage.emotion_model import get_emotion_model
            
            emotion_model = get_emotion_model(memory_log.db_path)
            current_mood = emotion_model.get_current_mood()
            
            print("\n🎭 Current Emotional State:")
            print("-" * 50)
            print(f"Mood: {current_mood.get('mood_label', 'neutral').title()}")
            print(f"Valence: {current_mood.get('valence', 0.0):.2f} (-1.0 to 1.0)")
            print(f"Arousal: {current_mood.get('arousal', 0.3):.2f} (0.0 to 1.0)")
            print(f"Confidence: {current_mood.get('confidence', 0.0):.2f}")
            
            duration = current_mood.get('duration', 0.0)
            if duration > 0:
                if duration < 60:
                    duration_str = f"{duration:.0f}s"
                elif duration < 3600:
                    duration_str = f"{duration/60:.0f}m"
                else:
                    duration_str = f"{duration/3600:.1f}h"
                print(f"Duration: {duration_str}")
            
            contributing_events = current_mood.get('contributing_events', {})
            if contributing_events:
                print(f"\nContributing Events:")
                for event, count in contributing_events.items():
                    print(f"  • {event}: {count}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error retrieving mood state: {e}")
            return 'continue'
    
    elif cmd == 'identity_signature':
        """Show current identity signature and traits"""
        try:
            from storage.self_model import get_self_aware_model
            
            self_model = get_self_aware_model(memory_log.db_path)
            identity_sig = self_model.get_identity_signature()
            
            print("\n🎭 Identity Signature:")
            print("-" * 50)
            print(f"Current Identity: {identity_sig}")
            
            # Show individual traits
            if self_model.identity_traits:
                print(f"\nDetailed Traits:")
                for trait_name, trait in self_model.identity_traits.items():
                    if trait.confidence >= 0.3:  # Only show reasonably confident traits
                        strength_desc = ""
                        if trait.strength > 0.8:
                            strength_desc = "Very Strong"
                        elif trait.strength > 0.6:
                            strength_desc = "Strong"
                        elif trait.strength > 0.4:
                            strength_desc = "Moderate"
                        else:
                            strength_desc = "Weak"
                        
                        print(f"  • {trait_name.title()}: {strength_desc} "
                              f"(strength: {trait.strength:.2f}, confidence: {trait.confidence:.2f})")
                        print(f"    Evidence count: {trait.evidence_count}, "
                              f"Last updated: {int((time.time() - trait.last_updated) / 3600)}h ago")
            else:
                print("No established identity traits yet.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error retrieving identity signature: {e}")
            return 'continue'
    
    elif cmd == 'emotion_log':
        """Show emotional history for a token or globally"""
        try:
            from storage.emotion_model import get_emotion_model
            
            emotion_model = get_emotion_model(memory_log.db_path)
            
            token_id = None
            hours_back = 24.0
            
            # Parse arguments: emotion_log [token_id] [hours]
            if len(parts) > 1:
                try:
                    token_id = int(parts[1])
                except ValueError:
                    try:
                        hours_back = float(parts[1])
                    except ValueError:
                        print("❌ Invalid token_id or hours argument")
                        return 'continue'
            
            if len(parts) > 2:
                try:
                    hours_back = float(parts[2])
                except ValueError:
                    print("❌ Invalid hours argument")
                    return 'continue'
            
            emotional_history = emotion_model.get_emotional_history(token_id, hours_back)
            
            print(f"\n😊 Emotional History (last {hours_back}h):")
            if token_id:
                print(f"Token ID: {token_id}")
            print("-" * 60)
            
            if not emotional_history:
                print("No emotional data found.")
                return 'continue'
            
            for record in emotional_history[-20:]:  # Show last 20 records
                timestamp = record.get('timestamp', 0)
                time_ago = (time.time() - timestamp) / 3600  # Hours ago
                valence = record.get('valence', 0.0)
                arousal = record.get('arousal', 0.0)
                event_type = record.get('event_type', 'unknown')
                token_ref = record.get('token_id', 'global')
                
                valence_emoji = "😊" if valence > 0.3 else "😐" if valence > -0.3 else "😞"
                arousal_emoji = "🔥" if arousal > 0.7 else "⚡" if arousal > 0.4 else "😴"
                
                print(f"{valence_emoji}{arousal_emoji} {time_ago:.1f}h ago: {event_type} "
                      f"(v:{valence:+.2f}, a:{arousal:.2f}) [{token_ref}]")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error retrieving emotion log: {e}")
            return 'continue'
    
    elif cmd == 'set_emotion':
        """Manually set emotion state for debugging"""
        try:
            if len(parts) < 3:
                print("Usage: set_emotion <valence> <arousal> [duration_seconds]")
                print("  valence: -1.0 to 1.0 (negative to positive)")
                print("  arousal: 0.0 to 1.0 (calm to intense)")
                print("  duration: seconds to maintain override (default: 300)")
                return 'continue'
            
            try:
                valence = float(parts[1])
                arousal = float(parts[2])
                duration = float(parts[3]) if len(parts) > 3 else 300.0
            except ValueError:
                print("❌ Invalid numeric values for valence/arousal/duration")
                return 'continue'
            
            if not (-1.0 <= valence <= 1.0):
                print("❌ Valence must be between -1.0 and 1.0")
                return 'continue'
            
            if not (0.0 <= arousal <= 1.0):
                print("❌ Arousal must be between 0.0 and 1.0")
                return 'continue'
            
            from storage.emotion_model import get_emotion_model
            
            emotion_model = get_emotion_model(memory_log.db_path)
            emotion_model.set_emotion_override(valence, arousal, duration)
            
            mood_label = emotion_model._classify_mood(valence, arousal)
            
            print(f"✅ Emotion override set:")
            print(f"  Valence: {valence:+.2f}, Arousal: {arousal:.2f}")
            print(f"  Mood: {mood_label.title()}")
            print(f"  Duration: {duration:.0f}s")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error setting emotion: {e}")
            return 'continue'
    
    elif cmd == 'emotional_summary':
        """Show emotional summary of recent memory"""
        try:
            hours_back = 24.0
            if len(parts) > 1:
                try:
                    hours_back = float(parts[1])
                except ValueError:
                    print("❌ Invalid hours argument")
                    return 'continue'
            
            # Use enhanced memory system if available
            if ENHANCED_MODE and hasattr(memory_log, 'enhanced_memory'):
                from config.settings import get_user_profile_id
                user_profile_id = get_user_profile_id()
                session_id = memory_log.current_session_id if hasattr(memory_log, 'current_session_id') else None
                
                summary = memory_log.enhanced_memory.get_emotional_summary(
                    user_profile_id, session_id, hours_back
                )
                
                print(f"\n💭 Emotional Memory Summary (last {hours_back}h):")
                print("-" * 50)
                
                if summary.get('emotional_facts_count', 0) == 0:
                    print("No emotionally tagged facts found.")
                    return 'continue'
                
                print(f"Emotionally tagged facts: {summary['emotional_facts_count']}")
                print(f"Average valence: {summary.get('avg_valence', 0.0):+.2f}")
                print(f"Average arousal: {summary.get('avg_arousal', 0.0):.2f}")
                
                val_range = summary.get('valence_range', (0, 0))
                arousal_range = summary.get('arousal_range', (0, 0))
                print(f"Valence range: {val_range[0]:+.2f} to {val_range[1]:+.2f}")
                print(f"Arousal range: {arousal_range[0]:.2f} to {arousal_range[1]:.2f}")
                
                emotion_tags = summary.get('emotion_tags', {})
                if emotion_tags:
                    print(f"\nEmotion tags:")
                    for tag, count in sorted(emotion_tags.items(), key=lambda x: x[1], reverse=True):
                        print(f"  • {tag}: {count}")
                
                mood_contexts = summary.get('mood_contexts', {})
                if mood_contexts:
                    print(f"\nMood contexts:")
                    for mood, count in sorted(mood_contexts.items(), key=lambda x: x[1], reverse=True):
                        print(f"  • {mood}: {count}")
            else:
                print("❌ Enhanced memory system not available")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error generating emotional summary: {e}")
            return 'continue'
    
    elif cmd == 'current_tone':
        """Show current personality tone profile"""
        try:
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            tone_profile = personality_engine.get_current_tone_profile()
            
            print("\n🎭 Current Personality Tone Profile:")
            print("-" * 50)
            print(f"Response Mode: {tone_profile.mode.value.title()}")
            print(f"Mood: {tone_profile.mood_label.title()}")
            print(f"Primary Traits: {', '.join(trait.title() for trait in tone_profile.primary_traits) if tone_profile.primary_traits else 'Developing'}")
            print(f"Emotional Intensity: {tone_profile.emotional_intensity:.2f}")
            print(f"Confidence Level: {tone_profile.confidence_level:.2f}")
            print(f"Conversational Energy: {tone_profile.conversational_energy:.2f}")
            print(f"Valence Bias: {tone_profile.valence_bias:+.2f}")
            
            # Show tone description
            description = tone_profile.get_description()
            print(f"\nTone Summary: {description}")
            
            # Show if mode is overridden
            if personality_engine.mode_override:
                print(f"\n🔒 Manual Override: {personality_engine.mode_override.value.title()}")
            else:
                print(f"\n🔄 Automatic Mode Selection: Active")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error retrieving current tone: {e}")
            return 'continue'
    
    elif cmd == 'set_response_mode':
        """Manually override personality response mode"""
        try:
            if len(parts) < 2:
                print("Usage: set_response_mode <mode>")
                print("Available modes: rational, empathetic, playful, assertive, auto")
                return 'continue'
            
            mode_name = parts[1].lower()
            
            from agents.personality_engine import get_personality_engine, ResponseMode
            
            personality_engine = get_personality_engine(memory_log.db_path)
            
            if mode_name == "auto":
                # Clear override for automatic selection
                personality_engine.set_response_mode_override(None)
                print("✅ Response mode set to automatic selection")
            else:
                # Set specific mode
                mode_map = {
                    "rational": ResponseMode.RATIONAL,
                    "empathetic": ResponseMode.EMPATHETIC,
                    "playful": ResponseMode.PLAYFUL,
                    "assertive": ResponseMode.ASSERTIVE
                }
                
                if mode_name not in mode_map:
                    print(f"❌ Invalid mode: {mode_name}")
                    print("Available modes: rational, empathetic, playful, assertive, auto")
                    return 'continue'
                
                selected_mode = mode_map[mode_name]
                success = personality_engine.set_response_mode_override(selected_mode)
                
                if success:
                    print(f"✅ Response mode manually set to: {mode_name.title()}")
                    print("Use 'set_response_mode auto' to return to automatic selection")
                else:
                    print(f"❌ Failed to set response mode")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error setting response mode: {e}")
            return 'continue'
    
    elif cmd == 'sample_response_modes':
        """Show example text in all four personality modes"""
        try:
            # Get sample text from user or use default
            if len(parts) > 1:
                base_text = " ".join(parts[1:])
            else:
                base_text = "I understand your question and I think the answer is based on the available information."
            
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            samples = personality_engine.get_sample_responses(base_text)
            
            print(f"\n🎭 Response Mode Samples:")
            print(f"Base text: \"{base_text}\"")
            print("-" * 60)
            
            mode_descriptions = {
                "rational": "📊 Clear, technical, evidence-based",
                "empathetic": "💝 Supportive, emotionally aware, gentle",
                "playful": "🎪 Enthusiastic, curious, casual",
                "assertive": "⚡ Direct, confident, concise"
            }
            
            for mode, sample_text in samples.items():
                description = mode_descriptions.get(mode, "")
                print(f"\n{description}")
                print(f"{mode.upper()}: {sample_text}")
            
            print(f"\nUse 'set_response_mode <mode>' to manually select a mode")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error generating sample responses: {e}")
            return 'continue'
    
    elif cmd == 'personality_banner':
        """Show current personality banner"""
        try:
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            banner = personality_engine.get_personality_banner()
            
            print(f"\n{banner}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error showing personality banner: {e}")
            return 'continue'
    
    elif cmd == 'personality_status':
        """Show detailed personality status and evolution history"""
        try:
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            status = personality_engine.get_personality_status()
            
            print("\n🧬 === PERSONALITY STATUS === 🧬")
            print(f"📊 Current State:")
            print(f"   • Tone: {status['current_state']['tone']}")
            print(f"   • Emotional State: {status['current_state']['emotional_state']}")
            print(f"   • Active Mode: {status['current_state']['mode']}")
            print(f"   • Current Mood: {status['current_state']['mood']}")
            
            print(f"\n🎭 Core Traits:")
            for trait, strength in sorted(status['core_traits'].items(), key=lambda x: x[1], reverse=True):
                strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                print(f"   • {trait.title()}: {strength:.2f} [{strength_bar}]")
            
            print(f"\n⚙️ Evolution Info:")
            evolution_info = status['evolution_info']
            print(f"   • Total Evolutions: {evolution_info['total_evolutions']}")
            print(f"   • Last Updated: {evolution_info['last_updated'][:19].replace('T', ' ')}")
            print(f"   • Updated By: {evolution_info['last_updated_by']}")
            print(f"   • Feedback Sensitivity: {evolution_info['feedback_sensitivity']:.2f}")
            print(f"   • Stability Factor: {evolution_info['stability_factor']:.2f}")
            print(f"   • Created: {evolution_info['created'][:10]}")
            
            if status['recent_evolution']:
                recent = status['recent_evolution']
                print(f"\n🔄 Most Recent Evolution:")
                print(f"   • Trigger: {recent['trigger']}")
                print(f"   • Reason: {recent['reason'][:100]}...")
                print(f"   • Changes: {len(recent['changes'])} traits modified")
            
            if status['manual_override']:
                print(f"\n🔒 Manual Override: {status['manual_override']}")
            
            # Phase 29: Add PersonalityEvolver status
            try:
                from agents.personality_evolver import get_personality_evolver
                evolver = get_personality_evolver()
                evolver_status = evolver.get_personality_status()
                
                print(f"\n🧬 **Evolution Status** (Phase 29):")
                evolution_meta = evolver_status['evolution_metadata']
                print(f"   • Total Evolutions: {evolution_meta['total_evolutions']}")
                if evolution_meta['last_evolution'] > 0:
                    from datetime import datetime
                    last_evo = datetime.fromtimestamp(evolution_meta['last_evolution']).strftime('%Y-%m-%d %H:%M')
                    print(f"   • Last Evolution: {last_evo}")
                else:
                    print(f"   • Last Evolution: Never")
                print(f"   • Evolution Magnitude: {evolution_meta['total_evolution_magnitude']:.3f}")
                
                config = evolver_status['configuration']
                print(f"   • Sensitivity Threshold: {config['sensitivity_threshold']}")
                print(f"   • Max Shift Rate: {config['max_shift_rate']}")
                
                recent_history = evolver_status['recent_evolution_history']
                if recent_history:
                    print(f"\n🔄 **Recent Evolution**: {recent_history[0]['summary']}")
                
            except ImportError:
                print(f"\n🧬 **Evolution Status**: PersonalityEvolver not available")
            except Exception as e:
                print(f"\n🧬 **Evolution Status**: Error loading status - {e}")
            
            print(f"\n💡 **Commands**:")
            print(f"   • /personality_evolve - Trigger manual evolution")
            print(f"   • /personality_history - View evolution history")
            print(f"   • /meta_reflect - Trigger weekly evolution check")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error showing personality status: {e}")
            return 'continue'
    
    elif cmd == 'adjust_personality':
        """Manually adjust personality traits"""
        try:
            if len(parts) < 2:
                print("Usage: adjust_personality <trait>=<value> [trait2=value2] ...")
                print("       adjust_personality tone=<mode> emotional_state=<state>")
                print("Examples:")
                print("  adjust_personality empathetic=0.8 curious=0.6")
                print("  adjust_personality tone=playful")
                print("  adjust_personality emotional_state=excited")
                print("\nAvailable tones: rational, empathetic, playful, assertive")
                print("Available states: neutral, calm, excited, focused, warm")
                return 'continue'
            
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            
            # Parse adjustments
            adjustments = {}
            for adjustment in parts[1:]:
                if '=' not in adjustment:
                    print(f"❌ Invalid format: {adjustment}. Use trait=value")
                    continue
                
                trait, value = adjustment.split('=', 1)
                trait = trait.strip().lower()
                value = value.strip().lower()
                
                # Handle special cases
                if trait in ['tone', 'emotional_state']:
                    adjustments[trait] = value
                else:
                    # Handle numeric traits
                    try:
                        adjustments[trait] = float(value)
                    except ValueError:
                        print(f"❌ Invalid numeric value for {trait}: {value}")
                        continue
            
            if not adjustments:
                print("❌ No valid adjustments provided")
                return 'continue'
            
            # Apply adjustments
            success = personality_engine.adjust_personality(**adjustments)
            
            if success:
                print(f"✅ Personality adjusted successfully!")
                print(f"🔄 {len(adjustments)} traits modified")
                
                # Show current banner
                banner = personality_engine.get_personality_banner()
                print(f"\n{banner}")
            else:
                print("❌ No changes were made (values may be identical)")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error adjusting personality: {e}")
            return 'continue'
    
    elif cmd == 'evolve_personality':
        """Evolve personality based on feedback"""
        try:
            if len(parts) < 2:
                print("Usage: evolve_personality feedback=\"<feedback text>\"")
                print("Examples:")
                print("  evolve_personality feedback=\"be more empathetic\"")
                print("  evolve_personality feedback=\"you were too harsh\"")
                print("  evolve_personality feedback=\"I need more analytical responses\"")
                return 'continue'
            
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            
            # Extract feedback from the rest of the command
            feedback_text = ' '.join(parts[1:])
            if feedback_text.startswith('feedback='):
                feedback_text = feedback_text[9:]
            
            # Remove quotes if present
            feedback_text = feedback_text.strip('"\'')
            
            if not feedback_text:
                print("❌ No feedback text provided")
                return 'continue'
            
            # Evolve personality
            print(f"🧬 Processing feedback: \"{feedback_text}\"")
            evolved = personality_engine.evolve_personality(feedback_text, "user_manual")
            
            if evolved:
                print(f"✅ Personality evolved based on feedback!")
                
                # Show updated summary
                summary = personality_engine.get_personality_summary()
                print(f"\n📖 Updated Personality:")
                print(summary)
                
                # Show current banner
                banner = personality_engine.get_personality_banner()
                print(f"\n{banner}")
            else:
                print("ℹ️ No significant changes detected from feedback")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error evolving personality: {e}")
            return 'continue'
    
    elif cmd == 'personality_summary':
        """Get human-readable personality summary"""
        try:
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine(memory_log.db_path)
            summary = personality_engine.get_personality_summary()
            
            print(f"\n📖 === PERSONALITY SUMMARY === 📖")
            print(summary)
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error getting personality summary: {e}")
            return 'continue'
    
    elif cmd == 'set_persona':
        """Set personality traits manually (for testing)"""
        try:
            if len(parts) < 2:
                print("Usage: set_persona <trait1>+<trait2>+...")
                print("Example: set_persona curious+analytical")
                print("Available traits: curious, analytical, empathetic, skeptical, optimistic, resilient")
                return 'continue'
            
            persona_string = parts[1].lower()
            traits = [trait.strip() for trait in persona_string.split('+')]
            
            # Validate traits
            valid_traits = ["curious", "analytical", "empathetic", "skeptical", "optimistic", "resilient"]
            invalid_traits = [trait for trait in traits if trait not in valid_traits]
            
            if invalid_traits:
                print(f"❌ Invalid traits: {', '.join(invalid_traits)}")
                print(f"Available traits: {', '.join(valid_traits)}")
                return 'continue'
            
            # This would require modifying the self-aware model to temporarily override traits
            # For now, just show what would be set
            print(f"🎭 Persona setting requested: {', '.join(trait.title() for trait in traits)}")
            print("Note: Trait override functionality coming soon!")
            print("Current traits are derived from behavioral patterns.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error setting persona: {e}")
            return 'continue'
    
    # Phase 10: Life Narrative & Reflective Memory Commands
    elif cmd == 'life_narrative':
        """Show major life episodes with timestamps"""
        try:
            from storage.life_narrative import get_life_narrative_manager
            
            narrative_manager = get_life_narrative_manager(memory_log.db_path)
            
            # Get limit parameter
            limit = 20
            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                    limit = max(1, min(50, limit))  # Clamp between 1-50
                except ValueError:
                    print("❌ Invalid limit. Using default of 20.")
            
            episodes = narrative_manager.get_all_episodes(limit=limit)
            
            if not episodes:
                print("\n📖 No life episodes found yet.")
                print("💡 Episodes are automatically created as you interact with the system.")
                return 'continue'
            
            print(f"\n📖 Life Narrative - {len(episodes)} Major Episodes")
            print("=" * 60)
            
            for i, episode in enumerate(episodes, 1):
                start_time = datetime.fromtimestamp(episode.start_timestamp)
                date_str = start_time.strftime('%Y-%m-%d %H:%M')
                duration = episode.get_timespan_description()
                emotional_tone = episode.get_emotional_description()
                
                print(f"\n{i:2d}. {episode.title}")
                print(f"    📅 {date_str} | ⏱️  {duration} | 😊 {emotional_tone}")
                print(f"    🎯 Importance: {episode.importance_score:.2f} | 🏷️  Themes: {', '.join(episode.themes[:3])}")
                print(f"    🆔 {episode.episode_id}")
                
                if episode.reflection_notes:
                    print(f"    💭 Reflection: {episode.reflection_notes[:100]}...")
            
            print(f"\n💡 Use 'replay_episode <id>' to relive a specific episode")
            print(f"💡 Use 'summarize_episode <id>' for a quick summary")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error showing life narrative: {e}")
            return 'continue'
    
    elif cmd == 'replay_episode':
        """Replay memory episode as narrative"""
        try:
            if len(parts) < 2:
                print("Usage: replay_episode <episode_id>")
                print("💡 Use 'life_narrative' to see available episodes")
                return 'continue'
            
            episode_id = parts[1]
            
            # Optional parameters
            include_facts = True
            reflection_depth = 'moderate'
            
            if len(parts) > 2:
                if 'brief' in parts[2:]:
                    include_facts = False
                if 'deep' in parts[2:]:
                    reflection_depth = 'deep'
                elif 'surface' in parts[2:]:
                    reflection_depth = 'surface'
            
            from agents.reflective_engine import get_reflective_engine
            
            reflective_engine = get_reflective_engine(memory_log.db_path)
            replay_result = reflective_engine.replay_episode(
                episode_id, 
                include_facts=include_facts,
                reflection_depth=reflection_depth
            )
            
            if 'error' in replay_result:
                print(f"❌ {replay_result['error']}")
                return 'continue'
            
            print("\n" + "=" * 70)
            print("🎬 EPISODE REPLAY")
            print("=" * 70)
            
            # Display narrative
            print(replay_result['narrative_text'])
            
            # Display reflection insights
            if replay_result.get('reflection_insights'):
                print("\n" + "💭 REFLECTION INSIGHTS")
                print("-" * 40)
                for i, insight in enumerate(replay_result['reflection_insights'], 1):
                    print(f"{i}. {insight}")
            
            # Display identity implications
            if replay_result.get('identity_implications'):
                print("\n" + "🧠 IDENTITY IMPLICATIONS")
                print("-" * 40)
                for trait, impact in replay_result['identity_implications'].items():
                    if abs(impact) > 0.05:
                        direction = "↗️" if impact > 0 else "↘️"
                        print(f"{direction} {trait}: {impact:+.2f}")
            
            # Display emotional learning
            if replay_result.get('emotional_learning'):
                print("\n" + "❤️ EMOTIONAL LEARNING")
                print("-" * 40)
                for learning in replay_result['emotional_learning']:
                    print(f"• {learning}")
            
            # Display causal patterns
            if replay_result.get('causal_patterns'):
                print("\n" + "🔗 CAUSAL PATTERNS")
                print("-" * 40)
                for pattern in replay_result['causal_patterns']:
                    print(f"• {pattern}")
            
            print("\n" + "=" * 70)
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error replaying episode: {e}")
            return 'continue'
    
    elif cmd == 'summarize_episode':
        """Short summary of experience and its impact"""
        try:
            if len(parts) < 2:
                print("Usage: summarize_episode <episode_id>")
                print("💡 Use 'life_narrative' to see available episodes")
                return 'continue'
            
            episode_id = parts[1]
            
            from agents.reflective_engine import get_reflective_engine
            
            reflective_engine = get_reflective_engine(memory_log.db_path)
            summary = reflective_engine.generate_episode_summary(episode_id)
            
            if 'error' in summary:
                print(f"❌ {summary['error']}")
                return 'continue'
            
            print(f"\n📋 EPISODE SUMMARY")
            print("=" * 50)
            print(f"🎯 Title: {summary['title']}")
            print(f"📅 Date: {summary['date']}")
            print(f"⏱️  Duration: {summary['duration']}")
            print(f"😊 Emotional Tone: {summary['emotional_tone']}")
            print(f"🎖️  Significance: {summary['importance']}")
            
            if summary['key_themes']:
                print(f"🏷️  Key Themes: {', '.join(summary['key_themes'])}")
            
            print(f"\n📊 Impact: {summary['impact_summary']}")
            print(f"🎓 Life Lesson: {summary['life_lesson']}")
            
            if summary['growth_contribution']:
                print(f"\n🌱 Growth Contribution:")
                for trait, change in summary['growth_contribution'].items():
                    print(f"   {trait}: {change}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error summarizing episode: {e}")
            return 'continue'
    
    elif cmd == 'reflect_on_last_n':
        """Trigger reflection on last N memories"""
        try:
            if len(parts) < 2:
                print("Usage: reflect_on_last_n <days_back> [max_episodes]")
                print("Example: reflect_on_last_n 7 10")
                return 'continue'
            
            try:
                days_back = int(parts[1])
                days_back = max(1, min(30, days_back))  # Clamp between 1-30 days
            except ValueError:
                print("❌ Invalid days_back. Please provide a number between 1-30.")
                return 'continue'
            
            max_episodes = 10
            if len(parts) > 2:
                try:
                    max_episodes = int(parts[2])
                    max_episodes = max(1, min(20, max_episodes))  # Clamp between 1-20
                except ValueError:
                    print("❌ Invalid max_episodes. Using default of 10.")
            
            from agents.reflective_engine import get_reflective_engine
            
            print(f"\n🧠 Reflecting on experiences from the last {days_back} days...")
            print("⏳ This may take a moment...")
            
            reflective_engine = get_reflective_engine(memory_log.db_path)
            reflection_results = reflective_engine.reflect_on_recent_episodes(
                days_back=days_back,
                max_episodes=max_episodes
            )
            
            if 'error' in reflection_results:
                print(f"❌ {reflection_results['error']}")
                return 'continue'
            
            if 'message' in reflection_results:
                print(f"ℹ️  {reflection_results['message']}")
                return 'continue'
            
            print(f"\n🎯 REFLECTION SESSION: {reflection_results['session_id']}")
            print("=" * 60)
            
            episodes_reflected = reflection_results.get('episodes_reflected', [])
            print(f"📚 Episodes Analyzed: {len(episodes_reflected)}")
            
            for episode in episodes_reflected:
                print(f"  • {episode['title']} ({episode['episode_id'][:12]}...)")
            
            # Overall insights
            if reflection_results.get('overall_insights'):
                print(f"\n💡 KEY INSIGHTS")
                print("-" * 30)
                for insight in reflection_results['overall_insights'][:5]:  # Show top 5
                    print(f"• {insight}")
            
            # Identity changes
            if reflection_results.get('identity_changes'):
                print(f"\n🧠 IDENTITY DEVELOPMENT")
                print("-" * 30)
                for trait, change in reflection_results['identity_changes'].items():
                    if abs(change) > 0.05:
                        direction = "📈" if change > 0 else "📉"
                        print(f"{direction} {trait.title()}: {change:+.2f}")
            
            # Life lessons
            if reflection_results.get('life_lessons'):
                print(f"\n🎓 LIFE LESSONS")
                print("-" * 30)
                for lesson in reflection_results['life_lessons']:
                    print(f"• {lesson}")
            
            print(f"\n✨ Reflection complete! Identity traits have been updated based on insights.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error reflecting on recent episodes: {e}")
            return 'continue'
    
    elif cmd == 'autobiography':
        """Full life narrative to date"""
        try:
            # Optional parameters
            max_episodes = 50
            if len(parts) > 1:
                try:
                    max_episodes = int(parts[1])
                    max_episodes = max(5, min(100, max_episodes))  # Clamp between 5-100
                except ValueError:
                    print("❌ Invalid max_episodes. Using default of 50.")
            
            from storage.life_narrative import get_life_narrative_manager
            
            narrative_manager = get_life_narrative_manager(memory_log.db_path)
            
            print("\n📖 Generating my autobiography...")
            print("⏳ This may take a moment to compile my life narrative...")
            
            life_narrative = narrative_manager.generate_life_narrative(max_episodes=max_episodes)
            
            print(life_narrative)
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error generating autobiography: {e}")
            return 'continue'
    
    elif cmd == 'cluster_memories':
        """Manually trigger memory clustering into episodes"""
        try:
            # Optional parameters
            hours_back = 24
            force_recluster = False
            
            if len(parts) > 1:
                try:
                    hours_back = int(parts[1])
                    hours_back = max(1, min(168, hours_back))  # Clamp between 1 hour - 1 week
                except ValueError:
                    print("❌ Invalid hours_back. Using default of 24.")
            
            if 'force' in parts:
                force_recluster = True
            
            from storage.life_narrative import get_life_narrative_manager
            
            print(f"\n🧠 Clustering memories from the last {hours_back} hours...")
            if force_recluster:
                print("🔄 Force reclustering enabled - will recluster existing episodes")
            print("⏳ This may take a moment...")
            
            narrative_manager = get_life_narrative_manager(memory_log.db_path)
            new_episodes = narrative_manager.scan_and_cluster_memories(
                hours_back=hours_back,
                force_recluster=force_recluster
            )
            
            if not new_episodes:
                print("ℹ️  No new episodes created. All recent memories may already be clustered.")
            else:
                print(f"✅ Created {len(new_episodes)} new episodes:")
                for episode in new_episodes:
                    print(f"  🎯 {episode.title} (importance: {episode.importance_score:.2f})")
                    print(f"      Themes: {', '.join(episode.themes)}")
                    print(f"      ID: {episode.episode_id}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error clustering memories: {e}")
            return 'continue'
    
    elif cmd == 'episode_themes':
        """Show episodes organized by themes"""
        try:
            from storage.life_narrative import get_life_narrative_manager
            
            # Get theme filter if provided
            theme_filter = None
            if len(parts) > 1:
                theme_filter = parts[1].lower()
            
            narrative_manager = get_life_narrative_manager(memory_log.db_path)
            
            if theme_filter:
                episodes = narrative_manager.get_episodes_by_theme(theme_filter)
                if not episodes:
                    print(f"📝 No episodes found with theme '{theme_filter}'")
                    return 'continue'
                
                print(f"\n📝 Episodes with theme '{theme_filter}' ({len(episodes)} found)")
                print("=" * 60)
                
                for episode in episodes:
                    start_time = datetime.fromtimestamp(episode.start_timestamp)
                    date_str = start_time.strftime('%Y-%m-%d')
                    print(f"🎯 {episode.title} ({date_str})")
                    print(f"    Importance: {episode.importance_score:.2f} | All themes: {', '.join(episode.themes)}")
                    print(f"    ID: {episode.episode_id}")
                    print()
            
            else:
                # Show all themes with episode counts
                all_episodes = narrative_manager.get_all_episodes()
                
                if not all_episodes:
                    print("📝 No episodes available to analyze themes.")
                    return 'continue'
                
                # Count themes
                theme_counts = {}
                theme_episodes = {}
                
                for episode in all_episodes:
                    for theme in episode.themes:
                        theme_counts[theme] = theme_counts.get(theme, 0) + 1
                        if theme not in theme_episodes:
                            theme_episodes[theme] = []
                        theme_episodes[theme].append(episode)
                
                # Sort themes by count
                sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n📊 Life Themes Analysis ({len(all_episodes)} total episodes)")
                print("=" * 60)
                
                for theme, count in sorted_themes:
                    print(f"🏷️  {theme}: {count} episodes")
                    
                    # Show most recent episode with this theme
                    recent_episode = max(theme_episodes[theme], key=lambda e: e.start_timestamp)
                    recent_date = datetime.fromtimestamp(recent_episode.start_timestamp).strftime('%Y-%m-%d')
                    print(f"    Most recent: {recent_episode.title} ({recent_date})")
                
                print(f"\n💡 Use 'episode_themes <theme_name>' to see all episodes with a specific theme")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error showing episode themes: {e}")
            return 'continue'
    
    # Phase 11: Recursive Planning Commands
    elif RECURSIVE_PLANNING_MODE and cmd == 'plan_goal' and len(parts) > 1:
        goal_text = ' '.join(parts[1:])
        return handle_plan_goal_command(goal_text)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'show_plan' and len(parts) > 1:
        plan_id = parts[1]
        return handle_show_plan_command(plan_id)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'execute_plan' and len(parts) > 1:
        plan_id = parts[1]
        return handle_execute_plan_command(plan_id)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'why_am_i_doing_this' and len(parts) > 1:
        goal_id = parts[1]
        return handle_why_command(goal_id)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'self_prompt':
        return handle_self_prompt_command()
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'list_plans':
        status_filter = parts[1] if len(parts) > 1 else None
        return handle_list_plans_command(status_filter)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'delete_plan' and len(parts) > 1:
        plan_id = parts[1]
        return handle_delete_plan_command(plan_id)
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'plan_stats':
        return handle_plan_stats_command()
    
    elif RECURSIVE_PLANNING_MODE and cmd == 'intention_stats':
        return handle_intention_stats_command()
    
    # Phase 12: Self-Repair Commands
    elif SELF_REPAIR_MODE and cmd == 'self_diagnose':
        return handle_self_diagnose_command()
    
    elif SELF_REPAIR_MODE and cmd == 'self_repair':
        top_n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
        return handle_self_repair_command(top_n)
    
    elif SELF_REPAIR_MODE and cmd == 'show_flaws':
        return handle_show_flaws_command()
    
    elif SELF_REPAIR_MODE and cmd == 'repair_log':
        return handle_repair_log_command()
    
    # Phase 15: Self-Upgrading Commands
    elif cmd == 'self_upgrade':
        component = parts[1] if len(parts) > 1 else None
        return handle_self_upgrade_command(component)
    
    elif cmd == 'upgrade_status':
        return handle_upgrade_status_command()
    
    elif cmd == 'show_upgrade_diff' and len(parts) > 1:
        return handle_show_upgrade_diff_command(parts[1])
    
    elif cmd == 'rollback_upgrade' and len(parts) > 1:
        return handle_rollback_upgrade_command(parts[1])
    
    elif cmd == 'upgrade_ledger':
        return handle_upgrade_ledger_command()
    
    # Phase 16: Autonomous Planning & Decision Layer Commands
    elif cmd == 'auto_plan' and len(parts) >= 2:
        goal = ' '.join(parts[1:])
        return handle_auto_plan_command(goal)
    
    elif cmd == 'plan_tree' and len(parts) > 1:
        return handle_plan_tree_command(parts[1])
    
    elif cmd == 'next_action':
        return handle_next_action_command()
    
    elif cmd == 'reroute' and len(parts) >= 2:
        failed_goal = ' '.join(parts[1:])
        return handle_reroute_command(failed_goal)
    
    # Phase 13: Command Router & Execution Commands
    elif cmd == 'run_shell' and len(parts) >= 2:
        if COMMAND_ROUTER_MODE:
            return handle_run_shell_command(' '.join(parts[1:]))
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'pip_install' and len(parts) >= 2:
        if COMMAND_ROUTER_MODE:
            return handle_pip_install_command(parts[1])
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'run_tool' and len(parts) >= 2:
        if COMMAND_ROUTER_MODE:
            return handle_run_tool_command(' '.join(parts[1:]))
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'restart_self':
        if COMMAND_ROUTER_MODE:
            return handle_restart_self_command()
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'agent_status':
        if COMMAND_ROUTER_MODE:
            return handle_agent_status_command()
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'tool_log':
        if COMMAND_ROUTER_MODE:
            query = ' '.join(parts[1:]) if len(parts) > 1 else ""
            return handle_tool_log_command(query)
        else:
            print("❌ Command router not available")
            return 'continue'
    
    elif cmd == 'tool_stats':
        if COMMAND_ROUTER_MODE:
            return handle_tool_stats_command()
        else:
            print("❌ Command router not available")
            return 'continue'
    
    # Phase 14: Recursive Execution Commands
    elif cmd == 'write_and_run':
        if RECURSIVE_EXECUTION_MODE:
            return handle_write_and_run_command(parts[1:])
        else:
            print("❌ Recursive execution system not available")
            return 'continue'
    
    elif cmd == 'recursive_run':
        if RECURSIVE_EXECUTION_MODE:
            if len(parts) < 2:
                print("❌ Usage: recursive_run <filename>")
                return 'continue'
            return handle_recursive_run_command(parts[1])
        else:
            print("❌ Recursive execution system not available")
            return 'continue'
    
    elif cmd == 'list_generated_files':
        if RECURSIVE_EXECUTION_MODE:
            return handle_list_generated_files_command()
        else:
            print("❌ Recursive execution system not available")
            return 'continue'
    
    elif cmd == 'show_file':
        if RECURSIVE_EXECUTION_MODE:
            if len(parts) < 2:
                print("❌ Usage: show_file <filename>")
                return 'continue'
            return handle_show_file_command(parts[1])
        else:
            print("❌ Recursive execution system not available")
            return 'continue'
    
    elif cmd == 'delete_file':
        if RECURSIVE_EXECUTION_MODE:
            if len(parts) < 2:
                print("❌ Usage: delete_file <filename>")
                return 'continue'
            return handle_delete_file_command(parts[1])
        else:
            print("❌ Recursive execution system not available")
            return 'continue'

    # Phase 18: World Modeling Commands
    elif cmd == 'world_state':
        try:
            from agents.world_modeler import get_world_modeler
            
            world_modeler = get_world_modeler()
            world_state = world_modeler.get_world_state_summary()
            
            print("\n🌍 **Current World State Summary**")
            print("=" * 60)
            
            # Current beliefs
            beliefs = world_state['current_beliefs'][:10]  # Top 10
            if beliefs:
                print("\n💭 **Current Beliefs:**")
                for belief in beliefs:
                    confidence_icon = "🟢" if belief['confidence'] >= 0.8 else "🟡" if belief['confidence'] >= 0.6 else "🔴"
                    recency_icon = "🔥" if belief['recency_score'] >= 0.8 else "⚡" if belief['recency_score'] >= 0.4 else "❄️"
                    print(f"{confidence_icon}{recency_icon} {belief['concept']}")
                    print(f"    Truth: {belief['truth_value']:.2f} | Confidence: {belief['confidence']:.2f}")
                    print(f"    Recency: {belief['recency_score']:.2f} | Source: {belief['source']}")
            else:
                print("\n💭 **Current Beliefs:** None established yet")
            
            # Top causal chains
            causal_chains = world_state['causal_chains'][:5]  # Top 5
            if causal_chains:
                print(f"\n🔗 **Top Causal Chains:**")
                for i, chain in enumerate(causal_chains, 1):
                    strength_icon = "🟢" if chain['strength'] >= 0.7 else "🟡" if chain['strength'] >= 0.4 else "🔴"
                    print(f"{i}. {strength_icon} {chain['explanation']}")
                    print(f"    Strength: {chain['strength']:.2f}")
            else:
                print(f"\n🔗 **Top Causal Chains:** None established yet")
            
            # Statistics
            belief_stats = world_state['belief_statistics']
            graph_stats = world_state['graph_statistics']
            
            print(f"\n📊 **Statistics:**")
            print(f"  Total Beliefs: {belief_stats.get('total_beliefs', 0)}")
            print(f"  Avg Confidence: {belief_stats.get('avg_confidence', 0.0):.2f}")
            print(f"  Graph Nodes: {graph_stats.get('nodes', 0)}")
            print(f"  Causal Edges: {graph_stats.get('edges', 0)}")
            
            print(f"\n💡 Use '/predict_event <event>' to predict outcomes")
            print(f"💡 Use '/belief <fact>' to analyze specific beliefs")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error accessing world state: {e}")
            return 'continue'
    
    elif cmd == 'predict_event' and len(parts) > 1:
        try:
            event = ' '.join(parts[1:])
            
            from agents.world_modeler import get_world_modeler
            
            world_modeler = get_world_modeler()
            prediction = world_modeler.predict_event_outcomes(event)
            
            print(f"\n🔮 **Event Prediction Analysis**")
            print("=" * 60)
            print(f"🎯 **Event:** {prediction['event']}")
            print(f"📊 **Prediction Confidence:** {prediction['prediction_confidence']:.2f}")
            print(f"⏰ **Time Horizon:** {prediction['time_horizon_hours']:.0f} hours")
            
            # Predicted outcomes
            outcomes = prediction['predicted_outcomes']
            if outcomes:
                print(f"\n📈 **Likely Outcomes:**")
                for i, outcome in enumerate(outcomes[:5], 1):
                    prob_icon = "🟢" if outcome['probability'] >= 0.7 else "🟡" if outcome['probability'] >= 0.4 else "🔴"
                    print(f"{i}. {prob_icon} {outcome['concept']}")
                    print(f"    Probability: {outcome['probability']:.2f}")
                    print(f"    Activation: {outcome.get('activation_strength', 0.0):.2f}")
            else:
                print(f"\n📈 **Likely Outcomes:** No clear predictions available")
            
            # Possible causes
            causes = prediction['possible_causes']
            if causes:
                print(f"\n🔍 **Possible Causes:**")
                for i, cause in enumerate(causes[:5], 1):
                    strength_icon = "🟢" if cause.get('causal_strength', 0) >= 0.7 else "🟡" if cause.get('causal_strength', 0) >= 0.4 else "🔴"
                    print(f"{i}. {strength_icon} {cause['concept']}")
                    print(f"    Causal Strength: {cause.get('causal_strength', 0.0):.2f}")
                    print(f"    Current Likelihood: {cause.get('current_likelihood', 0.0):.2f}")
                    if cause.get('indirect'):
                        print(f"    Indirect (through {cause.get('through', 'unknown')})")
            else:
                print(f"\n🔍 **Possible Causes:** No clear causal relationships found")
            
            # Reasoning
            print(f"\n💭 **Prediction Reasoning:**")
            print(f"  {prediction['reasoning']}")
            
            print(f"\n💭 **Causal Reasoning:**")
            print(f"  {prediction['causal_reasoning']}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error predicting event outcomes: {e}")
            return 'continue'
    
    elif cmd == 'belief' and len(parts) > 1:
        try:
            fact = ' '.join(parts[1:])
            
            from agents.world_modeler import get_world_modeler
            
            world_modeler = get_world_modeler()
            analysis = world_modeler.get_belief_analysis(fact)
            
            print(f"\n🧠 **Belief Analysis**")
            print("=" * 60)
            print(f"🎯 **Fact:** {analysis['fact']}")
            
            if analysis.get('status') == 'unknown':
                print(f"❓ {analysis['message']}")
                return 'continue'
            
            # Core belief information
            belief = analysis['belief']
            print(f"\n💭 **Belief Details:**")
            print(f"  Truth Value: {belief['truth_value']:.2f} (0=false, 1=true)")
            print(f"  Confidence: {belief['confidence']:.2f}")
            print(f"  Recency Score: {belief['recency_score']:.2f}")
            print(f"  Source: {belief['source']}")
            print(f"  Updates: {belief['update_count']}")
            
            # Evidence
            if belief['evidence']:
                print(f"\n📚 **Supporting Evidence:**")
                for i, evidence in enumerate(belief['evidence'][-5:], 1):  # Last 5 pieces
                    print(f"  {i}. {evidence}")
            
            # Related beliefs
            related = analysis['related_beliefs']
            if related:
                print(f"\n🔗 **Related Beliefs:**")
                for rel_belief in related[:3]:  # Top 3
                    confidence_icon = "🟢" if rel_belief['confidence'] >= 0.8 else "🟡" if rel_belief['confidence'] >= 0.6 else "🔴"
                    print(f"  {confidence_icon} {rel_belief['concept']}")
                    print(f"      Truth: {rel_belief['truth_value']:.2f} | Confidence: {rel_belief['confidence']:.2f}")
            
            # Causal relationships
            influences = analysis['causal_influences']
            if influences['causes']:
                print(f"\n⬅️ **What Influences This:**")
                for cause in influences['causes']:
                    print(f"  • {cause}")
            
            if influences['effects']:
                print(f"\n➡️ **What This Influences:**")
                for effect in influences['effects']:
                    print(f"  • {effect}")
            
            # Causal explanations
            explanations = analysis['causal_explanations']
            if explanations['incoming']:
                print(f"\n🔍 **Incoming Causal Explanations:**")
                for explanation in explanations['incoming']:
                    if explanation.strip():
                        print(f"  • {explanation}")
            
            if explanations['outgoing']:
                print(f"\n🔍 **Outgoing Causal Explanations:**")
                for explanation in explanations['outgoing']:
                    if explanation.strip():
                        print(f"  • {explanation}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error analyzing belief: {e}")
            return 'continue'

    # Phase 24: Timeline & Temporal Reasoning Commands
    elif cmd == 'timeline' and len(parts) > 1:
        try:
            subject = ' '.join(parts[1:])
            from agents.timeline_engine import get_timeline_engine
            
            timeline = get_timeline_engine()
            events = timeline.get_events_about(subject, time_window_hours=168)  # 1 week
            
            print(f"\n⏰ **Timeline for '{subject}'**")
            print("=" * 60)
            
            if events:
                print(f"Found {len(events)} events in the last week:")
                print()
                
                for event in events[-20:]:  # Show last 20 events
                    event_type_icon = {
                        'observation': '👁️',
                        'inferred': '🧠',
                        'contradiction': '⚠️',
                        'prediction': '🔮',
                        'update': '🔄',
                        'belief_change': '💭'
                    }.get(event.event_type.value, '📝')
                    
                    confidence_icon = "🟢" if event.confidence >= 0.8 else "🟡" if event.confidence >= 0.6 else "🔴"
                    
                    print(f"{event_type_icon}{confidence_icon} **{event.timestamp.strftime('%Y-%m-%d %H:%M')}**")
                    print(f"  {event.fact}")
                    print(f"  Source: {event.source} | Confidence: {event.confidence:.2f}")
                    if event.reasoning_agent:
                        print(f"  Agent: {event.reasoning_agent}")
                    print()
                
                print(f"💡 Use '/timeline_range <start> <end>' for specific time windows")
                print(f"💡 Use '/event_sequence <event1> -> <event2>' to find causal chains")
                
            else:
                print(f"No events found for '{subject}' in the last week.")
                print(f"💡 Try a broader search term or check '/timeline_summary' for recent activity")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error retrieving timeline: {e}")
            return 'continue'
    
    elif cmd == 'timeline_range' and len(parts) >= 3:
        try:
            start_date = parts[1]
            end_date = parts[2]
            
            from datetime import datetime
            from agents.timeline_engine import get_timeline_engine
            
            # Parse dates (simple format: YYYY-MM-DD)
            try:
                start_time = datetime.strptime(start_date, '%Y-%m-%d')
                end_time = datetime.strptime(end_date, '%Y-%m-%d')
                end_time = end_time.replace(hour=23, minute=59, second=59)  # End of day
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-01)")
                return 'continue'
            
            timeline = get_timeline_engine()
            summary = timeline.summarize_timeline((start_time, end_time))
            
            print(f"\n📅 **Timeline Summary: {start_date} to {end_date}**")
            print("=" * 60)
            
            stats = summary['statistics']
            print(f"**Overview:**")
            print(f"  Total Events: {stats['total_events']}")
            print(f"  Average Confidence: {stats['average_confidence']}")
            print(f"  Duration: {summary['time_window']['duration_hours']:.1f} hours")
            print(f"  Unique Sources: {stats['unique_sources']}")
            
            if stats['events_by_type']:
                print(f"\n**Event Types:**")
                for event_type, count in stats['events_by_type'].items():
                    type_icon = {
                        'observation': '👁️',
                        'inferred': '🧠',
                        'contradiction': '⚠️',
                        'prediction': '🔮',
                        'update': '🔄',
                        'belief_change': '💭'
                    }.get(event_type, '📝')
                    print(f"  {type_icon} {event_type.title()}: {count}")
            
            if summary['top_subjects']:
                print(f"\n**Most Active Subjects:**")
                for subject, count in summary['top_subjects'][:5]:
                    print(f"  • {subject}: {count} mentions")
            
            if summary['causal_sequences_detected'] > 0:
                print(f"\n🔗 **Causal Sequences Detected:** {summary['causal_sequences_detected']}")
            
            if summary['anomalies_detected'] > 0:
                print(f"\n⚠️ **Temporal Anomalies Detected:** {summary['anomalies_detected']}")
                print(f"   Use '/temporal_anomalies' to investigate")
            
            patterns = summary['temporal_patterns']
            if patterns.get('peak_hour', {}).get('count', 0) > 0:
                peak_hour = patterns['peak_hour']
                print(f"\n📊 **Peak Activity:** {peak_hour['hour']}:00 ({peak_hour['count']} events)")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error generating timeline summary: {e}")
            return 'continue'
    
    elif cmd == 'event_sequence' and len(parts) > 1:
        try:
            sequence_text = ' '.join(parts[1:])
            
            # Parse the sequence (e.g., "it rained -> street wet -> accidents")
            if '->' not in sequence_text:
                print("❌ Invalid format. Use: /event_sequence <event1> -> <event2> -> <event3>")
                print("   Example: /event_sequence it rained -> street wet -> car accidents")
                return 'continue'
            
            causal_chain = [event.strip() for event in sequence_text.split('->')]
            
            from agents.timeline_engine import get_timeline_engine
            timeline = get_timeline_engine()
            
            sequence = timeline.get_event_sequence(causal_chain)
            
            print(f"\n🔗 **Causal Sequence Analysis**")
            print("=" * 60)
            print(f"Searching for: {' → '.join(causal_chain)}")
            print()
            
            if sequence:
                print(f"✅ **Sequence Found!**")
                print(f"Causal Strength: {sequence.causal_strength:.2f}")
                print(f"Overall Confidence: {sequence.confidence:.2f}")
                print(f"Temporally Consistent: {'✅' if sequence.is_temporally_consistent() else '❌'}")
                print()
                
                print(f"**Event Timeline:**")
                for i, event in enumerate(sequence.events, 1):
                    event_type_icon = {
                        'observation': '👁️',
                        'inferred': '🧠',
                        'contradiction': '⚠️',
                        'prediction': '🔮',
                        'update': '🔄',
                        'belief_change': '💭'
                    }.get(event.event_type.value, '📝')
                    
                    print(f"{i}. {event_type_icon} **{event.timestamp.strftime('%Y-%m-%d %H:%M')}**")
                    print(f"   {event.fact}")
                    print(f"   Confidence: {event.confidence:.2f} | Source: {event.source}")
                    
                    if i < len(sequence.events):
                        time_diff = (sequence.events[i].timestamp - event.timestamp).total_seconds() / 60
                        print(f"   ⏱️ {time_diff:.1f} minutes later...")
                    print()
                
            else:
                print(f"❌ **No matching sequence found**")
                print(f"This could mean:")
                print(f"  • The events haven't occurred in this order")
                print(f"  • The causal relationship is too weak")
                print(f"  • The events are too far apart in time")
                print()
                print(f"💡 Try individual timeline searches: '/timeline <subject>'")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error analyzing event sequence: {e}")
            return 'continue'
    
    elif cmd == 'temporal_anomalies':
        try:
            from agents.timeline_engine import get_timeline_engine
            
            timeline = get_timeline_engine()
            anomalies = timeline.detect_temporal_inconsistencies()
            
            print(f"\n⚠️ **Temporal Anomalies Detection**")
            print("=" * 60)
            
            if anomalies:
                print(f"Found {len(anomalies)} temporal inconsistencies:")
                print()
                
                for i, anomaly in enumerate(anomalies[-10:], 1):  # Show last 10
                    anomaly_type = anomaly['type']
                    anomaly_icon = {
                        'temporal_violation': '⏰',
                        'belief_contradiction': '💭',
                        'impossible_sequence': '🚫'
                    }.get(anomaly_type, '⚠️')
                    
                    print(f"{i}. {anomaly_icon} **{anomaly_type.replace('_', ' ').title()}**")
                    print(f"   {anomaly['description']}")
                    
                    if 'events' in anomaly:
                        if isinstance(anomaly['events'], list) and len(anomaly['events']) > 0:
                            print(f"   Events involved:")
                            for event in anomaly['events'][:3]:  # Show first 3
                                print(f"     • {event}")
                    
                    if 'timestamps' in anomaly:
                        print(f"   Timestamps: {', '.join(anomaly['timestamps'])}")
                    
                    if 'reason' in anomaly:
                        print(f"   Reason: {anomaly['reason']}")
                    
                    print(f"   Detected: {anomaly['detected_at']}")
                    print()
                
                print(f"💡 Temporal anomalies can indicate:")
                print(f"  • Causal relationships with incorrect timing")
                print(f"  • Contradictory beliefs about the same events")
                print(f"  • Data quality issues in the timeline")
                
            else:
                print(f"✅ No temporal anomalies detected!")
                print(f"The timeline appears to be temporally consistent.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error detecting temporal anomalies: {e}")
            return 'continue'
    
    elif cmd == 'timeline_summary':
        try:
            from agents.timeline_engine import get_timeline_engine
            
            timeline = get_timeline_engine()
            stats = timeline.get_timeline_stats()
            
            print(f"\n📊 **Timeline System Status**")
            print("=" * 60)
            
            print(f"**Timeline Overview:**")
            print(f"  Total Events: {stats['total_events']}")
            print(f"  Timeline Span: {stats['timeline_span_hours']:.1f} hours")
            print(f"  Average Confidence: {stats['average_confidence']}")
            print(f"  Unique Subjects: {stats['unique_subjects']}")
            
            if stats['oldest_event']:
                print(f"  Oldest Event: {stats['oldest_event']}")
            if stats['newest_event']:
                print(f"  Newest Event: {stats['newest_event']}")
            
            print(f"\n**Event Distribution:**")
            for event_type, count in stats['event_type_distribution'].items():
                type_icon = {
                    'observation': '👁️',
                    'inferred': '🧠',
                    'contradiction': '⚠️',
                    'prediction': '🔮',
                    'update': '🔄',
                    'belief_change': '💭'
                }.get(event_type, '📝')
                print(f"  {type_icon} {event_type.title()}: {count}")
            
            print(f"\n**Source Distribution:**")
            for source, count in sorted(stats['source_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {source}: {count} events")
            
            print(f"\n**Temporal Analysis:**")
            print(f"  Causal Sequences: {stats['causal_sequences']}")
            print(f"  Detected Anomalies: {stats['detected_anomalies']}")
            
            print(f"\n💡 **Available Commands:**")
            print(f"  • /timeline <subject> - Events about a topic")
            print(f"  • /timeline_range <start> <end> - Events in date range")
            print(f"  • /event_sequence <event1> -> <event2> - Find causal chains")
            print(f"  • /temporal_anomalies - Detect inconsistencies")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error getting timeline summary: {e}")
            return 'continue'

    # Phase 19: Ethical & Constraint Reasoning Commands
    elif cmd == 'list_constraints':
        try:
            from agents.constraint_engine import get_constraint_engine
            
            constraint_engine = get_constraint_engine()
            rules = constraint_engine.list_rules()
            
            print("\n🛡️ **Active Constraint Rules**")
            print("=" * 60)
            
            if rules:
                for i, rule in enumerate(rules, 1):
                    priority_icon = "🔴" if rule.priority.value == "critical" else "🟠" if rule.priority.value == "high" else "🟡" if rule.priority.value == "medium" else "🟢"
                    scope_icon = "🌐" if rule.scope.value == "all" else "⚙️" if rule.scope.value == "execution" else "📋" if rule.scope.value == "planning" else "💬" if rule.scope.value == "communication" else "📝"
                    
                    print(f"{i:2d}. {priority_icon}{scope_icon} **{rule.rule_id}**")
                    print(f"     Condition: {rule.condition}")
                    print(f"     Priority: {rule.priority.value.title()} | Scope: {rule.scope.value.title()}")
                    print(f"     Origin: {rule.origin.value} | Violations: {rule.violation_count}")
                    
                    if rule.description:
                        print(f"     Description: {rule.description}")
                    
                    if rule.tags:
                        print(f"     Tags: {', '.join(rule.tags)}")
                    
                    print()
                
                # Show statistics
                stats = constraint_engine.get_statistics()
                print(f"📊 **Statistics:**")
                print(f"  Total Rules: {stats['total_rules']} ({stats['active_rules']} active)")
                print(f"  Total Violations: {stats['total_violations']}")
                print(f"  Enforcement Mode: {stats['enforcement_mode']}")
                print(f"  Recent Violations (24h): {stats['recent_violations']}")
            else:
                print("No active constraints found.")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error listing constraints: {e}")
            return 'continue'
    
    elif cmd == 'evaluate_action' and len(parts) > 1:
        try:
            action = ' '.join(parts[1:])
            
            from agents.constraint_engine import get_constraint_engine, get_ethical_evaluator
            
            constraint_engine = get_constraint_engine()
            ethical_evaluator = get_ethical_evaluator()
            
            print(f"\n🔍 **Action Evaluation Analysis**")
            print("=" * 60)
            print(f"🎯 **Action:** {action}")
            
            # Basic constraint evaluation
            evaluation = constraint_engine.evaluate(action)
            
            verdict_icons = {
                "allow": "✅",
                "warn": "⚠️", 
                "block": "❌",
                "modify": "🔄"
            }
            
            verdict_icon = verdict_icons.get(evaluation.verdict.value, "❓")
            print(f"\n🛡️ **Constraint Evaluation:** {verdict_icon} {evaluation.verdict.value.upper()}")
            print(f"  Confidence: {evaluation.confidence:.2f}")
            print(f"  Reason: {evaluation.reason}")
            
            if evaluation.triggered_rules:
                print(f"  Triggered Rules: {', '.join(evaluation.triggered_rules)}")
            
            if evaluation.warnings:
                print(f"\n⚠️ **Warnings:**")
                for warning in evaluation.warnings:
                    print(f"  • {warning}")
            
            if evaluation.suggested_modifications:
                print(f"\n💡 **Suggested Modifications:**")
                for suggestion in evaluation.suggested_modifications:
                    print(f"  • {suggestion}")
            
            # Ethical policy evaluation
            ethical_analysis = ethical_evaluator.evaluate_ethical_alignment(action)
            
            print(f"\n🧭 **Ethical Policy Evaluation:**")
            print(f"  Overall Score: {ethical_analysis['overall_ethical_score']:.2f}")
            print(f"  Verdict: {ethical_analysis['ethical_verdict']}")
            print(f"  Confidence: {ethical_analysis['confidence']:.2f}")
            
            framework_scores = ethical_analysis['framework_scores']
            print(f"\n📊 **Framework Scores:**")
            print(f"  Deontological (Duty): {framework_scores['deontological']:.2f}")
            print(f"  Consequentialist (Outcomes): {framework_scores['consequentialist']:.2f}")
            print(f"  Virtue Ethics (Character): {framework_scores['virtue_ethics']:.2f}")
            print(f"  Care Ethics (Relationships): {framework_scores['care_ethics']:.2f}")
            
            if ethical_analysis['recommendations']:
                print(f"\n🎯 **Recommendations:**")
                for rec in ethical_analysis['recommendations'][:5]:  # Top 5
                    print(f"  • {rec}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error evaluating action: {e}")
            return 'continue'
    
    elif cmd == 'add_constraint' and len(parts) > 1:
        try:
            # Parse constraint from command (simplified format)
            constraint_text = ' '.join(parts[1:])
            
            from agents.constraint_engine import get_constraint_engine, ConstraintRule
            import uuid
            
            constraint_engine = get_constraint_engine()
            
            # Create a simple constraint rule
            rule_id = f"user_constraint_{uuid.uuid4().hex[:8]}"
            
            new_rule = ConstraintRule(
                rule_id=rule_id,
                condition=constraint_text,
                scope="all",
                priority="medium",
                origin="user",
                description=f"User-defined constraint: {constraint_text}",
                tags=["user_defined"]
            )
            
            success = constraint_engine.add_rule(new_rule)
            
            if success:
                print(f"✅ **Constraint Added Successfully**")
                print(f"  Rule ID: {rule_id}")
                print(f"  Condition: {constraint_text}")
                print(f"  Priority: Medium")
                print(f"  Scope: All")
                print(f"\n💡 Use '/list_constraints' to see all active constraints")
            else:
                print(f"❌ Failed to add constraint")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error adding constraint: {e}")
            return 'continue'
    
    elif cmd == 'remove_constraint' and len(parts) > 1:
        try:
            rule_id = parts[1]
            
            from agents.constraint_engine import get_constraint_engine
            
            constraint_engine = get_constraint_engine()
            
            # Check if rule exists first
            if rule_id in constraint_engine.constraints:
                rule = constraint_engine.constraints[rule_id]
                success = constraint_engine.remove_rule(rule_id)
                
                if success:
                    print(f"✅ **Constraint Removed Successfully**")
                    print(f"  Rule ID: {rule_id}")
                    print(f"  Condition: {rule.condition}")
                    print(f"\n💡 Use '/list_constraints' to see remaining constraints")
                else:
                    print(f"❌ Failed to remove constraint")
            else:
                print(f"❌ Constraint not found: {rule_id}")
                print(f"💡 Use '/list_constraints' to see available constraint IDs")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error removing constraint: {e}")
            return 'continue'

    # Phase 20: Neural Evolution Tree & Genetic Self-Replication Commands
    elif cmd == 'replicate_genome' and len(parts) > 1:
        try:
            mutation_desc = ' '.join(parts[1:])
            
            # Remove quotes if present
            if mutation_desc.startswith('"') and mutation_desc.endswith('"'):
                mutation_desc = mutation_desc[1:-1]
            
            from agents.evolution_tree import get_evolution_tree, get_self_replicator
            from storage.genome_log import Mutation, MutationType
            import uuid
            
            evolution_tree = get_evolution_tree()
            
            # Create mutation based on description
            mutation = Mutation(
                mutation_id=str(uuid.uuid4()),
                mutation_type=MutationType.CODE_EVOLUTION,
                description=mutation_desc,
                target_component="user_requested",
                changes={"description": mutation_desc, "timestamp": time.time()}
            )
            
            # Fork genome
            new_genome_id = evolution_tree.fork_genome(
                mutations=[mutation],
                creator="user",
                branch_name=f"user_fork_{int(time.time())}"
            )
            
            print(f"🧬 **Genome Replication Successful**")
            print(f"  New Genome ID: {new_genome_id}")
            print(f"  Mutation: {mutation_desc}")
            print(f"  Parent: {evolution_tree.current_genome_id[:8]}")
            print(f"  Status: Experimental")
            print(f"  Branch: user_fork_{int(time.time())}")
            print(f"\n💡 Use '/activate_genome {new_genome_id}' to switch to this genome")
            print(f"💡 Use '/score_genome {new_genome_id} <score>' to rate its performance")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error replicating genome: {e}")
            return 'continue'
    
    elif cmd == 'score_genome' and len(parts) >= 3:
        try:
            genome_id = parts[1]
            score = float(parts[2])
            
            from agents.evolution_tree import get_evolution_tree
            
            evolution_tree = get_evolution_tree()
            
            # Create metrics from score
            metrics = {
                'overall_fitness': score,
                'user_rating': score,
                'manual_score': True
            }
            
            success = evolution_tree.score_genome(genome_id, metrics)
            
            if success:
                genome = evolution_tree.genome_log.get_genome(genome_id)
                
                print(f"📊 **Genome Fitness Updated**")
                print(f"  Genome ID: {genome_id}")
                print(f"  New Fitness: {genome.fitness_score:.3f}")
                print(f"  Status: {genome.status.value}")
                print(f"  Generation: {genome.generation}")
                
                # Check if status changed
                if genome.fitness_score >= 0.9:
                    print(f"  🌟 Promoted to Elite status!")
                elif genome.fitness_score <= 0.2:
                    print(f"  📁 Marked for archival due to low performance")
            else:
                print(f"❌ Failed to update fitness for genome: {genome_id}")
            
            return 'continue'
            
        except ValueError:
            print(f"❌ Invalid score. Please provide a number between 0.0 and 1.0")
            return 'continue'
        except Exception as e:
            print(f"❌ Error scoring genome: {e}")
            return 'continue'
    
    elif cmd == 'lineage_tree':
        try:
            from agents.evolution_tree import get_evolution_tree
            
            evolution_tree = get_evolution_tree()
            lineage_data = evolution_tree.visualize_lineage()
            
            print(f"\n🌳 **Genetic Lineage Tree**")
            print("=" * 60)
            
            if not lineage_data['nodes']:
                print("No genomes found in evolution tree.")
                return 'continue'
            
            print(f"📊 **Overview:**")
            print(f"  Total Genomes: {lineage_data['total_genomes']}")
            print(f"  Root Genome: {lineage_data['root'][:8] if lineage_data['root'] else 'None'}")
            print(f"  Lineage Depth: {max((node['generation'] for node in lineage_data['nodes']), default=0)}")
            
            # Group by generation
            generations = {}
            for node in lineage_data['nodes']:
                gen = node['generation']
                if gen not in generations:
                    generations[gen] = []
                generations[gen].append(node)
            
            print(f"\n🧬 **Lineage by Generation:**")
            for gen in sorted(generations.keys()):
                genomes = generations[gen]
                print(f"\n  Generation {gen}: ({len(genomes)} genomes)")
                
                for genome in genomes[:5]:  # Show max 5 per generation
                    status_icon = {"active": "🟢", "elite": "⭐", "archived": "📁", 
                                 "failed": "❌", "experimental": "🧪"}.get(genome['status'], "❓")
                    fitness_icon = "🔥" if genome['fitness'] >= 0.8 else "👍" if genome['fitness'] >= 0.6 else "👎"
                    
                    print(f"    {status_icon}{fitness_icon} {genome['id'][:8]} | "
                          f"Fitness: {genome['fitness']:.2f} | "
                          f"Mutations: {genome['mutations']} | "
                          f"Branch: {genome.get('label', 'unnamed').split('\\n')[1] if '\\n' in genome.get('label', '') else 'unnamed'}")
                
                if len(genomes) > 5:
                    print(f"    ... and {len(genomes) - 5} more")
            
            # Show current active genome
            current_genome_id = evolution_tree.current_genome_id
            current_node = next((n for n in lineage_data['nodes'] if n['id'] == current_genome_id), None)
            if current_node:
                print(f"\n🎯 **Current Active Genome:**")
                print(f"  ID: {current_genome_id[:8]}")
                print(f"  Generation: {current_node['generation']}")
                print(f"  Fitness: {current_node['fitness']:.3f}")
                print(f"  Status: {current_node['status']}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error displaying lineage tree: {e}")
            return 'continue'
    
    elif cmd == 'list_genomes':
        try:
            from agents.evolution_tree import get_evolution_tree
            from storage.genome_log import GenomeStatus
            
            evolution_tree = get_evolution_tree()
            all_genomes = evolution_tree.genome_log.get_all_genomes()
            
            print(f"\n🧬 **All Genomes in Evolution Tree**")
            print("=" * 80)
            
            if not all_genomes:
                print("No genomes found.")
                return 'continue'
            
            # Group by status
            status_groups = {}
            for genome in all_genomes:
                status = genome.status.value
                if status not in status_groups:
                    status_groups[status] = []
                status_groups[status].append(genome)
            
            # Display by status priority
            status_order = ['active', 'elite', 'experimental', 'archived', 'failed']
            
            for status in status_order:
                if status in status_groups:
                    genomes = status_groups[status]
                    status_icons = {
                        'active': '🟢', 'elite': '⭐', 'experimental': '🧪', 
                        'archived': '📁', 'failed': '❌'
                    }
                    icon = status_icons.get(status, '❓')
                    
                    print(f"\n{icon} **{status.title()} Genomes ({len(genomes)}):**")
                    
                    # Sort by fitness descending
                    genomes.sort(key=lambda g: g.fitness_score, reverse=True)
                    
                    for genome in genomes[:10]:  # Show top 10 per status
                        fitness_bar = "🔥" if genome.fitness_score >= 0.8 else "👍" if genome.fitness_score >= 0.6 else "👎" if genome.fitness_score >= 0.4 else "💀"
                        
                        created = datetime.fromtimestamp(genome.origin_timestamp).strftime('%m-%d %H:%M')
                        
                        print(f"  {fitness_bar} {genome.genome_id[:12]} | "
                              f"Fitness: {genome.fitness_score:.3f} | "
                              f"Gen: {genome.generation:2d} | "
                              f"Mutations: {len(genome.mutations):2d} | "
                              f"Branch: {(genome.branch_name or 'unnamed')[:15]:15s} | "
                              f"Created: {created}")
                    
                    if len(genomes) > 10:
                        print(f"  ... and {len(genomes) - 10} more")
            
            # Statistics
            stats = evolution_tree.get_evolution_statistics()
            print(f"\n📊 **Statistics:**")
            print(f"  Total Genomes: {stats['total_genomes']}")
            print(f"  Average Fitness: {stats['fitness_stats']['average']:.3f}")
            print(f"  Max Generation: {stats['max_generation']}")
            print(f"  Active Branches: {stats['active_branches']}")
            print(f"  Elite Count: {stats['elite_count']}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error listing genomes: {e}")
            return 'continue'
    
    elif cmd == 'activate_genome' and len(parts) > 1:
        try:
            genome_id = parts[1]
            
            from agents.evolution_tree import get_evolution_tree
            
            evolution_tree = get_evolution_tree()
            
            # Check if genome exists
            target_genome = evolution_tree.genome_log.get_genome(genome_id)
            if not target_genome:
                print(f"❌ Genome not found: {genome_id}")
                print(f"💡 Use '/list_genomes' to see available genomes")
                return 'continue'
            
            # Get current genome for comparison
            current_genome = evolution_tree.genome_log.get_genome(evolution_tree.current_genome_id)
            
            # Activate genome
            success = evolution_tree.activate_genome(genome_id)
            
            if success:
                print(f"🎯 **Genome Activation Successful**")
                print(f"  New Active Genome: {genome_id}")
                print(f"  Previous Genome: {current_genome.genome_id[:8] if current_genome else 'None'}")
                print(f"  Generation: {target_genome.generation}")
                print(f"  Fitness: {target_genome.fitness_score:.3f}")
                print(f"  Status: {target_genome.status.value}")
                print(f"  Branch: {target_genome.branch_name or 'unnamed'}")
                print(f"  Mutations: {len(target_genome.mutations)}")
                
                if target_genome.mutations:
                    print(f"\n🧬 **Recent Mutations:**")
                    for mutation in target_genome.mutations[-3:]:  # Show last 3
                        print(f"  • {mutation.mutation_type.value}: {mutation.description}")
                
                print(f"\n⚠️  **Note:** This switches the system's active genome configuration.")
                print(f"    Monitor performance and use '/score_genome' to evaluate results.")
            else:
                print(f"❌ Failed to activate genome: {genome_id}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error activating genome: {e}")
            return 'continue'
    
    # ===== PHASE 21: DEBATE SYSTEM CLI HANDLERS =====
    
    elif cmd == 'debate':
        if len(parts) < 2:
            print("❌ Usage: /debate \"<claim to debate>\"")
            return 'continue'
        
        claim = ' '.join(parts[1:]).strip('"')
        result = debate_claim(claim)
        print(result)
        return 'continue'
    
    elif cmd == 'critique':
        if len(parts) < 2:
            print("❌ Usage: /critique \"<plan or action to critique>\"")
            return 'continue'
        
        plan_or_action = ' '.join(parts[1:]).strip('"')
        result = critique_plan(plan_or_action)
        print(result)
        return 'continue'
    
    elif cmd == 'simulate_debate':
        if len(parts) < 2:
            print("❌ Usage: /simulate_debate \"<topic with opposing sides>\"")
            print("💡 Example: /simulate_debate \"Plan A: gradual vs Plan B: immediate\"")
            return 'continue'
        
        topic = ' '.join(parts[1:]).strip('"')
        result = simulate_debate(topic)
        print(result)
        return 'continue'
    
    elif cmd == 'reflect_on':
        if len(parts) < 2:
            print("❌ Usage: /reflect_on \"<memory_id>\"")
            return 'continue'
        
        memory_id = ' '.join(parts[1:]).strip('"')
        result = reflect_on_memory(memory_id)
        print(result)
        return 'continue'
    
    # ===== PHASE 22: MEMORY CONSOLIDATION CLI HANDLERS =====
    
    elif cmd == 'consolidate_memory':
        mode = parts[1] if len(parts) > 1 else "incremental"
        if mode not in ["incremental", "full", "prune", "cluster"]:
            print("❌ Usage: /consolidate_memory [incremental|full|prune|cluster]")
            return 'continue'
        
        result = consolidate_memory(mode)
        print(result)
        return 'continue'
    
    elif cmd == 'list_clusters':
        cluster_type = parts[1] if len(parts) > 1 else None
        if cluster_type and cluster_type not in ["semantic", "temporal", "causal"]:
            print("❌ Usage: /list_clusters [semantic|temporal|causal]")
            return 'continue'
        
        result = list_clusters(cluster_type)
        print(result)
        return 'continue'
    
    elif cmd == 'prioritize_memory':
        try:
            limit = int(parts[1]) if len(parts) > 1 else 20
            if limit < 1 or limit > 100:
                print("❌ Usage: /prioritize_memory [1-100]")
                return 'continue'
        except ValueError:
            print("❌ Usage: /prioritize_memory [number]")
            return 'continue'
        
        result = prioritize_memory(limit)
        print(result)
        return 'continue'
    
    elif cmd == 'memory_stats':
        result = memory_stats()
        print(result)
        return 'continue'
    
    # ===== PHASE 22: AGENT REPLICATION CLI HANDLERS =====
    
    elif cmd == 'fork_agent':
        if len(parts) < 2:
            print("❌ Usage: /fork_agent <agent_name>")
            print("💡 Available agents: critic, planner, debater, reflector, self_prompter")
            return 'continue'
        
        agent_name = parts[1]
        result = fork_agent_command(agent_name)
        print(result)
        return 'continue'
    
    elif cmd == 'run_fork':
        if len(parts) < 2:
            print("❌ Usage: /run_fork <fork_id>")
            return 'continue'
        
        fork_id = parts[1]
        result = run_fork_command(fork_id)
        print(result)
        return 'continue'
    
    elif cmd == 'mutate_agent':
        if len(parts) < 2:
            print("❌ Usage: /mutate_agent <fork_id>")
            return 'continue'
        
        fork_id = parts[1]
        result = mutate_agent_command(fork_id)
        print(result)
        return 'continue'
    
    elif cmd == 'score_forks':
        result = score_forks_command()
        print(result)
        return 'continue'
    
    elif cmd == 'prune_forks':
        result = prune_forks_command()
        print(result)
        return 'continue'
    
    elif cmd == 'fork_status':
        result = fork_status_command()
        print(result)
        return 'continue'
    
    elif cmd == 'promote_agent':
        if len(parts) < 2:
            print("❌ Usage: /promote_agent <agent_name>")
            print("💡 Available agents: critic, planner, debater, reflector, self_prompter, etc.")
            return 'continue'
        
        agent_name = parts[1]
        result = promote_agent_command(agent_name)
        print(result)
        return 'continue'
    
    elif cmd == 'retire_agent':
        if len(parts) < 2:
            print("❌ Usage: /retire_agent <agent_name>")
            print("💡 Available agents: critic, planner, debater, reflector, self_prompter, etc.")
            return 'continue'
        
        agent_name = parts[1]
        result = retire_agent_command(agent_name)
        print(result)
        return 'continue'
    
    elif cmd == 'lifecycle_status':
        if len(parts) < 2:
            print("❌ Usage: /lifecycle_status <agent_name>")
            print("💡 Available agents: critic, planner, debater, reflector, self_prompter, etc.")
            return 'continue'
        
        agent_name = parts[1]
        result = lifecycle_status_command(agent_name)
        print(result)
        return 'continue'
    
    elif cmd == 'evaluate_lifecycle':
        if len(parts) < 2:
            print("❌ Usage: /evaluate_lifecycle <agent_name>")
            print("💡 Available agents: critic, planner, debater, reflector, self_prompter, etc.")
            return 'continue'
        
        agent_name = parts[1]
        result = evaluate_lifecycle_command(agent_name)
        print(result)
        return 'continue'
    
    elif cmd == 'list_agents_with_drift':
        result = list_agents_with_drift_command()
        print(result)
        return 'continue'
    
    elif cmd == 'test_vectorizer':
        """Test vectorizer backends"""
        backend = parts[1] if len(parts) > 1 else 'default'
        
        try:
            from vector_memory import get_vectorizer
            from vector_memory.config import get_vectorizer_info
            import time
            
            print(f"\n🧪 Testing Vectorizer Backend: {backend}")
            print("=" * 60)
            
            # Get vectorizer function
            vectorizer = get_vectorizer(backend)
            
            # Test with sample text
            test_texts = [
                "Hello world",
                "The quick brown fox jumps over the lazy dog",
                "Artificial intelligence and machine learning"
            ]
            
            total_time = 0
            vectors = []
            
            for i, text in enumerate(test_texts, 1):
                print(f"\n📝 Test {i}: '{text}'")
                
                start_time = time.time()
                vector = vectorizer(text)
                end_time = time.time()
                
                duration = end_time - start_time
                total_time += duration
                vectors.append(vector)
                
                print(f"✅ Generated vector of length {len(vector)}")
                print(f"⏱️  Time: {duration:.3f}s")
                
                # Show first few values
                if len(vector) > 0:
                    preview = [f"{v:.3f}" for v in vector[:5]]
                    print(f"🔍 Preview: [{', '.join(preview)}, ...]")
            
            # Test determinism (same input should give same output)
            print(f"\n🔄 Testing Determinism...")
            repeat_vector = vectorizer(test_texts[0])
            is_deterministic = vectors[0] == repeat_vector
            print(f"✅ Deterministic: {is_deterministic}")
            
            # Summary
            print(f"\n📊 Summary:")
            print(f"Backend: {backend}")
            print(f"Vector dimensions: {len(vectors[0]) if vectors else 0}")
            print(f"Average time: {total_time/len(test_texts):.3f}s")
            print(f"Total time: {total_time:.3f}s")
            print(f"Deterministic: {is_deterministic}")
            
            # Backend info
            info = get_vectorizer_info()
            if 'backend_info' in info:
                backend_info = info['backend_info']
                print(f"\nBackend Info: {backend_info.get('description', 'N/A')}")
                if 'features' in backend_info:
                    print(f"Features: {', '.join(backend_info['features'])}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error testing vectorizer '{backend}': {e}")
            print("\nAvailable backends: default, hrrformer, vecsymr, hlb")
            return 'continue'

    # Phase 25: Reflective Self-Awareness Commands
    elif cmd == 'self_summary':
        """Show current identity snapshot"""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            summary = self_model.generate_self_summary()
            
            print(summary)
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error generating self summary: {e}")
            return 'continue'
    
    elif cmd == 'self_journal':
        """Show self-reflection journal"""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            journal = self_model.get_self_journal()
            
            if not journal:
                print("📔 Self-reflection journal is empty.")
                return 'continue'
            
            print(f"\n📔 SELF-REFLECTION JOURNAL ({len(journal)} entries)")
            print("=" * 60)
            
            # Show recent entries (last 10)
            recent_entries = journal[-10:] if len(journal) > 10 else journal
            
            for entry in recent_entries:
                timestamp = entry.get('timestamp', 'Unknown')
                trigger = entry.get('trigger', 'Unknown')
                summary = entry.get('summary', 'No summary')
                
                print(f"\n🕐 {timestamp}")
                print(f"📍 Trigger: {trigger}")
                print(f"💭 {summary}")
                
                # Show changes if available
                changes = entry.get('changes', {})
                if changes and len(changes) > 0:
                    print("🔄 Changes:")
                    for key, value in changes.items():
                        if key == "traits" and isinstance(value, dict):
                            for trait, change in value.items():
                                old_val, new_val = change
                                direction = "↗️" if new_val > old_val else "↘️"
                                print(f"   {direction} {trait}: {old_val:.2f} → {new_val:.2f}")
                        elif isinstance(value, list) and len(value) == 2:
                            old_val, new_val = value
                            print(f"   • {key}: {old_val} → {new_val}")
                        else:
                            print(f"   • {key}: {value}")
                
                print("-" * 40)
            
            if len(journal) > 10:
                print(f"\n(Showing last 10 of {len(journal)} total entries)")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error accessing self journal: {e}")
            return 'continue'
    
    elif cmd == 'self_reflect':
        """Manually trigger a reflection cycle"""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            result = self_model.manual_reflection()
            
            print(f"🧠 SELF-REFLECTION TRIGGERED")
            print(f"{result}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error during self reflection: {e}")
            return 'continue'
    
    elif cmd == 'self_sync':
        """Sync self model with personality engine"""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            success = self_model.sync_from_personality_engine()
            
            if success:
                print("✅ Successfully synced self model with personality engine")
                
                # Show brief update
                summary = self_model.generate_self_summary()
                recent_changes = self_model.get_recent_changes()
                
                if recent_changes.get('status') == 'evolving':
                    print("\n🔄 Recent changes detected:")
                    for change in recent_changes.get('recent_changes', [])[-3:]:
                        print(f"   • {change.get('summary', 'Unknown change')}")
                else:
                    print("🟢 Identity remains stable after sync")
            else:
                print("❌ Failed to sync self model with personality engine")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error syncing self model: {e}")
            return 'continue'
    
    elif cmd == 'contract':
        """Show agent contract information"""
        try:
            # Import contract functionality
            from agents.agent_contract import load_or_create_contract
            from agents.registry import get_agent_registry
            
            if len(parts) < 2:
                print("📋 Agent Contracts")
                print("Usage: contract <agent_name>")
                print("Available agents:")
                
                # Get agent registry to list available agents
                try:
                    registry = get_agent_registry()
                    agent_names = registry.get_agent_names() if hasattr(registry, 'get_agent_names') else []
                    if agent_names:
                        for agent_name in sorted(agent_names):
                            print(f"  • {agent_name}")
                    else:
                        print("  • planner, critic, debater, reflector, architect_analyzer")
                        print("  • code_refactorer, world_modeler, constraint_engine, self_healer")
                except:
                    print("  • planner, critic, debater, reflector, architect_analyzer")
                    print("  • code_refactorer, world_modeler, constraint_engine, self_healer")
                return 'continue'
            
            agent_name = parts[1].lower()
            
            # Load contract
            contract = load_or_create_contract(agent_name)
            
            if contract:
                summary = contract.get_summary()
                
                print(f"\n📋 Contract for {agent_name.title()}Agent")
                print("=" * 50)
                print(f"Purpose: {contract.purpose}")
                print(f"Version: {contract.version}")
                print(f"Last Updated: {contract.last_updated.strftime('%Y-%m-%d %H:%M')}")
                print()
                
                print("💪 Capabilities:")
                for cap in contract.capabilities:
                    print(f"  • {cap}")
                print()
                
                print("🎯 Preferred Tasks:")
                for task in contract.preferred_tasks:
                    print(f"  • {task}")
                print()
                
                print("📊 Confidence Vector:")
                for skill, confidence in sorted(contract.confidence_vector.items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                    print(f"  {skill:15} [{bar}] {confidence:.2f}")
                print()
                
                if contract.weaknesses:
                    print("⚠️  Known Weaknesses:")
                    for weakness in contract.weaknesses:
                        print(f"  • {weakness}")
                    print()
                
                # Performance history
                if summary.get('recent_performance'):
                    perf = summary['recent_performance']
                    print("📈 Recent Performance:")
                    print(f"  Success Rate: {perf['success_rate']:.1%}")
                    print(f"  Avg Performance: {perf['avg_performance']:.2f}")
                    print(f"  Total Tasks: {perf['total_tasks']}")
                    print()
                
                # Specialization trends
                if contract.specialization_drift:
                    print("🔄 Specialization Trends:")
                    for task_type, trend in contract.specialization_drift.items():
                        if len(trend) >= 3:
                            recent_trend = contract.get_specialization_trend(task_type)
                            if recent_trend is not None:
                                trend_str = "📈" if recent_trend > 0.05 else "📉" if recent_trend < -0.05 else "➡️"
                                print(f"  {task_type}: {trend_str} {recent_trend:+.3f}")
                
            else:
                print(f"❌ Could not load contract for {agent_name}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error showing contract: {e}")
            return 'continue'
    
    elif cmd == 'score_task':
        """Score all agents for a given task"""
        try:
            from agents.agent_contract import score_all_agents_for_task
            
            if len(parts) < 2:
                print("🎯 Task Scoring")
                print("Usage: score_task <task_description>")
                print("Example: score_task 'analyze system performance issues'")
                return 'continue'
            
            # Join all parts except command to form task description
            task_description = ' '.join(parts[1:])
            
            print(f"\n🎯 Scoring agents for task: '{task_description}'")
            print("=" * 60)
            
            # Score all agents
            agent_scores = score_all_agents_for_task(task_description)
            
            if not agent_scores:
                print("❌ No agents available for scoring")
                return 'continue'
            
            print(f"📊 Results ({len(agent_scores)} agents scored):")
            print()
            
            for i, agent in enumerate(agent_scores, 1):
                score = agent['alignment_score']
                agent_name = agent['agent_name']
                purpose = agent.get('purpose', 'No purpose defined')
                
                # Score visualization
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                # Score interpretation
                if score > 0.8:
                    score_emoji = "🟢"
                elif score > 0.6:
                    score_emoji = "🟡"
                elif score > 0.4:
                    score_emoji = "🟠"
                else:
                    score_emoji = "🔴"
                
                print(f"{i:2}. {score_emoji} {agent_name:<20} [{bar}] {score:.3f}")
                print(f"    Purpose: {purpose[:80]}{'...' if len(purpose) > 80 else ''}")
                
                if i == 1:
                    print(f"    👑 RECOMMENDED AGENT")
                print()
            
            # Recommendation summary
            best_agent = agent_scores[0]
            print(f"🏆 Best Match: {best_agent['agent_name']} (Score: {best_agent['alignment_score']:.3f})")
            
            confidence_level = "High" if best_agent['alignment_score'] > 0.7 else "Medium" if best_agent['alignment_score'] > 0.4 else "Low"
            print(f"🎯 Confidence: {confidence_level}")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error scoring task: {e}")
            return 'continue'
    
    elif cmd == 'update_contract':
        """Update agent contract with performance feedback"""
        try:
            from agents.agent_contract import load_or_create_contract
            
            if len(parts) < 3:
                print("🔄 Update Agent Contract")
                print("Usage: update_contract <agent_name> feedback=\"feedback_string\"")
                print("       update_contract <agent_name> success=true/false score=0.8")
                print("Examples:")
                print("  update_contract planner feedback=\"excellent task breakdown\"")
                print("  update_contract critic success=false score=0.3")
                return 'continue'
            
            agent_name = parts[1].lower()
            
            # Parse feedback parameters
            feedback_data = {}
            for part in parts[2:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip().strip('"\'')
                    
                    # Convert boolean and numeric values
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif key == 'score' or key.endswith('_score'):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    elif key.startswith('feedback'):
                        key = 'notes'  # Map feedback to notes
                    
                    feedback_data[key] = value
            
            if not feedback_data:
                print("❌ No valid feedback parameters provided")
                return 'continue'
            
            # Load contract
            contract = load_or_create_contract(agent_name)
            
            if not contract:
                print(f"❌ Could not load contract for {agent_name}")
                return 'continue'
            
            # Set defaults for required feedback fields
            if 'task_type' not in feedback_data:
                feedback_data['task_type'] = 'general'
            if 'success' not in feedback_data:
                feedback_data['success'] = True
            if 'performance_score' not in feedback_data:
                feedback_data['performance_score'] = 0.7
            
            print(f"\n🔄 Updating contract for {agent_name}")
            print(f"Feedback data: {feedback_data}")
            
            # Update contract
            contract.update_from_performance_feedback(feedback_data)
            
            # Save updated contract
            contract.save_to_file(f"output/contracts/{agent_name}.json")
            
            print(f"✅ Contract updated for {agent_name}")
            print(f"Performance history now contains {len(contract.performance_history)} entries")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error updating contract: {e}")
            return 'continue'
    
    elif cmd == 'list_contracts':
        """List all available agent contracts"""
        try:
            import os
            from pathlib import Path
            from agents.agent_contract import AgentContract
            
            contracts_dir = Path("output/contracts")
            
            if not contracts_dir.exists():
                print("❌ Contracts directory not found")
                return 'continue'
            
            contract_files = list(contracts_dir.glob("*.json"))
            
            if not contract_files:
                print("❌ No contract files found")
                return 'continue'
            
            print(f"\n📋 Available Agent Contracts ({len(contract_files)} found)")
            print("=" * 70)
            
            contracts_data = []
            
            for contract_file in sorted(contract_files):
                try:
                    contract = AgentContract.load_from_file(contract_file)
                    if contract:
                        summary = contract.get_summary()
                        contracts_data.append({
                            'name': contract.agent_name,
                            'purpose': contract.purpose,
                            'version': contract.version,
                            'last_updated': contract.last_updated,
                            'capabilities_count': len(contract.capabilities),
                            'recent_performance': summary.get('recent_performance')
                        })
                except Exception as e:
                    print(f"⚠️  Error loading {contract_file.name}: {e}")
            
            # Sort by name
            contracts_data.sort(key=lambda x: x['name'])
            
            for i, contract_data in enumerate(contracts_data, 1):
                name = contract_data['name']
                purpose = contract_data['purpose']
                version = contract_data['version']
                updated = contract_data['last_updated'].strftime('%Y-%m-%d')
                caps_count = contract_data['capabilities_count']
                
                print(f"{i:2}. {name:<20} v{version}")
                print(f"    📝 Purpose: {purpose[:60]}{'...' if len(purpose) > 60 else ''}")
                print(f"    💪 Capabilities: {caps_count}, Updated: {updated}")
                
                # Show performance if available
                if contract_data['recent_performance']:
                    perf = contract_data['recent_performance']
                    success_rate = perf['success_rate']
                    total_tasks = perf['total_tasks']
                    
                    perf_emoji = "🟢" if success_rate > 0.8 else "🟡" if success_rate > 0.6 else "🔴"
                    print(f"    {perf_emoji} Performance: {success_rate:.1%} success ({total_tasks} tasks)")
                
                print()
            
            print(f"💡 Use 'contract <agent_name>' to view detailed contract information")
            print(f"💡 Use 'score_task <task>' to find the best agent for a task")
            
            return 'continue'
            
        except Exception as e:
            print(f"❌ Error listing contracts: {e}")
            return 'continue'

    # === PHASE 26: COGNITIVE DISSONANCE COMMANDS ===
    
    elif cmd == 'dissonance_report':
        result = dissonance_report_command()
        print(result)
        return 'continue'
    
    elif cmd == 'resolve_dissonance':
        # Parse optional parameters
        belief_id = parts[1] if len(parts) > 1 else None
        force_evolution = len(parts) > 2 and parts[2].lower() in ['true', 'yes', '1', 'force']
        result = resolve_dissonance_command(belief_id, force_evolution)
        print(result)
        return 'continue'
    
    elif cmd == 'dissonance_history':
        # Parse optional hours parameter
        hours = 24
        if len(parts) > 1:
            try:
                hours = int(parts[1])
            except ValueError:
                print("❌ Invalid hours parameter. Using default of 24 hours.")
                hours = 24
        result = dissonance_history_command(hours)
        print(result)
        return 'continue'
    
    # Phase 27 pt 2: Meta-Self Agent Commands
    elif cmd == 'meta_status':
        return handle_meta_status_command()
    
    elif cmd == 'meta_reflect':
        return handle_meta_reflect_command()
    
    elif cmd == 'meta_goals':
        limit = 10  # default
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                print("❌ Invalid limit parameter. Using default of 10.")
                limit = 10
        return handle_meta_goals_command(limit)
    
    # Phase 29: Personality Evolution Commands
    elif cmd == 'personality_evolve':
        return handle_personality_evolve_command()
    
    elif cmd == 'personality_history':
        limit = 10  # default
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                print("❌ Invalid limit parameter. Using default of 10.")
                limit = 10
        return handle_personality_history_command(limit)
    
    # Phase 28: Autonomous Planning Commands
    elif cmd == 'plan_status':
        return handle_plan_status_command()
    
    elif cmd == 'plan_next':
        return handle_plan_next_command()
    
    elif cmd == 'plan_eval':
        return handle_plan_eval_command()
    
    # Phase 31: Fast Reflex Agent Commands
    elif cmd == 'reflex_mode' and len(parts) > 1:
        mode = parts[1].lower()
        return handle_reflex_mode_command(mode)
    
    elif cmd == 'reflex_status':
        return handle_reflex_status_command()
    
    # Phase 32: Distributed Agent Mesh Commands
    elif cmd == 'mesh_status':
        return handle_mesh_status_command()
    
    elif cmd == 'mesh_enable':
        return handle_mesh_enable_command()
    
    elif cmd == 'mesh_disable':
        return handle_mesh_disable_command()
    
    elif cmd == 'mesh_sync':
        return handle_mesh_sync_command()
    
    elif cmd == 'mesh_agents':
        return handle_mesh_agents_command()
    
    # Phase 33: Neural Control Hooks Commands  
    elif cmd == 'arbiter_trace':
        limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
        return handle_arbiter_trace_command(limit)

    # At the end, return 'not_handled' if no match
    return 'not_handled'

def start_dashboard():
    """Start the MeRNSTA dashboard server (FastAPI + React UI)"""
    print(f"\n🚀 Starting dashboard at http://localhost:{dashboard_port} ...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=dashboard_port, reload=True) 

# ===== PHASE 11: RECURSIVE PLANNING CLI HANDLERS =====

def handle_plan_goal_command(goal_text: str) -> str:
    """Handle /plan_goal <goal text> command"""
    try:
        planner = RecursivePlanner()
        plan = planner.plan_goal(goal_text)
        
        print(f"\n🎯 **Plan Generated for:** {goal_text}")
        print(f"📋 **Plan ID:** {plan.plan_id}")
        print(f"🔧 **Plan Type:** {plan.plan_type}")
        print(f"📊 **Confidence:** {plan.confidence:.2f}")
        print(f"⭐ **Priority:** {plan.priority}")
        print(f"📅 **Created:** {plan.created_at}")
        
        print(f"\n📝 **Steps ({len(plan.steps)}):**")
        for i, step in enumerate(plan.steps, 1):
            print(f"\n{i}. **{step.subgoal}**")
            print(f"   💭 Why: {step.why}")
            print(f"   🎯 Expected: {step.expected_result}")
            if step.prerequisites:
                print(f"   📋 Prerequisites: {', '.join(step.prerequisites)}")
            if step.resources_needed:
                print(f"   🔧 Resources: {', '.join(step.resources_needed)}")
        
        if plan.success_criteria:
            print(f"\n✅ **Success Criteria:**")
            for criterion in plan.success_criteria:
                print(f"   • {criterion}")
        
        if plan.risk_factors:
            print(f"\n⚠️ **Risk Factors:**")
            for risk in plan.risk_factors:
                print(f"   • {risk}")
        
        # Score the plan
        score = planner.score_plan(plan)
        print(f"\n📊 **Plan Score:** {score:.2f}/1.0")
        
        print(f"\n💡 Use 'execute_plan {plan.plan_id}' to execute this plan")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error generating plan: {e}")
        return 'continue'

def handle_show_plan_command(plan_id: str) -> str:
    """Handle /show_plan <plan_id> command"""
    try:
        plan_memory = PlanMemory()
        plan = plan_memory.get_plan_by_id(plan_id)
        
        if not plan:
            print(f"❌ Plan not found: {plan_id}")
            return 'continue'
        
        # Convert to dict if needed for display
        if hasattr(plan, '__dict__'):
            plan_dict = plan.__dict__
        else:
            plan_dict = plan
        
        print(f"\n🎯 **Plan Details**")
        print(f"📋 **ID:** {plan_dict['plan_id']}")
        print(f"🎯 **Goal:** {plan_dict['goal_text']}")
        print(f"📊 **Status:** {plan_dict.get('status', 'unknown')}")
        print(f"🔧 **Type:** {plan_dict.get('plan_type', 'sequential')}")
        print(f"📊 **Confidence:** {plan_dict.get('confidence', 0.0):.2f}")
        print(f"⭐ **Priority:** {plan_dict.get('priority', 1)}")
        print(f"📅 **Created:** {plan_dict.get('created_at', 'unknown')}")
        print(f"🔄 **Updated:** {plan_dict.get('updated_at', 'unknown')}")
        
        if plan_dict.get('parent_goal_id'):
            print(f"👆 **Parent Goal:** {plan_dict['parent_goal_id']}")
        
        steps = plan_dict.get('steps', [])
        print(f"\n📝 **Steps ({len(steps)}):**")
        for i, step in enumerate(steps, 1):
            if hasattr(step, '__dict__'):
                step_dict = step.__dict__
            else:
                step_dict = step
                
            status_emoji = {
                'pending': '⏳',
                'in_progress': '🔄', 
                'completed': '✅',
                'failed': '❌'
            }.get(step_dict.get('status', 'pending'), '❓')
            
            print(f"\n{i}. {status_emoji} **{step_dict.get('subgoal', 'Unknown step')}**")
            print(f"   💭 Why: {step_dict.get('why', 'No reason provided')}")
            print(f"   🎯 Expected: {step_dict.get('expected_result', 'No expected result')}")
            
            prereqs = step_dict.get('prerequisites', [])
            if prereqs:
                print(f"   📋 Prerequisites: {', '.join(prereqs)}")
            
            resources = step_dict.get('resources_needed', [])
            if resources:
                print(f"   🔧 Resources: {', '.join(resources)}")
        
        success_criteria = plan_dict.get('success_criteria', [])
        if success_criteria:
            print(f"\n✅ **Success Criteria:**")
            for criterion in success_criteria:
                print(f"   • {criterion}")
        
        risk_factors = plan_dict.get('risk_factors', [])
        if risk_factors:
            print(f"\n⚠️ **Risk Factors:**")
            for risk in risk_factors:
                print(f"   • {risk}")
        
        intention_chain = plan_dict.get('intention_chain', [])
        if intention_chain:
            print(f"\n🧠 **Intention Chain:**")
            for intention in intention_chain:
                print(f"   • {intention}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error showing plan: {e}")
        return 'continue'

def handle_execute_plan_command(plan_id: str) -> str:
    """Handle /execute_plan <plan_id> command"""
    try:
        plan_memory = PlanMemory()
        plan = plan_memory.get_plan_by_id(plan_id)
        
        if not plan:
            print(f"❌ Plan not found: {plan_id}")
            return 'continue'
        
        planner = RecursivePlanner()
        
        print(f"\n🚀 **Executing Plan:** {plan_id}")
        print(f"🎯 **Goal:** {getattr(plan, 'goal_text', plan.get('goal_text', 'Unknown')) if hasattr(plan, 'goal_text') or isinstance(plan, dict) else 'Unknown'}")
        print("⏳ Starting execution...\n")
        
        results = planner.execute_plan(plan)
        
        print(f"\n📊 **Execution Results:**")
        print(f"✅ **Success:** {results.get('overall_success', False)}")
        print(f"📈 **Completion:** {results.get('completion_percentage', 0):.1f}%")
        print(f"⏱️ **Duration:** {results.get('execution_start', '')} to {results.get('execution_end', '')}")
        
        executed_steps = results.get('steps_executed', [])
        if executed_steps:
            print(f"\n✅ **Completed Steps ({len(executed_steps)}):**")
            for step in executed_steps:
                print(f"   • {step.get('subgoal', 'Unknown step')}")
        
        failed_steps = results.get('steps_failed', [])
        if failed_steps:
            print(f"\n❌ **Failed Steps ({len(failed_steps)}):**")
            for step in failed_steps:
                print(f"   • {step.get('subgoal', 'Unknown step')}: {step.get('reason', 'Unknown reason')}")
        
        execution_log = results.get('execution_log', [])
        if execution_log:
            print(f"\n📝 **Execution Log:**")
            for entry in execution_log[-5:]:  # Show last 5 entries
                timestamp = entry.get('timestamp', '')[:19]  # Truncate timestamp
                print(f"   {timestamp}: {entry.get('message', '')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error executing plan: {e}")
        return 'continue'

def handle_why_command(goal_id: str) -> str:
    """Handle /why_am_i_doing_this <goal_id> command"""
    try:
        intention_model = IntentionModel()
        explanation = intention_model.trace_why_formatted(goal_id)
        
        print(explanation)
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error tracing intention: {e}")
        return 'continue'

def handle_self_prompt_command() -> str:
    """Handle /self_prompt command"""
    try:
        self_prompter = SelfPromptGenerator()
        goals = self_prompter.propose_goals()
        
        if not goals:
            print("🤔 No self-directed goals identified at this time.")
            print("💡 This could mean:")
            print("   • System is performing well")
            print("   • No recent failure patterns detected")
            print("   • Memory analysis didn't find improvement opportunities")
            return 'continue'
        
        print(f"\n🧠 **Self-Generated Goals ({len(goals)}):**")
        print("🎯 These goals were autonomously identified based on system analysis\n")
        
        for i, goal in enumerate(goals, 1):
            print(f"{i}. {goal}")
        
        print(f"\n💡 **Next Steps:**")
        print(f"   • Use 'plan_goal <goal text>' to create a detailed plan")
        print(f"   • Goals are prioritized by impact and feasibility")
        print(f"   • System will continue monitoring for new opportunities")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error generating self-directed goals: {e}")
        return 'continue'

def handle_list_plans_command(status_filter: str = None) -> str:
    """Handle /list_plans [status] command"""
    try:
        plan_memory = PlanMemory()
        
        if status_filter:
            plans = plan_memory.get_plans_by_status(status_filter)
            print(f"\n📋 **Plans with status: {status_filter}**")
        else:
            # Get plans from multiple statuses
            all_plans = []
            for status in ['active', 'completed', 'failed', 'draft']:
                status_plans = plan_memory.get_plans_by_status(status, limit=10)
                all_plans.extend(status_plans)
            plans = sorted(all_plans, key=lambda p: p.get('updated_at', ''), reverse=True)[:20]
            print(f"\n📋 **Recent Plans (last 20)**")
        
        if not plans:
            print("   No plans found.")
            return 'continue'
        
        print(f"\n{'ID':<10} {'Status':<12} {'Goal':<50} {'Updated':<20}")
        print("-" * 95)
        
        for plan in plans:
            plan_id = plan.get('plan_id', 'unknown')[:8]
            status = plan.get('status', 'unknown')
            goal = plan.get('goal_text', 'No goal text')[:45]
            updated = plan.get('updated_at', 'unknown')[:19]
            
            status_emoji = {
                'draft': '📝',
                'active': '🔄',
                'completed': '✅',
                'failed': '❌',
                'abandoned': '🚫'
            }.get(status, '❓')
            
            print(f"{plan_id:<10} {status_emoji} {status:<10} {goal:<50} {updated:<20}")
        
        print(f"\n💡 Use 'show_plan <plan_id>' to see detailed plan information")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error listing plans: {e}")
        return 'continue'

def handle_delete_plan_command(plan_id: str) -> str:
    """Handle /delete_plan <plan_id> command"""
    try:
        plan_memory = PlanMemory()
        
        # Check if plan exists
        plan = plan_memory.get_plan_by_id(plan_id)
        if not plan:
            print(f"❌ Plan not found: {plan_id}")
            return 'continue'
        
        # Get goal text for confirmation
        goal_text = getattr(plan, 'goal_text', plan.get('goal_text', 'Unknown')) if hasattr(plan, 'goal_text') or isinstance(plan, dict) else 'Unknown'
        
        # Delete the plan
        success = plan_memory.delete_plan(plan_id)
        
        if success:
            print(f"✅ Plan deleted: {plan_id}")
            print(f"   Goal: {goal_text}")
        else:
            print(f"❌ Failed to delete plan: {plan_id}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error deleting plan: {e}")
        return 'continue'

def handle_plan_stats_command() -> str:
    """Handle /plan_stats command"""
    try:
        plan_memory = PlanMemory()
        stats = plan_memory.get_plan_statistics()
        
        if not stats:
            print("❌ Could not retrieve plan statistics")
            return 'continue'
        
        print(f"\n📊 **Plan Memory Statistics**")
        print(f"📋 **Total Plans:** {stats.get('total_plans', 0)}")
        print(f"🚀 **Total Executions:** {stats.get('total_executions', 0)}")
        print(f"🎯 **Executed Plans:** {stats.get('executed_plans', 0)}")
        print(f"📈 **Average Success Rate:** {stats.get('average_success_rate', 0.0):.1%}")
        
        status_breakdown = stats.get('status_breakdown', {})
        if status_breakdown:
            print(f"\n📊 **Status Breakdown:**")
            for status, count in status_breakdown.items():
                status_emoji = {
                    'draft': '📝',
                    'active': '🔄', 
                    'completed': '✅',
                    'failed': '❌',
                    'abandoned': '🚫'
                }.get(status, '❓')
                print(f"   {status_emoji} {status.title()}: {count}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error getting plan statistics: {e}")
        return 'continue'

def handle_intention_stats_command() -> str:
    """Handle /intention_stats command"""
    try:
        intention_model = IntentionModel()
        stats = intention_model.get_intention_statistics()
        
        if not stats:
            print("❌ Could not retrieve intention statistics")
            return 'continue'
        
        print(f"\n🧠 **Intention Model Statistics**")
        print(f"🎯 **Total Intentions:** {stats.get('total_intentions', 0)}")
        print(f"✅ **Fulfilled:** {stats.get('fulfilled_count', 0)} ({stats.get('fulfillment_rate', 0.0):.1%})")
        print(f"❌ **Abandoned:** {stats.get('abandoned_count', 0)} ({stats.get('abandonment_rate', 0.0):.1%})")
        print(f"🔄 **Active:** {stats.get('active_count', 0)}")
        print(f"🔗 **Relationships:** {stats.get('relationships_count', 0)}")
        
        drive_breakdown = stats.get('drive_breakdown', {})
        if drive_breakdown:
            print(f"\n🎭 **Top Motivation Drives:**")
            for drive, count in list(drive_breakdown.items())[:5]:
                print(f"   • {drive}: {count}")
        
        importance_by_depth = stats.get('importance_by_depth', {})
        if importance_by_depth:
            print(f"\n📊 **Average Importance by Depth:**")
            for depth, importance in importance_by_depth.items():
                print(f"   Depth {depth}: {importance:.2f}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error getting intention statistics: {e}")
        return 'continue'

# ===== PHASE 12: SELF-REPAIR CLI HANDLERS =====

def handle_self_diagnose_command() -> str:
    """Handle /self_diagnose command"""
    try:
        self_healer = SelfHealer()
        
        print("\n🔍 **Running System Diagnostic...**")
        print("⏳ Analyzing code health, architecture patterns, and system metrics...\n")
        
        diagnostic_report = self_healer.run_diagnostic_suite()
        
        print(f"📊 **System Health Report**")
        print(f"🎯 **Overall Health Score:** {diagnostic_report.system_health_score:.2f}/1.0")
        
        # Health score interpretation
        if diagnostic_report.system_health_score >= 0.8:
            health_status = "🟢 Excellent"
        elif diagnostic_report.system_health_score >= 0.6:
            health_status = "🟡 Good"
        elif diagnostic_report.system_health_score >= 0.4:
            health_status = "🟠 Fair"
        else:
            health_status = "🔴 Poor"
        
        print(f"📈 **System Status:** {health_status}")
        print(f"🐛 **Issues Found:** {len(diagnostic_report.issues)}")
        print(f"⚠️ **Patterns Detected:** {len(diagnostic_report.patterns)}")
        
        # Issue breakdown
        if diagnostic_report.issues:
            issue_counts = {}
            for issue in diagnostic_report.issues:
                issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1
            
            print(f"\n🔍 **Issue Breakdown:**")
            for severity in ['critical', 'high', 'medium', 'low']:
                count = issue_counts.get(severity, 0)
                if count > 0:
                    severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': '💡'}
                    print(f"   {severity_emoji.get(severity, '•')} {severity.title()}: {count}")
        
        # Critical issues
        critical_issues = [i for i in diagnostic_report.issues if i.severity == 'critical']
        if critical_issues:
            print(f"\n🚨 **Critical Issues Requiring Immediate Attention:**")
            for issue in critical_issues[:5]:
                print(f"   • {issue.description} in {issue.component}")
        
        # Architecture patterns
        critical_patterns = [p for p in diagnostic_report.patterns if p.risk_level in ['critical', 'high']]
        if critical_patterns:
            print(f"\n🏗️ **High-Risk Architecture Patterns:**")
            for pattern in critical_patterns[:3]:
                print(f"   • {pattern.name}: {pattern.frequency} occurrences")
        
        # Top recommendations
        if diagnostic_report.recommendations:
            print(f"\n💡 **Top Recommendations:**")
            for rec in diagnostic_report.recommendations[:3]:
                print(f"   • {rec}")
        
        # Top repair goals
        if diagnostic_report.repair_goals:
            print(f"\n🎯 **Generated Repair Goals:**")
            for i, goal in enumerate(diagnostic_report.repair_goals[:5], 1):
                print(f"   {i}. {goal}")
        
        print(f"\n💡 **Next Steps:**")
        print(f"   • Use '/self_repair {len(diagnostic_report.repair_goals[:3])}' to execute top repair goals")
        print(f"   • Use '/show_flaws' to see detailed issue analysis")
        print(f"   • Use '/repair_log' to view repair history")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error running diagnostic: {e}")
        return 'continue'

def handle_self_repair_command(top_n: int = 3) -> str:
    """Handle /self_repair [top_n] command"""
    try:
        self_healer = SelfHealer()
        planner = RecursivePlanner()
        repair_log = SelfRepairLog()
        
        print(f"\n🔧 **Executing Self-Repair for Top {top_n} Issues**")
        
        # Run diagnostic to get repair goals
        diagnostic_report = self_healer.run_diagnostic_suite()
        
        if not diagnostic_report.repair_goals:
            print("✅ No repair goals identified - system appears healthy!")
            return 'continue'
        
        top_goals = diagnostic_report.repair_goals[:top_n]
        
        print(f"🎯 **Repair Goals Selected:**")
        for i, goal in enumerate(top_goals, 1):
            print(f"   {i}. {goal}")
        
        print(f"\n⏳ **Executing Repairs...**")
        
        for i, goal in enumerate(top_goals, 1):
            print(f"\n🔄 **Repair {i}/{len(top_goals)}: {goal}**")
            
            # Log repair attempt start
            attempt_id = repair_log.log_repair_attempt(
                goal=goal,
                approach="Automated self-repair using recursive planning",
                issues_addressed=[issue.issue_id for issue in diagnostic_report.issues if issue.severity in ['critical', 'high']]
            )
            
            try:
                # Generate plan for repair
                plan = planner.plan_goal(goal)
                
                print(f"   📋 Generated plan with {len(plan.steps)} steps")
                
                # Execute the plan
                results = planner.execute_plan(plan)
                
                success = results.get('overall_success', False)
                completion = results.get('completion_percentage', 0)
                
                print(f"   📊 Execution: {completion:.1f}% complete, Success: {success}")
                
                # Log completion
                repair_log.complete_repair_attempt(
                    attempt_id=attempt_id,
                    result=f"Plan execution: {completion:.1f}% complete",
                    score=completion / 100.0,
                    issues_resolved=results.get('steps_executed', []),
                    side_effects=[]
                )
                
                if success:
                    print(f"   ✅ Repair completed successfully")
                else:
                    print(f"   ⚠️ Repair partially completed ({completion:.1f}%)")
            
            except Exception as e:
                print(f"   ❌ Repair failed: {e}")
                repair_log.complete_repair_attempt(
                    attempt_id=attempt_id,
                    result=f"Repair failed: {str(e)}",
                    score=0.0,
                    issues_resolved=[],
                    side_effects=[str(e)]
                )
        
        print(f"\n📊 **Self-Repair Session Complete**")
        print(f"   • {top_n} repair goals attempted")
        print(f"   • Use '/repair_log' to view detailed results")
        print(f"   • Use '/self_diagnose' to verify improvements")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error executing self-repair: {e}")
        return 'continue'

def handle_show_flaws_command() -> str:
    """Handle /show_flaws command"""
    try:
        self_healer = SelfHealer()
        
        print("\n🔍 **Detailed System Flaw Analysis**")
        
        # Get issues and patterns
        issues = self_healer.analyze_code_health()
        patterns = self_healer.detect_architecture_flaws()
        
        if not issues and not patterns:
            print("✅ No significant flaws detected - system appears healthy!")
            return 'continue'
        
        # Group issues by category
        issues_by_category = {}
        for issue in issues:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
        
        # Display issues by category
        for category, category_issues in issues_by_category.items():
            print(f"\n📂 **{category.replace('_', ' ').title()} Issues ({len(category_issues)})**")
            
            # Sort by severity and impact
            sorted_issues = sorted(category_issues, 
                                 key=lambda x: (x.severity == 'critical', x.severity == 'high', x.impact_score),
                                 reverse=True)
            
            for issue in sorted_issues[:10]:  # Show top 10 per category
                severity_emoji = {
                    'critical': '🚨',
                    'high': '⚠️',
                    'medium': '⚡',
                    'low': '💡'
                }
                
                print(f"   {severity_emoji.get(issue.severity, '•')} **{issue.description}**")
                print(f"      📁 Component: {issue.component}")
                print(f"      📊 Impact: {issue.impact_score:.2f}, Difficulty: {issue.fix_difficulty:.2f}")
                
                if issue.evidence:
                    print(f"      🔍 Evidence: {issue.evidence[0]}")
                
                if issue.repair_suggestions:
                    print(f"      💡 Suggestion: {issue.repair_suggestions[0]}")
                
                print()
        
        # Display architecture patterns
        if patterns:
            print(f"\n🏗️ **Architecture Patterns ({len(patterns)})**")
            
            # Sort by risk level
            sorted_patterns = sorted(patterns, 
                                   key=lambda x: (x.risk_level == 'critical', x.risk_level == 'high', x.frequency),
                                   reverse=True)
            
            for pattern in sorted_patterns[:10]:
                risk_emoji = {
                    'critical': '🚨',
                    'high': '⚠️',
                    'medium': '⚡',
                    'low': '💡'
                }
                
                print(f"   {risk_emoji.get(pattern.risk_level, '•')} **{pattern.name}**")
                print(f"      📝 Description: {pattern.description}")
                print(f"      📊 Frequency: {pattern.frequency}, Risk: {pattern.risk_level}")
                print(f"      🎯 Action: {pattern.recommended_action}")
                
                if pattern.locations:
                    locations_display = pattern.locations[:3]
                    if len(pattern.locations) > 3:
                        locations_display.append(f"... and {len(pattern.locations) - 3} more")
                    print(f"      📍 Locations: {', '.join(locations_display)}")
                
                print()
        
        print(f"💡 **Next Steps:**")
        print(f"   • Use '/self_repair' to address high-priority issues")
        print(f"   • Use '/plan_goal <specific issue>' to create targeted repair plans")
        print(f"   • Review code in flagged components for manual improvements")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error analyzing flaws: {e}")
        return 'continue'

def handle_repair_log_command() -> str:
    """Handle /repair_log command"""
    try:
        repair_log = SelfRepairLog()
        
        print("\n📋 **Self-Repair History**")
        
        # Get recent repair summary
        summary = repair_log.summarize_recent_repairs(days_back=30)
        
        if not summary or summary.get('total_attempts', 0) == 0:
            print("📝 No repair attempts found in the last 30 days")
            return 'continue'
        
        print(f"📊 **Last 30 Days Summary:**")
        print(f"   🎯 Total Attempts: {summary.get('total_attempts', 0)}")
        print(f"   ✅ Successful: {summary.get('successful_attempts', 0)}")
        print(f"   ❌ Failed: {summary.get('failed_attempts', 0)}")
        print(f"   📈 Success Rate: {summary.get('success_rate', 0.0):.1%}")
        print(f"   📊 Average Score: {summary.get('average_score', 0.0):.2f}")
        
        # Most common goals
        common_goals = summary.get('common_goals', [])
        if common_goals:
            print(f"\n🎯 **Most Common Repair Goals:**")
            for goal_info in common_goals[:5]:
                print(f"   • {goal_info['goal']} ({goal_info['attempts']} attempts, {goal_info['avg_score']:.2f} avg score)")
        
        # Successful approaches
        successful_approaches = summary.get('successful_approaches', [])
        if successful_approaches:
            print(f"\n✅ **Most Successful Approaches:**")
            for approach_info in successful_approaches[:5]:
                print(f"   • {approach_info['approach']} ({approach_info['attempts']} uses, {approach_info['avg_score']:.2f} avg score)")
        
        # Recent trends
        daily_trends = summary.get('daily_trends', [])
        if daily_trends:
            print(f"\n📈 **Recent Daily Activity:**")
            for trend in daily_trends[:7]:
                print(f"   {trend['date']}: {trend['attempts']} attempts (avg score: {trend['avg_score']:.2f})")
        
        # Common side effects
        side_effects = summary.get('common_side_effects', {})
        if side_effects:
            print(f"\n⚠️ **Common Side Effects:**")
            for effect, count in list(side_effects.items())[:5]:
                print(f"   • {effect}: {count} occurrences")
        
        # Failed repairs analysis
        failed_repairs = repair_log.get_failed_repairs(limit=10, days_back=30)
        if failed_repairs:
            print(f"\n❌ **Recent Failed Repairs ({len(failed_repairs)}):**")
            for repair in failed_repairs[:5]:
                print(f"   • {repair['goal']}")
                print(f"     📅 {repair['start_time'][:10]} - Score: {repair['score']:.2f}")
                print(f"     📝 {repair['result']}")
                print()
        
        # Health trend
        health_trend = repair_log.get_health_trend(days_back=30)
        if len(health_trend) >= 2:
            recent_health = health_trend[-1]['health_score']
            previous_health = health_trend[0]['health_score']
            health_change = recent_health - previous_health
            
            print(f"📊 **System Health Trend:**")
            if health_change > 0.05:
                print(f"   📈 Improving: {previous_health:.2f} → {recent_health:.2f} (+{health_change:.2f})")
            elif health_change < -0.05:
                print(f"   📉 Declining: {previous_health:.2f} → {recent_health:.2f} ({health_change:.2f})")
            else:
                print(f"   ➡️ Stable: ~{recent_health:.2f}")
        
        print(f"\n💡 **Next Steps:**")
        print(f"   • Use '/self_diagnose' to run a fresh system analysis")
        print(f"   • Review failed repairs to improve repair strategies")
        print(f"   • Use successful approaches for similar future repairs")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error accessing repair log: {e}")
        return 'continue'

# Phase 13: Command Router Handler Functions

def handle_run_shell_command(command: str) -> str:
    """Handle shell command execution via command router."""
    try:
        print(f"🐚 Executing shell command: {command}")
        result = route_command(f"/run_shell \"{command}\"", "user")
        
        if result['success']:
            if result['output']:
                print(f"📤 Output:\n{result['output']}")
            if result['error']:
                print(f"⚠️ Stderr:\n{result['error']}")
            print(f"✅ Command completed (exit code: {result.get('exit_code', 0)})")
        else:
            print(f"❌ Command failed: {result['error']}")
        
        return 'continue'
    except Exception as e:
        print(f"❌ Error executing shell command: {e}")
        return 'continue'

def handle_pip_install_command(package: str) -> str:
    """Handle pip install command via command router."""
    try:
        print(f"📦 Installing package: {package}")
        result = route_command(f"/pip_install {package}", "user")
        
        if result['success']:
            if result['output']:
                print(f"📤 Output:\n{result['output']}")
            if result['error']:
                print(f"⚠️ Stderr:\n{result['error']}")
            print(f"✅ Package installation completed")
        else:
            print(f"❌ Package installation failed: {result['error']}")
        
        return 'continue'
    except Exception as e:
        print(f"❌ Error installing package: {e}")
        return 'continue'

def handle_run_tool_command(tool_args: str) -> str:
    """Handle tool execution command via command router."""
    try:
        print(f"🔧 Running tool: {tool_args}")
        result = route_command(f"/run_tool {tool_args}", "user")
        
        if result['success']:
            print(f"✅ Tool execution result:\n{result['output']}")
        else:
            print(f"❌ Tool execution failed: {result['error']}")
        
        return 'continue'
    except Exception as e:
        print(f"❌ Error running tool: {e}")
        return 'continue'

def handle_restart_self_command() -> str:
    """Handle system restart command via command router."""
    try:
        print("🔄 Restarting MeRNSTA system...")
        result = route_command("/restart_self", "user")
        
        if result['success']:
            print(f"✅ {result['output']}")
            # Note: This would actually restart the system
        else:
            print(f"❌ Restart failed: {result['error']}")
        
        return 'continue'
    except Exception as e:
        print(f"❌ Error restarting system: {e}")
        return 'continue'

def handle_agent_status_command() -> str:
    """Handle agent status command via command router."""
    try:
        print("🤖 Getting agent status...")
        result = route_command("/agent_status", "user")
        
        if result['success']:
            try:
                import json
                status = json.loads(result['output'])
                print("\n📊 Agent Registry Status:")
                print(f"   • Enabled: {status.get('enabled', False)}")
                print(f"   • Total Agents: {status.get('total_agents', 0)}")
                print(f"   • Agent Names: {', '.join(status.get('agent_names', []))}")
                print(f"   • Debate Mode: {status.get('debate_mode', False)}")
                
                agent_health = status.get('agent_health', {})
                if agent_health:
                    print("\n🏥 Agent Health:")
                    for name, health in agent_health.items():
                        print(f"   • {name}: {'✅' if health.get('initialized') else '❌'}")
            except json.JSONDecodeError:
                print(f"📊 Agent Status:\n{result['output']}")
        else:
            print(f"❌ Failed to get agent status: {result['error']}")
        
        return 'continue'
    except Exception as e:
        print(f"❌ Error getting agent status: {e}")
        return 'continue'

def handle_tool_log_command(query: str = "") -> str:
    """Handle tool log query command."""
    try:
        if COMMAND_ROUTER_MODE:
            tool_logger = get_tool_logger()
            logs = tool_logger.query_logs(query, limit=20)
            
            print(f"\n📋 Tool Execution Log {'(filtered)' if query else '(recent 20)'}")
            print("-" * 80)
            
            if not logs:
                print("No log entries found.")
                return 'continue'
            
            for entry in logs:
                log_type = entry.get('log_type', 'unknown')
                timestamp = entry.get('timestamp', '')[:19]  # YYYY-MM-DD HH:MM:SS
                
                if log_type == 'command':
                    command = entry.get('command', '')[:50]
                    success = '✅' if entry.get('success') else '❌'
                    executor = entry.get('executor', 'unknown')
                    print(f"{timestamp} {success} [{executor}] {command}")
                    
                elif log_type == 'tool':
                    tool_name = entry.get('tool_name', '')
                    success = '✅' if entry.get('success') else '❌'
                    executor = entry.get('executor', 'unknown')
                    print(f"{timestamp} {success} [{executor}] Tool: {tool_name}")
                    
                elif log_type == 'agent':
                    agent_name = entry.get('agent_name', '')
                    method_name = entry.get('method_name', '')
                    success = '✅' if entry.get('success') else '❌'
                    executor = entry.get('executor', 'unknown')
                    print(f"{timestamp} {success} [{executor}] Agent: {agent_name}.{method_name}")
            
            return 'continue'
        else:
            print("❌ Tool logger not available")
            return 'continue'
    except Exception as e:
        print(f"❌ Error querying tool log: {e}")
        return 'continue'

def handle_tool_stats_command() -> str:
    """Handle tool statistics command."""
    try:
        if COMMAND_ROUTER_MODE:
            tool_logger = get_tool_logger()
            stats = tool_logger.get_execution_stats(hours=24)
            
            print("\n📊 Tool Usage Statistics (Last 24 Hours)")
            print("-" * 50)
            print(f"📋 Command Executions: {stats['command_executions']}")
            print(f"   • Successful: {stats['successful_commands']}")
            print(f"   • Failed: {stats['failed_commands']}")
            print(f"🔧 Tool Usage: {stats['tool_usage']}")
            print(f"🤖 Agent Method Calls: {stats['agent_method_calls']}")
            print(f"📡 System Events: {stats['system_events']}")
            
            if stats['top_commands']:
                print(f"\n🔝 Top Command Types:")
                for cmd in stats['top_commands']:
                    print(f"   • {cmd['type']}: {cmd['count']} times")
            
            if stats['top_tools']:
                print(f"\n🔧 Top Tools:")
                for tool in stats['top_tools']:
                    print(f"   • {tool['name']}: {tool['count']} times")
            
            if stats['top_agents']:
                print(f"\n🤖 Top Agents:")
                for agent in stats['top_agents']:
                    print(f"   • {agent['name']}: {agent['count']} calls")
            
            # Get tool registry status
            tool_registry = get_tool_registry()
            registry_status = tool_registry.get_status()
            
            print(f"\n🏭 Tool Registry Status:")
            print(f"   • Unrestricted Mode: {registry_status['unrestricted_mode']}")
            print(f"   • Shell Commands: {registry_status['shell_commands_enabled']}")
            print(f"   • File Operations: {registry_status['file_operations_enabled']}")
            print(f"   • Total Tools: {registry_status['total_tools']}")
            print(f"   • Available Tools: {', '.join(registry_status['tool_names'])}")
            
            return 'continue'
        else:
            print("❌ Tool logger not available")
            return 'continue'
    except Exception as e:
        print(f"❌ Error getting tool statistics: {e}")
        return 'continue'

# ===== PHASE 14: RECURSIVE EXECUTION CLI HANDLERS =====

def handle_write_and_run_command(parts: List[str]) -> str:
    """Handle /write_and_run "<code>" [filename] command"""
    try:
        if len(parts) < 1:
            print("❌ Usage: write_and_run \"<code>\" [filename]")
            print("   Example: write_and_run \"print('Hello World')\" hello.py")
            return 'continue'
        
        # Parse code and filename
        if len(parts) == 1:
            # Code provided without filename
            code = parts[0].strip('"\'')
            filename = "generated_script.py"
        else:
            # Code and filename provided
            code = parts[0].strip('"\'')
            filename = parts[1]
        
        # Write and execute
        result = write_and_execute(code, filename, "cli_user")
        
        print(f"\n📝 **Write and Execute Results**")
        print(f"📄 **File:** {result.get('filepath', filename)}")
        print(f"✅ **Write Success:** {result['write_result']['success']}")
        print(f"🚀 **Execute Success:** {result['execution_result']['success']}")
        
        if result['execution_result']['success']:
            print(f"\n📤 **Output:**")
            print(result['execution_result'].get('output', 'No output'))
        else:
            print(f"\n❌ **Error:**")
            print(result['execution_result'].get('error', 'Unknown error'))
        
        if result['execution_result'].get('duration'):
            print(f"⏱️ **Duration:** {result['execution_result']['duration']:.2f}s")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error in write_and_run: {e}")
        return 'continue'

def handle_recursive_run_command(filename: str) -> str:
    """Handle /recursive_run <filename> command"""
    try:
        import asyncio
        from pathlib import Path
        
        # Check if file exists
        file_path = Path(filename)
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            print("💡 Use 'list_generated_files' to see available files")
            return 'continue'
        
        # Read file content
        with open(file_path, 'r') as f:
            code = f.read()
        
        print(f"\n🔄 **Starting Recursive Improvement Loop**")
        print(f"📄 **File:** {filename}")
        print(f"⏳ **This may take several attempts to perfect the code...**\n")
        
        # Run the edit loop
        edit_loop = get_edit_loop()
        result = asyncio.run(edit_loop.run_loop(
            initial_code=code,
            filename=file_path.name,
            goal_description=f"Improve and fix the code in {filename}"
        ))
        
        print(f"\n📊 **Recursive Improvement Results**")
        print(f"✅ **Success:** {result.success}")
        print(f"🔄 **Total Attempts:** {result.total_attempts}")
        print(f"⏱️ **Duration:** {result.duration:.2f}s")
        print(f"🏁 **Termination:** {result.termination_reason}")
        
        if result.success and result.winning_file_path:
            print(f"🏆 **Winning File:** {result.winning_file_path}")
            print(f"📈 **Final Confidence:** {result.final_attempt.analysis_metrics.confidence_score:.2f}")
        
        if result.all_attempts:
            print(f"\n📝 **Attempt Summary:**")
            for i, attempt in enumerate(result.all_attempts, 1):
                status = "✅" if attempt.analysis_metrics.overall_success else "❌"
                print(f"   {i}. {status} Confidence: {attempt.analysis_metrics.confidence_score:.2f}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error in recursive_run: {e}")
        return 'continue'

def handle_list_generated_files_command() -> str:
    """Handle /list_generated_files command"""
    try:
        file_writer = get_file_writer()
        result = file_writer.list_generated_files()
        
        if not result['success']:
            print(f"❌ Error listing files: {result.get('error')}")
            return 'continue'
        
        files = result['files']
        
        print(f"\n📁 **Generated Files** ({result['count']} files)")
        print(f"📂 **Directory:** {result['directory']}")
        print("-" * 80)
        
        if not files:
            print("No generated files found.")
            print("💡 Use 'write_and_run' to create executable scripts")
            return 'continue'
        
        print(f"{'Name':<30} {'Size':<10} {'Executable':<10} {'Modified':<20}")
        print("-" * 80)
        
        for file_info in files:
            name = file_info['name']
            size = f"{file_info['size']} B"
            executable = "✅" if file_info['executable'] else "❌"
            modified = file_info['modified'][:19]  # Truncate timestamp
            
            print(f"{name:<30} {size:<10} {executable:<10} {modified:<20}")
        
        print(f"\n💡 Use 'show_file <name>' to view file contents")
        print(f"💡 Use 'recursive_run <name>' to improve a file")
        print(f"💡 Use 'delete_file <name>' to remove a file")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error listing generated files: {e}")
        return 'continue'

def handle_show_file_command(filename: str) -> str:
    """Handle /show_file <filename> command"""
    try:
        from pathlib import Path
        
        file_writer = get_file_writer()
        
        # Try to find file in write directory
        file_path = file_writer.write_dir / filename
        
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            print("💡 Use 'list_generated_files' to see available files")
            return 'continue'
        
        # Read and display file
        with open(file_path, 'r') as f:
            content = f.read()
        
        file_stats = file_path.stat()
        
        print(f"\n📄 **File: {filename}**")
        print(f"📍 **Path:** {file_path}")
        print(f"📏 **Size:** {file_stats.st_size} bytes")
        print(f"🗓️ **Modified:** {file_stats.st_mtime}")
        print(f"🔒 **Executable:** {'Yes' if file_stats.st_mode & 0o111 else 'No'}")
        print("-" * 60)
        
        # Display content with line numbers
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}│ {line}")
        
        print("-" * 60)
        print(f"📝 **Total Lines:** {len(lines)}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error showing file: {e}")
        return 'continue'

def handle_delete_file_command(filename: str) -> str:
    """Handle /delete_file <filename> command"""
    try:
        file_writer = get_file_writer()
        result = file_writer.delete_file(filename)
        
        if result['success']:
            print(f"✅ File deleted: {filename}")
            print(f"📍 Path: {result['deleted_file']}")
        else:
            print(f"❌ Failed to delete file: {result.get('error')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error deleting file: {e}")
        return 'continue'


# === PHASE 15: SELF-UPGRADING COMMANDS ===

def handle_self_upgrade_command(component: str = None) -> str:
    """Handle /self_upgrade [component] command"""
    try:
        from agents.registry import get_agent_registry
        
        registry = get_agent_registry()
        upgrade_manager = registry.get_agent('upgrade_manager')
        
        if not upgrade_manager:
            print("❌ UpgradeManager not available")
            return 'continue'
        
        print("🔄 Starting self-upgrade process...")
        
        if component:
            # Target specific component
            target_path = component
            print(f"🎯 Targeting component: {component}")
        else:
            # Full codebase scan
            target_path = None
            print("🌐 Scanning entire codebase")
        
        # Trigger scan and queue upgrades
        scan_result = upgrade_manager.trigger_scan(target_path)
        
        if scan_result.get("success"):
            print(f"✅ Scan completed successfully")
            print(f"📊 Analysis Summary:")
            print(f"   • Total suggestions: {scan_result['total_suggestions']}")
            print(f"   • Queued upgrades: {scan_result['queued_upgrades']}")
            
            if scan_result.get("analysis_summary"):
                summary = scan_result["analysis_summary"]
                print(f"   • Modules analyzed: {summary.get('analyzable_modules', 0)}")
                print(f"   • Circular imports: {summary.get('circular_imports', 0)}")
                print(f"   • Architectural violations: {summary.get('architectural_violations', 0)}")
        else:
            print(f"❌ Scan failed: {scan_result.get('error')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error during self-upgrade: {e}")
        return 'continue'


def handle_upgrade_status_command() -> str:
    """Handle /upgrade_status command"""
    try:
        from agents.registry import get_agent_registry
        
        registry = get_agent_registry()
        upgrade_manager = registry.get_agent('upgrade_manager')
        
        if not upgrade_manager:
            print("❌ UpgradeManager not available")
            return 'continue'
        
        status = upgrade_manager.get_upgrade_status()
        
        print("\n🎯 **Upgrade Status Overview**")
        print("=" * 50)
        
        # Queue status
        print(f"📋 **Queue Status:**")
        print(f"   • Pending upgrades: {status['queue_length']}")
        print(f"   • Currently processing: {status['in_progress']}")
        print(f"   • Completed today: {status['completed_today']}")
        
        # Scan information
        if status.get('last_scan'):
            print(f"🔍 **Last Scan:** {status['last_scan']}")
        if status.get('next_auto_scan'):
            print(f"⏰ **Next Auto Scan:** {status['next_auto_scan']}")
        
        # Pending tasks
        if status['pending_tasks']:
            print(f"\n📋 **Pending Upgrades ({len(status['pending_tasks'])}):**")
            for task in status['pending_tasks']:
                print(f"   • {task['id']}: {task['title']} (Priority: {task['priority']})")
        
        # In-progress tasks
        if status['in_progress_tasks']:
            print(f"\n⚙️  **In Progress ({len(status['in_progress_tasks'])}):**")
            for task in status['in_progress_tasks']:
                print(f"   • {task['id']}: {task['title']}")
        
        # Recent completed
        if status['recent_completed']:
            print(f"\n✅ **Recently Completed ({len(status['recent_completed'])}):**")
            for task in status['recent_completed'][-5:]:  # Show last 5
                status_icon = "✅" if task['status'] == 'completed' else "❌"
                print(f"   {status_icon} {task['id']}: {task['title']}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error getting upgrade status: {e}")
        return 'continue'


def handle_show_upgrade_diff_command(upgrade_id: str) -> str:
    """Handle /show_upgrade_diff <upgrade_id> command"""
    try:
        from agents.registry import get_agent_registry
        
        registry = get_agent_registry()
        upgrade_manager = registry.get_agent('upgrade_manager')
        
        if not upgrade_manager:
            print("❌ UpgradeManager not available")
            return 'continue'
        
        if not upgrade_id:
            print("❌ Usage: /show_upgrade_diff <upgrade_id>")
            return 'continue'
        
        diff_result = upgrade_manager.get_upgrade_diff(upgrade_id)
        
        if not diff_result.get("success"):
            print(f"❌ {diff_result.get('error')}")
            return 'continue'
        
        diff_info = diff_result["diff"]
        
        print(f"\n📄 **Upgrade Diff: {upgrade_id}**")
        print("=" * 60)
        print(f"🔧 **Type:** {diff_info['upgrade_type']}")
        print(f"📁 **Affected Modules:** {', '.join(diff_info['affected_modules'])}")
        
        print(f"\n📝 **Changes ({len(diff_info['changes'])}):**")
        
        for i, change in enumerate(diff_info['changes'], 1):
            print(f"\n{i}. **{change['type'].upper()}:** {change['path']}")
            print(f"   📏 Lines: {change['lines_added']}")
            
            if 'diff' in change:
                diff = change['diff']
                print(f"   📊 Changes: +{diff.get('lines_added', 0)} -{diff.get('lines_changed', 0)} lines")
            elif 'diff_error' in change:
                print(f"   ⚠️  Diff calculation error: {change['diff_error']}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error showing upgrade diff: {e}")
        return 'continue'


def handle_rollback_upgrade_command(upgrade_id: str) -> str:
    """Handle /rollback_upgrade <upgrade_id> command"""
    try:
        from agents.registry import get_agent_registry
        
        registry = get_agent_registry()
        upgrade_manager = registry.get_agent('upgrade_manager')
        
        if not upgrade_manager:
            print("❌ UpgradeManager not available")
            return 'continue'
        
        if not upgrade_id:
            print("❌ Usage: /rollback_upgrade <upgrade_id>")
            return 'continue'
        
        print(f"🔄 Rolling back upgrade: {upgrade_id}")
        
        rollback_result = upgrade_manager.rollback_upgrade(upgrade_id)
        
        if rollback_result.get("success"):
            print(f"✅ Rollback completed successfully")
            print(f"📁 Restored files:")
            for file_path in rollback_result.get("restored_files", []):
                print(f"   • {file_path}")
            print(f"⏰ Rollback time: {rollback_result['rollback_timestamp']}")
        else:
            print(f"❌ Rollback failed: {rollback_result.get('error')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error during rollback: {e}")
        return 'continue'


def handle_upgrade_ledger_command() -> str:
    """Handle /upgrade_ledger command"""
    try:
        from storage.upgrade_ledger import UpgradeLedger
        
        ledger = UpgradeLedger()
        
        print("\n📊 **Upgrade Ledger Overview**")
        print("=" * 50)
        
        # Get statistics
        stats = ledger.get_upgrade_statistics()
        
        print(f"📈 **Overall Statistics:**")
        print(f"   • Total upgrades: {stats['total_upgrades']}")
        print(f"   • Successful: {stats['successful_upgrades']}")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Total rollbacks: {stats['total_rollbacks']}")
        print(f"   • Recent upgrades (30d): {stats['recent_upgrades_30d']}")
        
        # By type statistics
        if stats.get('by_type'):
            print(f"\n📊 **By Upgrade Type:**")
            for upgrade_type, type_stats in stats['by_type'].items():
                print(f"   • {upgrade_type}: {type_stats['successful']}/{type_stats['total']} ({type_stats['success_rate']:.1%})")
        
        # File change statistics
        print(f"\n📁 **File Changes:**")
        print(f"   • Total changes: {stats['total_file_changes']}")
        print(f"   • Unique files affected: {stats['unique_files_changed']}")
        
        # Recent history
        print(f"\n📜 **Recent Upgrade History (Last 10):**")
        history = ledger.get_upgrade_history(limit=10)
        
        if not history:
            print("   No upgrades recorded")
        else:
            for upgrade in history:
                status_icon = "✅" if upgrade['success'] else "❌"
                print(f"   {status_icon} {upgrade['upgrade_id']}: {upgrade['upgrade_type']} ({upgrade['started_at']})")
        
        # Failure analysis
        failure_analysis = ledger.analyze_failure_patterns()
        if failure_analysis['total_failures'] > 0:
            print(f"\n⚠️  **Failure Analysis:**")
            print(f"   • Total failures: {failure_analysis['total_failures']}")
            
            if failure_analysis['common_errors']:
                print(f"   • Common errors:")
                for error_type, count in list(failure_analysis['common_errors'].items())[:3]:
                    print(f"     - {error_type}: {count}")
            
            if failure_analysis['recommendations']:
                print(f"   • Recommendations:")
                for rec in failure_analysis['recommendations'][:3]:
                    print(f"     - {rec}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error accessing upgrade ledger: {e}")
        return 'continue'


# === Phase 16: Autonomous Planning & Decision Layer Commands ===

def handle_auto_plan_command(goal: str) -> str:
    """Handle /auto_plan <goal> command"""
    try:
        from agents.registry import get_agent_registry

        registry = get_agent_registry()
        decision_planner = registry.get_agent('decision_planner')

        if not decision_planner:
            print("❌ DecisionPlanner not available")
            return 'continue'

        if not goal:
            print("❌ Usage: /auto_plan <goal>")
            return 'continue'

        print(f"🎯 **Autonomous Planning for Goal:** {goal}")
        print("=" * 60)

        # Generate and select best strategy
        best_strategy = decision_planner.select_best_strategy(goal)

        if not best_strategy:
            print("❌ Could not generate viable strategies for this goal")
            return 'continue'

        print(f"✅ **Selected Strategy:** {best_strategy.strategy_id}")
        print(f"🎯 **Goal:** {best_strategy.goal_text}")
        print(f"📊 **Score:** {best_strategy.score:.3f}")
        print(f"📈 **Success Probability:** {best_strategy.success_probability:.1%}")
        print(f"⏱️  **Estimated Completion:** {best_strategy.estimated_completion}")

        # Display plan details
        print(f"\n📋 **Plan Steps ({len(best_strategy.plan.steps)}):**")
        for i, step in enumerate(best_strategy.plan.steps, 1):
            status_icon = "⏳" if step.status == "pending" else "✅" if step.status == "completed" else "❌"
            print(f"   {i}. {status_icon} {step.subgoal}")
            print(f"      💡 Why: {step.why}")
            print(f"      🎯 Expected: {step.expected_result}")
            if step.prerequisites:
                print(f"      📋 Prerequisites: {', '.join(step.prerequisites)}")

        # Display enhanced features
        if hasattr(best_strategy, 'nodes') and best_strategy.nodes:
            print(f"\n🔗 **Enhanced Features:**")
            
            # Count special features
            next_steps = sum(1 for node in best_strategy.nodes if node.step.next_step)
            fallbacks = sum(1 for node in best_strategy.nodes if node.step.fallback_step)
            conditionals = sum(1 for node in best_strategy.nodes if node.step.conditional_logic)
            
            if next_steps > 0:
                print(f"   🔗 Sequential chaining: {next_steps} step(s)")
            if fallbacks > 0:
                print(f"   🔄 Fallback options: {fallbacks} step(s)")
            if conditionals > 0:
                print(f"   🤔 Conditional logic: {conditionals} step(s)")

        # Display strategy metadata
        if best_strategy.scores_breakdown:
            print(f"\n📊 **Scoring Breakdown:**")
            for criterion, score in best_strategy.scores_breakdown.items():
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {criterion.replace('_', ' ').title()}: {bar} {score:.3f}")

        print(f"\n💡 **Recommendation:** Use `/next_action` to start execution")
        print(f"📈 **Monitor:** Use `/plan_tree {best_strategy.strategy_id}` to visualize progress")

        return 'continue'

    except Exception as e:
        print(f"❌ Error in auto planning: {e}")
        return 'continue'


def handle_plan_tree_command(plan_id: str) -> str:
    """Handle /plan_tree <id> command"""
    try:
        from agents.registry import get_agent_registry

        registry = get_agent_registry()
        decision_planner = registry.get_agent('decision_planner')

        if not decision_planner:
            print("❌ DecisionPlanner not available")
            return 'continue'

        if not plan_id:
            print("❌ Usage: /plan_tree <plan_id>")
            return 'continue'

        # Get strategy visualization
        visualization = decision_planner.visualize_strategy(plan_id)

        if "error" in visualization:
            print(f"❌ {visualization['error']}")
            return 'continue'

        print(f"\n🌳 **Plan Tree Visualization: {plan_id}**")
        print("=" * 60)
        print(f"🎯 **Goal:** {visualization['goal']}")
        print(f"📊 **Type:** {visualization['type']}")
        print(f"⭐ **Score:** {visualization['score']:.3f}")

        # Display statistics
        stats = visualization['stats']
        print(f"\n📈 **Progress Statistics:**")
        print(f"   📋 Total steps: {stats['total_steps']}")
        print(f"   ✅ Completed: {stats['completed_steps']}")
        print(f"   ❌ Failed: {stats['failed_steps']}")
        print(f"   ⏱️  Estimated duration: {stats['estimated_duration']} minutes")
        
        progress_percentage = (stats['completed_steps'] / stats['total_steps'] * 100) if stats['total_steps'] > 0 else 0
        progress_bar_length = int(progress_percentage / 5)  # 20 char bar
        progress_bar = "█" * progress_bar_length + "░" * (20 - progress_bar_length)
        print(f"   📊 Progress: {progress_bar} {progress_percentage:.1f}%")

        # Display nodes in tree format
        print(f"\n🌳 **Plan Tree Structure:**")
        nodes = visualization['nodes']
        edges = visualization['edges']
        
        # Build adjacency for tree display
        children = {}
        roots = set(node['id'] for node in nodes)
        
        for edge in edges:
            if edge['type'] == 'dependency':
                parent = edge['from']
                child = edge['to']
                if parent not in children:
                    children[parent] = []
                children[parent].append(child)
                roots.discard(child)
        
        # Display tree structure
        def print_tree(node_id, level=0, prefix=""):
            node = next((n for n in nodes if n['id'] == node_id), None)
            if not node:
                return
            
            # Status icon
            status_icons = {
                'completed': '✅',
                'failed': '❌',
                'in_progress': '⚙️',
                'pending': '⏳'
            }
            status_icon = status_icons.get(node['status'], '❓')
            
            # Priority indicator
            priority_indicator = "🔥" if node['priority'] > 7 else "⭐" if node['priority'] > 4 else "📝"
            
            indent = "    " * level
            print(f"{indent}{prefix}{status_icon} {priority_indicator} {node['label']}")
            print(f"{indent}    💪 Confidence: {node['confidence']:.2f} | Priority: {node['priority']}")
            
            # Show dependencies
            deps = [edge['from'] for edge in edges if edge['to'] == node_id and edge['type'] == 'dependency']
            if deps:
                print(f"{indent}    📋 Depends on: {', '.join(deps)}")
            
            # Show special relationships
            fallbacks = [edge['to'] for edge in edges if edge['from'] == node_id and edge['type'] == 'fallback']
            if fallbacks:
                print(f"{indent}    🔄 Fallback: {', '.join(fallbacks)}")
            
            conditionals = [edge['to'] for edge in edges if edge['from'] == node_id and edge['type'] == 'branch']
            if conditionals:
                print(f"{indent}    🤔 Branches: {', '.join(conditionals)}")
            
            # Recurse to children
            child_nodes = children.get(node_id, [])
            for i, child_id in enumerate(child_nodes):
                is_last = i == len(child_nodes) - 1
                child_prefix = "└── " if is_last else "├── "
                print_tree(child_id, level + 1, child_prefix)
        
        # Print all root nodes
        if not roots:
            roots = {nodes[0]['id']} if nodes else set()
        
        for root_id in sorted(roots):
            print_tree(root_id)

        # Show execution recommendations
        print(f"\n💡 **Next Steps:**")
        ready_nodes = [node for node in nodes if node['status'] == 'pending']
        if ready_nodes:
            next_node = max(ready_nodes, key=lambda n: n['priority'])
            print(f"   🎯 Execute: {next_node['label']} (ID: {next_node['id']})")
            print(f"   ⚡ Use: `/next_action` to get detailed execution plan")
        else:
            completed = sum(1 for node in nodes if node['status'] == 'completed')
            if completed == len(nodes):
                print(f"   🎉 Plan completed successfully!")
            else:
                print(f"   ⚠️  No ready steps - check for blockers or failures")

        return 'continue'

    except Exception as e:
        print(f"❌ Error visualizing plan tree: {e}")
        return 'continue'


def handle_next_action_command() -> str:
    """Handle /next_action command"""
    try:
        from agents.registry import get_agent_registry

        registry = get_agent_registry()
        decision_planner = registry.get_agent('decision_planner')

        if not decision_planner:
            print("❌ DecisionPlanner not available")
            return 'continue'

        # Get the best next action across all active strategies
        next_action = decision_planner.get_next_action()

        if not next_action:
            print("🤷 **No Actions Available**")
            print("=" * 40)
            print("No executable actions found in active strategies.")
            print("")
            print("💡 **Suggestions:**")
            print("   • Create a new plan with `/auto_plan <goal>`")
            print("   • Check strategy status with `/plan_tree <id>`")
            print("   • Review active strategies for blockers")
            return 'continue'

        print(f"⚡ **Next Action Recommended**")
        print("=" * 50)
        print(f"🎯 **Goal:** {next_action['goal']}")
        print(f"🆔 **Strategy:** {next_action['strategy_id']}")
        print(f"📝 **Step:** {next_action['step_id']}")
        print("")
        print(f"🎬 **Action:** {next_action['action']}")
        print(f"💡 **Why:** {next_action['why']}")
        print(f"🎯 **Expected Result:** {next_action['expected_result']}")
        print("")
        print(f"📊 **Metadata:**")
        print(f"   💪 Confidence: {next_action['confidence']:.1%}")
        print(f"   ⭐ Priority: {next_action['priority']}/10")
        print(f"   ⏱️  Estimated Duration: {next_action.get('estimated_duration', 'N/A')} minutes")

        # Show execution context
        if 'strategy_score' in next_action:
            print(f"   🏆 Strategy Score: {next_action['strategy_score']:.3f}")

        print(f"\n🚀 **Ready to Execute!**")
        print(f"💡 **Next Steps:**")
        print(f"   1. Review the action details above")
        print(f"   2. Execute the action: {next_action['action']}")
        print(f"   3. Update progress when complete")
        print(f"   4. Run `/next_action` again for the next step")

        print(f"\n📈 **Monitoring:**")
        print(f"   • View full plan: `/plan_tree {next_action['strategy_id']}`")
        print(f"   • If action fails: `/reroute {next_action['goal']}`")

        return 'continue'

    except Exception as e:
        print(f"❌ Error getting next action: {e}")
        return 'continue'


def handle_reroute_command(failed_goal: str) -> str:
    """Handle /reroute <failed_goal> command"""
    try:
        from agents.registry import get_agent_registry

        registry = get_agent_registry()
        decision_planner = registry.get_agent('decision_planner')

        if not decision_planner:
            print("❌ DecisionPlanner not available")
            return 'continue'

        if not failed_goal:
            print("❌ Usage: /reroute <failed_goal>")
            print("Example: /reroute 'Implement user authentication'")
            return 'continue'

        print(f"🔄 **Re-routing Failed Goal:** {failed_goal}")
        print("=" * 60)

        # Analyze the failure and generate new strategy
        failure_context = {
            "goal": failed_goal,
            "failure_time": datetime.now().isoformat(),
            "requested_by": "user",
            "context": "Manual re-routing requested"
        }

        # Generate new strategy
        new_strategy = decision_planner.adaptive_replan(None, failure_context)

        if not new_strategy:
            print("❌ Could not generate alternative strategy")
            print("")
            print("💡 **Suggestions:**")
            print("   • Check if the goal is clearly defined")
            print("   • Try breaking down the goal into smaller parts")
            print("   • Use `/auto_plan <simpler_goal>` for individual components")
            return 'continue'

        print(f"✅ **New Strategy Generated!**")
        print(f"🆔 **Strategy ID:** {new_strategy.strategy_id}")
        print(f"⭐ **Score:** {new_strategy.score:.3f}")
        print(f"📈 **Success Probability:** {new_strategy.success_probability:.1%}")

        # Show what changed
        print(f"\n🔄 **Strategy Improvements:**")
        if hasattr(new_strategy, 'plan') and new_strategy.plan:
            print(f"   📋 Steps: {len(new_strategy.plan.steps)}")
            print(f"   🏗️  Type: {new_strategy.plan.plan_type}")
            
            # Show risk factors addressed
            if new_strategy.plan.risk_factors:
                print(f"   ⚠️  Risk factors identified: {len(new_strategy.plan.risk_factors)}")
                for risk in new_strategy.plan.risk_factors[:3]:  # Show top 3
                    print(f"      • {risk}")

        # Display new plan overview
        print(f"\n📋 **New Plan Overview:**")
        if hasattr(new_strategy, 'plan') and new_strategy.plan.steps:
            for i, step in enumerate(new_strategy.plan.steps[:5], 1):  # Show first 5 steps
                status_icon = "⏳" if step.status == "pending" else "✅"
                print(f"   {i}. {status_icon} {step.subgoal}")
                if step.confidence < 0.7:
                    print(f"      ⚠️  Lower confidence: {step.confidence:.2f}")
            
            if len(new_strategy.plan.steps) > 5:
                print(f"   ... and {len(new_strategy.plan.steps) - 5} more steps")

        # Show enhanced failure mitigation
        if hasattr(new_strategy, 'nodes'):
            fallback_count = sum(1 for node in new_strategy.nodes if node.step.fallback_step)
            retry_enabled = sum(1 for node in new_strategy.nodes if node.max_retries > 0)
            
            if fallback_count > 0 or retry_enabled > 0:
                print(f"\n🛡️  **Failure Mitigation:**")
                if fallback_count > 0:
                    print(f"   🔄 Fallback options: {fallback_count} step(s)")
                if retry_enabled > 0:
                    print(f"   🔁 Retry logic: {retry_enabled} step(s)")

        print(f"\n🚀 **Ready to Execute!**")
        print(f"💡 **Next Steps:**")
        print(f"   1. Review the new strategy above")
        print(f"   2. Start execution: `/next_action`")
        print(f"   3. Monitor progress: `/plan_tree {new_strategy.strategy_id}`")

        # Log the re-routing for learning
        try:
            reflective_engine = registry.get_agent('reflective_engine')
            if reflective_engine:
                reflective_engine.log_failed_strategy(
                    strategy_id="manual_reroute",
                    goal_text=failed_goal,
                    failure_reason="User requested re-routing",
                    failure_context=failure_context
                )
        except Exception as re_error:
            # Don't fail the command if reflection logging fails
            pass

        return 'continue'

    except Exception as e:
        print(f"❌ Error re-routing goal: {e}")
        return 'continue'


def debate_claim(claim: str) -> str:
    """
    🎯 Start a full internal debate on a claim.
    
    Args:
        claim: The claim to debate (e.g., "AI should prioritize safety over performance")
        
    Returns:
        Debate results with arguments and conclusion
        
    Usage: /debate "AI development should prioritize safety over speed"
    """
    try:
        registry = get_agent_registry()
        debate_engine = registry.get_agent('debate_engine')
        
        if not debate_engine:
            return "❌ Debate engine not available. Ensure Phase 21 agents are loaded."
        
        print(f"🎯 **Starting Internal Debate**")
        print(f"📋 Claim: '{claim}'")
        print(f"⚡ Orchestrating multi-agent dialectical reasoning...")
        print()
        
        # Initiate debate
        result = debate_engine.initiate_debate(claim)
        
        # Format and display results
        print(f"🏁 **Debate Complete!**")
        print(f"📊 **Conclusion:** {result.conclusion}")
        print(f"🎖️  **Confidence:** {result.confidence:.2f}")
        print(f"🤝 **Consensus Reached:** {'Yes' if result.consensus_reached else 'No'}")
        print()
        
        if result.key_arguments:
            print(f"🔑 **Key Arguments:**")
            for i, arg in enumerate(result.key_arguments, 1):
                stance_emoji = "👍" if "pro" in arg.stance.value else "👎" if "con" in arg.stance.value else "🤔"
                print(f"   {i}. {stance_emoji} [{arg.stance.value.upper()}] {arg.content}")
                print(f"      💪 Strength: {arg.strength_score:.2f} | 🧠 Logic: {arg.logical_validity:.2f}")
            print()
        
        if result.contradictions_resolved:
            print(f"✅ **Contradictions Resolved:**")
            for contradiction in result.contradictions_resolved:
                print(f"   • {contradiction}")
            print()
        
        if result.open_questions:
            print(f"❓ **Open Questions:**")
            for question in result.open_questions:
                print(f"   • {question}")
            print()
        
        if result.synthesis:
            print(f"🔄 **Synthesis:**")
            print(f"   {result.synthesis}")
            print()
        
        return f"✅ Debate on '{claim}' completed with conclusion: {result.conclusion}"
        
    except Exception as e:
        return f"❌ Debate failed: {str(e)}"


def critique_plan(plan_or_action: str) -> str:
    """
    🔍 Get critical review from CriticAgent.
    
    Args:
        plan_or_action: Plan or action to critique
        
    Returns:
        Critical analysis with identified issues
        
    Usage: /critique "Deploy AI model to production without testing"
    """
    try:
        registry = get_agent_registry()
        critic_agent = registry.get_agent('critic')
        
        if not critic_agent:
            return "❌ Critic agent not available. Ensure agents are loaded."
        
        print(f"🔍 **Initiating Critical Analysis**")
        print(f"📋 Target: '{plan_or_action}'")
        print(f"🎯 Analyzing for flaws, risks, and logical issues...")
        print()
        
        # Get critical analysis
        criticism = critic_agent.respond(plan_or_action)
        
        # Analyze logical flaws
        logical_flaws = critic_agent.analyze_logical_flaws(plan_or_action)
        
        # Challenge assumptions
        assumptions = critic_agent.challenge_assumptions(plan_or_action)
        
        print(f"📝 **Critical Analysis:**")
        print(criticism)
        print()
        
        if logical_flaws:
            print(f"🧠 **Logical Issues Identified:**")
            for flaw in logical_flaws:
                severity_emoji = "🔴" if flaw['severity'] == 'high' else "🟡" if flaw['severity'] == 'medium' else "🟢"
                print(f"   {severity_emoji} {flaw['type'].replace('_', ' ').title()}")
                print(f"      📍 {flaw['description']}")
            print()
        
        if assumptions:
            print(f"🤔 **Assumptions Challenged:**")
            for assumption in assumptions:
                print(f"   ❓ {assumption['type'].replace('_', ' ').title()}")
                print(f"      📝 {assumption['description']}")
                if 'challenge' in assumption:
                    print(f"      💭 Challenge: {assumption['challenge']}")
            print()
        
        return f"✅ Critical analysis complete. Found {len(logical_flaws)} logical issues and challenged {len(assumptions)} assumptions."
        
    except Exception as e:
        return f"❌ Critique failed: {str(e)}"


def simulate_debate(topic: str) -> str:
    """
    ⚔️ Force opposing stances in a debate (e.g., Plan A vs Plan B).
    
    Args:
        topic: Topic with opposing options (use " vs " to separate)
        
    Returns:
        Debate simulation results
        
    Usage: /simulate_debate "Plan A: Gradual rollout vs Plan B: Immediate deployment"
    """
    try:
        registry = get_agent_registry()
        debate_engine = registry.get_agent('debate_engine')
        
        if not debate_engine:
            return "❌ Debate engine not available."
        
        # Parse opposing sides
        if " vs " in topic:
            sides = topic.split(" vs ", 1)
            side_a = sides[0].strip()
            side_b = sides[1].strip()
        else:
            side_a = f"Support: {topic}"
            side_b = f"Oppose: {topic}"
        
        print(f"⚔️  **Simulating Opposing Debate**")
        print(f"🔵 Side A: {side_a}")
        print(f"🔴 Side B: {side_b}")
        print(f"⚡ Forcing systematic opposition...")
        print()
        
        # Create debate context that forces opposition
        context = {
            'simulation_mode': True,
            'forced_opposition': True,
            'side_a': side_a,
            'side_b': side_b
        }
        
        # Formulate claim that will generate opposition
        claim = f"'{side_a}' is the better approach compared to '{side_b}'"
        
        result = debate_engine.initiate_debate(claim, context)
        
        print(f"🏁 **Debate Simulation Complete!**")
        print(f"📊 **Final Assessment:** {result.conclusion}")
        print(f"🎯 **Confidence:** {result.confidence:.2f}")
        
        return f"✅ Debate simulation complete: {result.conclusion}"
        
    except Exception as e:
        return f"❌ Debate simulation failed: {str(e)}"


def reflect_on_memory(memory_id: str) -> str:
    """
    🤔 Trigger debate and reflection on a past belief or decision.
    
    Args:
        memory_id: ID of memory to reflect on
        
    Returns:
        Reflection results with insights
        
    Usage: /reflect_on "belief_123" or /reflect_on "decision_456"
    """
    try:
        registry = get_agent_registry()
        reflection_orchestrator = registry.get_agent('reflection_orchestrator')
        
        if not reflection_orchestrator:
            return "❌ Reflection orchestrator not available."
        
        print(f"🤔 **Initiating Reflective Analysis**")
        print(f"🧠 Memory ID: {memory_id}")
        print()
        
        # Get memory context
        memory_context = {
            'memory_id': memory_id,
            'reflection_type': 'memory_analysis',
            'trigger_source': 'user_request'
        }
        
        # Trigger reflection
        from agents.reflection_orchestrator import ReflectionTrigger
        result = reflection_orchestrator.trigger_reflection(
            f"Analysis of memory {memory_id}",
            ReflectionTrigger.FEEDBACK_ANALYSIS,
            memory_context
        )
        
        print(f"📋 **Reflection Results:**")
        print(f"🎯 **Debate Triggered:** {'Yes' if result.debate_triggered else 'No'}")
        
        if result.insights:
            print(f"\n💡 **Insights Discovered:**")
            for i, insight in enumerate(result.insights, 1):
                print(f"   {i}. {insight}")
        
        return f"✅ Reflection on {memory_id} completed. Resolved: {result.resolved}"
        
    except Exception as e:
        return f"❌ Reflection failed: {str(e)}"


def consolidate_memory(mode: str = "incremental") -> str:
    """
    🧠 Run autonomous memory consolidation.
    
    Args:
        mode: Consolidation mode ("incremental", "full", "prune", "cluster")
        
    Returns:
        Consolidation results
        
    Usage: /consolidate_memory full
    """
    try:
        from storage.memory_consolidator import MemoryConsolidator
        from storage.memory_log import MemoryLog
        
        print(f"🧠 **Starting Memory Consolidation**")
        print(f"🔧 Mode: {mode}")
        print(f"⚡ Analyzing memory system...")
        print()
        
        # Initialize consolidator
        memory_log = MemoryLog()
        consolidator = MemoryConsolidator(memory_log)
        
        if mode == "full":
            print(f"🚀 **Running Full Consolidation**")
            result = consolidator.consolidate_memory(full_consolidation=True)
            
        elif mode == "prune":
            print(f"🧹 **Running Memory Pruning**")
            prune_result = consolidator.prune()
            return f"✅ Pruned {prune_result.get('pruned_facts', 0)} low-priority facts"
            
        elif mode == "cluster":
            print(f"🔗 **Running Memory Clustering**")
            clusters = consolidator.cluster()
            return f"✅ Created {len(clusters)} memory clusters"
            
        else:  # incremental
            print(f"⚡ **Running Incremental Consolidation**")
            result = consolidator.consolidate_memory(full_consolidation=False)
        
        # Display results
        print(f"📊 **Consolidation Results:**")
        print(f"   📝 Facts processed: {result.facts_processed}")
        print(f"   🗑️  Facts pruned: {result.facts_pruned}")
        print(f"   🔄 Facts consolidated: {result.facts_consolidated}")
        print(f"   🔗 Clusters created: {result.clusters_created}")
        print(f"   ⏱️  Duration: {(result.completed_at - result.started_at).total_seconds():.2f}s")
        print()
        
        if result.statistics:
            stats = result.statistics
            print(f"📈 **Memory Statistics:**")
            print(f"   📊 Total facts: {stats.get('total_facts', 0)}")
            print(f"   🛡️  Protected facts: {stats.get('protected_facts', 0)}")
            print(f"   📊 Average confidence: {stats.get('avg_confidence', 0):.3f}")
            
            confidence_dist = stats.get('confidence_distribution', {})
            if confidence_dist:
                print(f"   🎯 High confidence: {confidence_dist.get('high', 0)}")
                print(f"   📊 Medium confidence: {confidence_dist.get('medium', 0)}")
                print(f"   ⚠️  Low confidence: {confidence_dist.get('low', 0)}")
            
            print(f"   📅 Temporal span: {stats.get('temporal_span_days', 0)} days")
        
        if result.warnings:
            print(f"\n⚠️  **Warnings:**")
            for warning in result.warnings[:3]:
                print(f"   • {warning}")
        
        if result.errors:
            print(f"\n❌ **Errors:**")
            for error in result.errors[:3]:
                print(f"   • {error}")
        
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        return f"{status} Memory consolidation completed: {result.facts_processed} processed, {result.facts_consolidated} consolidated"
        
    except Exception as e:
        return f"❌ Memory consolidation failed: {str(e)}"


def list_clusters(cluster_type: str = None) -> str:
    """
    🔗 List memory clusters with their details.
    
    Args:
        cluster_type: Filter by cluster type ("semantic", "temporal", "causal")
        
    Returns:
        List of memory clusters
        
    Usage: /list_clusters semantic
    """
    try:
        from storage.memory_consolidator import MemoryConsolidator
        from storage.memory_log import MemoryLog
        
        print(f"🔗 **Memory Clusters**")
        if cluster_type:
            print(f"📋 Filter: {cluster_type}")
        print()
        
        # Initialize consolidator
        memory_log = MemoryLog()
        consolidator = MemoryConsolidator(memory_log)
        
        # Get clusters
        clusters = consolidator.list_clusters(cluster_type)
        
        if not clusters:
            print(f"📭 No clusters found")
            if cluster_type:
                print(f"💡 Try without the '{cluster_type}' filter")
            else:
                print(f"💡 Run '/consolidate_memory cluster' to create clusters")
            return "ℹ️  No memory clusters found"
        
        print(f"📊 **Found {len(clusters)} clusters:**")
        print()
        
        for i, cluster in enumerate(clusters, 1):
            print(f"🔗 **Cluster {i}: {cluster['cluster_id']}**")
            print(f"   📋 Type: {cluster['cluster_type']}")
            print(f"   📊 Facts: {cluster['fact_count']}")
            print(f"   🎯 Score: {cluster['consolidation_score']:.3f}")
            print(f"   📅 Created: {cluster['created_at'][:19]}")
            print(f"   🔄 Updated: {cluster['last_updated'][:19]}")
            
            print(f"   📝 Sample facts:")
            for j, fact in enumerate(cluster['sample_facts'], 1):
                print(f"      {j}. {fact}")
            
            print()
        
        return f"✅ Listed {len(clusters)} memory clusters"
        
    except Exception as e:
        return f"❌ Failed to list clusters: {str(e)}"


def prioritize_memory(limit: int = 20) -> str:
    """
    🎯 Show prioritized memories based on scoring factors.
    
    Args:
        limit: Number of top memories to show
        
    Returns:
        List of prioritized memories
        
    Usage: /prioritize_memory 10
    """
    try:
        from storage.memory_consolidator import MemoryConsolidator
        from storage.memory_log import MemoryLog
        
        print(f"🎯 **Memory Prioritization**")
        print(f"📊 Showing top {limit} memories")
        print()
        
        # Initialize consolidator
        memory_log = MemoryLog()
        consolidator = MemoryConsolidator(memory_log)
        
        # Get prioritized facts
        prioritized_facts = consolidator.prioritize()
        
        if not prioritized_facts:
            return "📭 No facts found for prioritization"
        
        # Show top facts
        top_facts = prioritized_facts[:limit]
        
        print(f"🏆 **Top {len(top_facts)} Priority Memories:**")
        print()
        
        for i, fact in enumerate(top_facts, 1):
            priority_score = getattr(fact, 'consolidation_metadata', {}).get('priority_score', 0)
            protection_level = getattr(fact, 'consolidation_metadata', {}).get('protection_level', 'none')
            
            # Priority emoji based on score
            if priority_score > 0.8:
                priority_emoji = "🔴"
            elif priority_score > 0.6:
                priority_emoji = "🟡"
            else:
                priority_emoji = "🟢"
            
            # Protection emoji
            protection_emoji = {
                'permanent': '🛡️',
                'protected': '🔒',
                'soft': '🔓',
                'none': ''
            }.get(str(protection_level).split('.')[-1].lower(), '')
            
            print(f"{priority_emoji} **{i}. {fact.subject} {fact.predicate} {fact.object}**")
            print(f"   🎯 Priority: {priority_score:.3f} {protection_emoji}")
            print(f"   📊 Confidence: {fact.confidence:.3f}")
            print(f"   📅 Date: {fact.timestamp[:19]}")
            print()
        
        # Show distribution
        total_facts = len(prioritized_facts)
        high_priority = len([f for f in prioritized_facts if getattr(f, 'consolidation_metadata', {}).get('priority_score', 0) > 0.7])
        medium_priority = len([f for f in prioritized_facts if 0.3 <= getattr(f, 'consolidation_metadata', {}).get('priority_score', 0) <= 0.7])
        low_priority = total_facts - high_priority - medium_priority
        
        print(f"📊 **Priority Distribution ({total_facts} total):**")
        print(f"   🔴 High priority: {high_priority}")
        print(f"   🟡 Medium priority: {medium_priority}")
        print(f"   🟢 Low priority: {low_priority}")
        
        return f"✅ Prioritized {total_facts} memories, showing top {len(top_facts)}"
        
    except Exception as e:
        return f"❌ Memory prioritization failed: {str(e)}"


def memory_stats() -> str:
    """
    📊 Display comprehensive memory system statistics.
    
    Returns:
        Memory system statistics
        
    Usage: /memory_stats
    """
    try:
        from storage.memory_consolidator import MemoryConsolidator
        from storage.memory_log import MemoryLog
        
        print(f"📊 **Memory System Statistics**")
        print(f"=" * 50)
        
        # Initialize systems
        memory_log = MemoryLog()
        consolidator = MemoryConsolidator(memory_log)
        
        # Get consolidation statistics
        stats = consolidator.get_consolidation_statistics()
        
        print(f"🧠 **Memory Overview:**")
        print(f"   📝 Total facts: {stats.get('total_facts', 0)}")
        print(f"   🛡️  Protected facts: {stats.get('protected_facts', 0)}")
        print(f"   🔗 Active clusters: {stats.get('active_clusters', 0)}")
        print(f"   📊 Average confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"   📅 Temporal span: {stats.get('temporal_span_days', 0)} days")
        print()
        
        # Confidence distribution
        confidence_dist = stats.get('confidence_distribution', {})
        if confidence_dist:
            print(f"📊 **Confidence Distribution:**")
            total = sum(confidence_dist.values())
            for level, count in confidence_dist.items():
                percentage = (count / total * 100) if total > 0 else 0
                emoji = {'high': '🎯', 'medium': '📊', 'low': '⚠️'}.get(level, '📋')
                print(f"   {emoji} {level.title()}: {count} ({percentage:.1f}%)")
            print()
        
        # Cluster information
        clusters = consolidator.list_clusters()
        if clusters:
            print(f"🔗 **Cluster Statistics:**")
            cluster_types = {}
            total_facts_in_clusters = 0
            
            for cluster in clusters:
                cluster_type = cluster['cluster_type']
                cluster_types[cluster_type] = cluster_types.get(cluster_type, 0) + 1
                total_facts_in_clusters += cluster['fact_count']
            
            for cluster_type, count in cluster_types.items():
                print(f"   📋 {cluster_type.title()}: {count} clusters")
            
            print(f"   🔗 Facts in clusters: {total_facts_in_clusters}")
            print()
        
        # Consolidation rules
        rules_applied = stats.get('consolidation_rules_applied', 0)
        print(f"⚙️  **Consolidation Configuration:**")
        print(f"   📋 Active rules: {rules_applied}")
        print(f"   🧹 Pruning: {'Enabled' if consolidator.config.get('pruning', {}).get('enabled') else 'Disabled'}")
        print(f"   🔗 Clustering: {'Enabled' if consolidator.config.get('clustering', {}).get('enabled') else 'Disabled'}")
        print(f"   🔄 Consolidation: {'Enabled' if consolidator.config.get('consolidation', {}).get('enabled') else 'Disabled'}")
        print()
        
        # Recent consolidation history
        if consolidator.consolidation_history:
            recent_consolidation = consolidator.consolidation_history[-1]
            print(f"📈 **Recent Consolidation:**")
            print(f"   📅 Last run: {recent_consolidation.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   📊 Type: {recent_consolidation.operation_type}")
            print(f"   ✅ Success: {'Yes' if recent_consolidation.success else 'No'}")
            print(f"   📝 Processed: {recent_consolidation.facts_processed}")
            print(f"   🔄 Consolidated: {recent_consolidation.facts_consolidated}")
        
        return "✅ Memory statistics displayed"
        
    except Exception as e:
        return f"❌ Failed to get memory statistics: {str(e)}"


# ===== PHASE 22: AGENT REPLICATION COMMAND HANDLERS =====

def fork_agent_command(agent_name: str) -> str:
    """Fork an agent to create a mutable copy"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        fork_id = replicator.fork_agent(agent_name)
        
        if fork_id:
            return f"✅ Successfully forked {agent_name}\n" \
                   f"🔀 Fork ID: {fork_id[:8]}\n" \
                   f"📁 Location: agent_forks/{fork_id}/\n" \
                   f"🔧 Use '/mutate_agent {fork_id[:8]}' to apply mutations"
        else:
            return f"❌ Failed to fork {agent_name}. Check if agent exists and fork limit not exceeded."
            
    except Exception as e:
        return f"❌ Error forking agent: {str(e)}"


def mutate_agent_command(fork_id: str) -> str:
    """Apply mutations to a forked agent"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        
        # Find full fork ID if only partial provided
        full_fork_id = None
        for active_fork_id in replicator.active_forks:
            if active_fork_id.startswith(fork_id):
                full_fork_id = active_fork_id
                break
        
        if not full_fork_id:
            return f"❌ Fork not found: {fork_id}"
        
        fork_info = replicator.active_forks[full_fork_id]
        success = replicator.mutate_agent(fork_info['fork_file'])
        
        if success:
            mutations = fork_info['mutations']
            return f"✅ Successfully mutated fork {fork_id}\n" \
                   f"🧬 Total mutations: {mutations}\n" \
                   f"🧪 Use '/run_fork {fork_id}' to test the mutated agent"
        else:
            return f"❌ Failed to mutate fork {fork_id}"
            
    except Exception as e:
        return f"❌ Error mutating agent: {str(e)}"


def run_fork_command(fork_id: str) -> str:
    """Test and evaluate a forked agent"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        
        # Find full fork ID if only partial provided
        full_fork_id = None
        for active_fork_id in replicator.active_forks:
            if active_fork_id.startswith(fork_id):
                full_fork_id = active_fork_id
                break
        
        if not full_fork_id:
            return f"❌ Fork not found: {fork_id}"
        
        fork_info = replicator.active_forks[full_fork_id]
        agent_name = f"{fork_info['agent_name']}_{full_fork_id[:8]}"
        
        results = replicator.test_agent(agent_name)
        
        if 'error' in results:
            return f"❌ Testing failed: {results['error']}"
        
        survival_threshold = replicator.survival_threshold
        score = results['overall_score']
        status_icon = "✅" if score >= survival_threshold else "❌"
        status_text = "SURVIVOR" if score >= survival_threshold else "CANDIDATE FOR PRUNING"
        
        return f"🧪 Test Results for {agent_name}:\n" \
               f"📝 Syntax Valid: {'✅' if results['syntax_valid'] else '❌'}\n" \
               f"⚙️  Functionality Score: {results['functionality_score']:.2f}\n" \
               f"📊 Overall Score: {score:.2f}\n" \
               f"🎯 Threshold: {survival_threshold}\n" \
               f"{status_icon} Status: {status_text}"
        
    except Exception as e:
        return f"❌ Error running fork: {str(e)}"


def score_forks_command() -> str:
    """Evaluate performance of all forks"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        results = replicator.evaluate_performance(str(replicator.fork_logs_path))
        
        if 'error' in results:
            return f"❌ Evaluation failed: {results['error']}"
        
        response = f"📊 Fork Performance Evaluation:\n"
        response += f"📝 Total log entries: {results['total_log_entries']}\n"
        response += f"🧪 Tested forks: {results['tested_forks']}\n\n"
        
        if results['ranked_performance']:
            response += f"🏆 Performance Rankings:\n"
            for i, (fork_id, score) in enumerate(results['ranked_performance'][:5]):
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                response += f"{rank_emoji} {fork_id[:8]}: {score:.3f}\n"
            
            if results['top_performer']:
                top_id, top_score = results['top_performer']
                response += f"\n🎯 Top Performer: {top_id[:8]} ({top_score:.3f})"
        else:
            response += "No tested forks found."
        
        return response
        
    except Exception as e:
        return f"❌ Error scoring forks: {str(e)}"


def prune_forks_command() -> str:
    """Remove low-performing forks"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        results = replicator.prune_forks()
        
        response = f"🗑️ Fork Pruning Results:\n"
        response += f"✂️ Forks pruned: {results['pruned_count']}\n"
        response += f"📁 Remaining forks: {results['remaining_forks']}\n"
        
        if results['errors']:
            response += f"\n⚠️ Errors ({len(results['errors'])}):\n"
            for error in results['errors'][:3]:
                response += f"  • {error}\n"
        else:
            response += f"\n✅ All pruning operations completed successfully"
        
        return response
        
    except Exception as e:
        return f"❌ Error pruning forks: {str(e)}"


def fork_status_command() -> str:
    """Display status of all active forks"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        
        if not replicator.active_forks:
            return f"📂 No active forks.\n💡 Use '/fork_agent <agent_name>' to create one."
        
        response = f"📂 Active Forks ({len(replicator.active_forks)}/{replicator.max_forks}):\n\n"
        
        for fork_id, fork_info in replicator.active_forks.items():
            response += f"🔀 {fork_id[:8]} ({fork_info['agent_name']})\n"
            response += f"   📊 Status: {fork_info['status']}\n"
            response += f"   🧬 Mutations: {fork_info['mutations']}\n"
            
            if fork_info.get('tested', False):
                score = fork_info.get('score', 0.0)
                threshold = replicator.survival_threshold
                status_icon = "✅" if score >= threshold else "❌"
                response += f"   🎯 Score: {score:.2f} {status_icon}\n"
            else:
                response += f"   🎯 Score: Not tested\n"
            
            response += "\n"
        
        response += f"💡 Commands: /mutate_agent, /run_fork, /score_forks, /prune_forks, /replication_cycle"
        
        return response
        
    except Exception as e:
        return f"❌ Error getting fork status: {str(e)}"


def replication_cycle_command() -> str:
    """Run an automated replication cycle"""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        orchestrator = registry.get_agent('reflection_orchestrator')
        if not orchestrator:
            return "❌ Reflection orchestrator not available"
        
        if not hasattr(orchestrator, 'process_automated_replication_cycle'):
            return "❌ Automated replication cycle not supported"
        
        results = orchestrator.process_automated_replication_cycle()
        
        response = "🔄 **Automated Replication Cycle Results**\n\n"
        response += f"Status: {results['status']}\n"
        response += f"Replications triggered: {results['replications_triggered']}\n"
        response += f"Tests completed: {results['tests_completed']}\n"
        response += f"Prunes performed: {results['prunes_performed']}\n\n"
        
        if results['actions']:
            response += "Actions taken:\n"
            for action in results['actions']:
                response += f"• {action}\n"
        else:
            response += "No actions required at this time.\n"
        
        if results['status'] == 'error':
            response += f"\n❌ Error: {results.get('error', 'Unknown error')}"
        
        return response
        
    except Exception as e:
        return f"❌ Error running replication cycle: {str(e)}"


def tune_replication_command(parameter: str, value: str) -> str:
    """Tune replication parameters"""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        replicator = registry.get_agent('agent_replicator')
        orchestrator = registry.get_agent('reflection_orchestrator')
        
        if not replicator:
            return "❌ Agent replicator not available"
        
        # Convert value to appropriate type
        try:
            if '.' in value:
                numeric_value = float(value)
            else:
                numeric_value = int(value)
        except ValueError:
            return f"❌ Invalid numeric value: {value}"
        
        response = f"🔧 **Replication Tuning**\n\n"
        
        # Handle different parameters
        if parameter == 'mutation_rate':
            if 0.0 <= numeric_value <= 1.0:
                old_value = replicator.mutation_rate
                replicator.mutation_rate = numeric_value
                response += f"Mutation rate: {old_value:.3f} → {numeric_value:.3f}\n"
            else:
                return "❌ Mutation rate must be between 0.0 and 1.0"
        
        elif parameter == 'survival_threshold':
            if 0.0 <= numeric_value <= 1.0:
                old_value = replicator.survival_threshold
                replicator.survival_threshold = numeric_value
                response += f"Survival threshold: {old_value:.3f} → {numeric_value:.3f}\n"
            else:
                return "❌ Survival threshold must be between 0.0 and 1.0"
        
        elif parameter == 'max_forks':
            if 1 <= numeric_value <= 50:
                old_value = replicator.max_forks
                replicator.max_forks = int(numeric_value)
                response += f"Max forks: {old_value} → {int(numeric_value)}\n"
            else:
                return "❌ Max forks must be between 1 and 50"
        
        elif parameter == 'contradiction_threshold' and orchestrator:
            if 1 <= numeric_value <= 20:
                old_value = orchestrator.replication_contradiction_threshold
                orchestrator.replication_contradiction_threshold = int(numeric_value)
                response += f"Contradiction threshold: {old_value} → {int(numeric_value)}\n"
            else:
                return "❌ Contradiction threshold must be between 1 and 20"
        
        elif parameter == 'uncertainty_threshold' and orchestrator:
            if 0.0 <= numeric_value <= 1.0:
                old_value = orchestrator.replication_uncertainty_threshold
                orchestrator.replication_uncertainty_threshold = numeric_value
                response += f"Uncertainty threshold: {old_value:.3f} → {numeric_value:.3f}\n"
            else:
                return "❌ Uncertainty threshold must be between 0.0 and 1.0"
        
        else:
            available_params = ['mutation_rate', 'survival_threshold', 'max_forks', 'contradiction_threshold', 'uncertainty_threshold']
            return f"❌ Unknown parameter: {parameter}\nAvailable: {', '.join(available_params)}"
        
        response += "\n✅ Parameter updated successfully!"
        return response
        
    except Exception as e:
        return f"❌ Error tuning replication: {str(e)}"


def promote_agent_command(agent_name: str) -> str:
    """Promote an agent by boosting its contract confidence values"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        result = replicator.promote_agent(agent_name, "Manual promotion via CLI")
        
        if result.get('success'):
            original = result.get('original_confidence', {})
            boosted = result.get('confidence_boost', {})
            
            response = f"✅ Successfully promoted agent {agent_name}!\n\n"
            response += "📈 Confidence Updates:\n"
            
            for capability, new_confidence in boosted.items():
                original_val = original.get(capability, 0.0)
                improvement = new_confidence - original_val
                response += f"   • {capability}: {original_val:.3f} → {new_confidence:.3f} (+{improvement:.3f})\n"
            
            response += f"\n🎯 Reason: {result.get('reason', 'Manual promotion')}"
            return response
        else:
            return f"❌ Failed to promote agent {agent_name}: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error promoting agent: {str(e)}"


def retire_agent_command(agent_name: str) -> str:
    """Retire an agent by marking it as inactive"""
    try:
        from agents.self_replicator import get_agent_replicator
        
        replicator = get_agent_replicator()
        result = replicator.retire_agent(agent_name, "Manual retirement via CLI")
        
        if result.get('success'):
            response = f"🏁 Successfully retired agent {agent_name}!\n\n"
            response += f"📝 Reason: {result.get('reason', 'Manual retirement')}\n"
            
            if result.get('archived'):
                response += "📁 Contract archived for future reference\n"
            
            response += "\n⚠️  Agent is now inactive but preserved for potential reactivation"
            return response
        else:
            return f"❌ Failed to retire agent {agent_name}: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error retiring agent: {str(e)}"


def lifecycle_status_command(agent_name: str) -> str:
    """Show detailed lifecycle status for an agent"""
    try:
        from agents.registry import get_agent_registry
        from agents.agent_lifecycle import AgentLifecycleManager
        
        registry = get_agent_registry()
        agent = registry.get_agent(agent_name)
        
        if not agent:
            return f"❌ Agent {agent_name} not found in registry"
        
        # Get lifecycle metrics
        metrics = agent.get_lifecycle_metrics() if hasattr(agent, 'get_lifecycle_metrics') else {}
        
        # Initialize lifecycle manager to evaluate drift
        lifecycle_manager = AgentLifecycleManager()
        drift_score = lifecycle_manager.evaluate_drift(agent)
        
        response = f"📊 Lifecycle Status for {agent_name}\n"
        response += "=" * 50 + "\n\n"
        
        # Performance Metrics
        response += "🎯 Performance Metrics:\n"
        response += f"   • Execution Count: {metrics.get('execution_count', 0)}\n"
        response += f"   • Success Rate: {metrics.get('success_rate', 0.0):.3f}\n"
        response += f"   • Current Performance: {metrics.get('current_performance', 0.0):.3f}\n"
        response += f"   • Confidence Variance: {metrics.get('confidence_variance', 0.0):.3f}\n"
        
        # Drift Analysis
        response += f"\n🔄 Drift Analysis:\n"
        response += f"   • Drift Score: {drift_score:.3f}\n"
        response += f"   • Contract Alignment: {(1.0 - drift_score):.3f}\n"
        
        # Lifecycle Events
        response += f"\n📅 Lifecycle History:\n"
        if hasattr(agent, 'last_promotion') and agent.last_promotion:
            response += f"   • Last Promotion: {agent.last_promotion.strftime('%Y-%m-%d %H:%M')}\n"
        else:
            response += f"   • Last Promotion: Never\n"
            
        if hasattr(agent, 'last_mutation') and agent.last_mutation:
            response += f"   • Last Mutation: {agent.last_mutation.strftime('%Y-%m-%d %H:%M')}\n"
        else:
            response += f"   • Last Mutation: Never\n"
        
        response += f"   • History Length: {metrics.get('lifecycle_history_length', 0)} events\n"
        
        # Contract Info
        if agent.contract:
            response += f"\n📋 Contract Information:\n"
            response += f"   • Version: {agent.contract.version}\n"
            response += f"   • Purpose: {agent.contract.purpose[:60]}...\n"
            response += f"   • Capabilities: {len(agent.contract.capabilities)} listed\n"
            response += f"   • Confidence Vector: {len(agent.contract.confidence_vector)} values\n"
        
        # Recommendations
        response += f"\n💡 Recommendations:\n"
        if lifecycle_manager.should_promote(agent):
            response += "   🎉 PROMOTE: High performance detected!\n"
        elif lifecycle_manager.should_retire(agent):
            response += "   🏁 RETIRE: Poor performance or misalignment detected\n"
        elif lifecycle_manager.should_mutate(agent):
            response += "   🔄 MUTATE: Drift or performance issues detected\n"
        else:
            response += "   ✅ MONITOR: Performance within acceptable ranges\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error getting lifecycle status: {str(e)}"


def evaluate_lifecycle_command(agent_name: str) -> str:
    """Evaluate an agent and apply appropriate lifecycle decision"""
    try:
        from agents.registry import get_agent_registry
        from agents.agent_lifecycle import AgentLifecycleManager
        
        registry = get_agent_registry()
        agent = registry.get_agent(agent_name)
        
        if not agent:
            return f"❌ Agent {agent_name} not found in registry"
        
        # Initialize lifecycle manager and make decision
        lifecycle_manager = AgentLifecycleManager()
        decision = lifecycle_manager.apply_lifecycle_decision(agent)
        
        response = f"🔍 Lifecycle Evaluation for {agent_name}\n"
        response += "=" * 50 + "\n\n"
        
        response += f"📋 Decision: {decision.action.value.upper()}\n"
        response += f"💭 Reason: {decision.reason}\n"
        response += f"🎯 Confidence: {decision.confidence:.3f}\n"
        response += f"⏰ Timestamp: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if decision.metrics:
            response += f"\n📊 Key Metrics:\n"
            response += f"   • Drift Score: {decision.metrics.drift_score:.3f}\n"
            response += f"   • Success Rate: {decision.metrics.success_rate:.3f}\n"
            response += f"   • Execution Count: {decision.metrics.execution_count}\n"
        
        # Apply the decision automatically if it's not just monitoring
        if decision.action.value != 'monitor':
            from agents.self_replicator import get_agent_replicator
            replicator = get_agent_replicator()
            
            if decision.action.value == 'promote':
                result = replicator.promote_agent(agent_name, decision.reason)
                if result.get('success'):
                    response += f"\n✅ Promotion applied successfully!"
                else:
                    response += f"\n❌ Promotion failed: {result.get('error')}"
                    
            elif decision.action.value == 'mutate':
                result = replicator.mutate_agent(agent_name, decision.reason)
                if result.get('success'):
                    response += f"\n✅ Mutation applied successfully! Fork ID: {result.get('fork_id', '')[:8]}"
                else:
                    response += f"\n❌ Mutation failed: {result.get('error')}"
                    
            elif decision.action.value == 'retire':
                result = replicator.retire_agent(agent_name, decision.reason)
                if result.get('success'):
                    response += f"\n✅ Retirement applied successfully!"
                else:
                    response += f"\n❌ Retirement failed: {result.get('error')}"
        
        return response
        
    except Exception as e:
        return f"❌ Error evaluating lifecycle: {str(e)}"


def list_agents_with_drift_command() -> str:
    """List all agents with their drift scores"""
    try:
        from agents.registry import get_agent_registry
        from agents.agent_lifecycle import AgentLifecycleManager
        
        registry = get_agent_registry()
        lifecycle_manager = AgentLifecycleManager()
        
        agents = registry.get_all_agents()
        
        if not agents:
            return "❌ No agents found in registry"
        
        response = "🔄 Agent Drift Analysis\n"
        response += "=" * 60 + "\n\n"
        
        drift_data = []
        
        for agent_name, agent in agents.items():
            try:
                if agent and hasattr(agent, 'get_lifecycle_metrics'):
                    drift_score = lifecycle_manager.evaluate_drift(agent)
                    metrics = agent.get_lifecycle_metrics()
                    
                    drift_data.append({
                        'name': agent_name,
                        'drift': drift_score,
                        'success_rate': metrics.get('success_rate', 0.0),
                        'executions': metrics.get('execution_count', 0),
                        'should_promote': lifecycle_manager.should_promote(agent),
                        'should_mutate': lifecycle_manager.should_mutate(agent),
                        'should_retire': lifecycle_manager.should_retire(agent)
                    })
            except Exception as e:
                drift_data.append({
                    'name': agent_name,
                    'drift': 0.5,
                    'success_rate': 0.0,
                    'executions': 0,
                    'error': str(e)
                })
        
        # Sort by drift score (highest first)
        drift_data.sort(key=lambda x: x.get('drift', 0.5), reverse=True)
        
        response += f"{'Agent':<20} {'Drift':<8} {'Success':<8} {'Execs':<8} {'Recommendation':<15}\n"
        response += "-" * 70 + "\n"
        
        for data in drift_data:
            name = data['name'][:19]
            drift = f"{data.get('drift', 0.5):.3f}"
            success = f"{data.get('success_rate', 0.0):.3f}"
            execs = str(data.get('executions', 0))
            
            # Determine recommendation
            if data.get('error'):
                recommendation = "ERROR"
            elif data.get('should_retire'):
                recommendation = "🏁 RETIRE"
            elif data.get('should_promote'):
                recommendation = "🎉 PROMOTE"
            elif data.get('should_mutate'):
                recommendation = "🔄 MUTATE"
            else:
                recommendation = "✅ MONITOR"
            
            response += f"{name:<20} {drift:<8} {success:<8} {execs:<8} {recommendation:<15}\n"
        
        # Summary statistics
        valid_drifts = [d['drift'] for d in drift_data if 'error' not in d]
        if valid_drifts:
            avg_drift = sum(valid_drifts) / len(valid_drifts)
            high_drift_count = len([d for d in valid_drifts if d > lifecycle_manager.drift_threshold])
            
            response += f"\n📊 Summary:\n"
            response += f"   • Total Agents: {len(drift_data)}\n"
            response += f"   • Average Drift: {avg_drift:.3f}\n"
            response += f"   • High Drift (>{lifecycle_manager.drift_threshold}): {high_drift_count}\n"
            response += f"   • Promotion Candidates: {len([d for d in drift_data if d.get('should_promote')])}\n"
            response += f"   • Mutation Candidates: {len([d for d in drift_data if d.get('should_mutate')])}\n"
            response += f"   • Retirement Candidates: {len([d for d in drift_data if d.get('should_retire')])}\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error listing agents with drift: {str(e)}"


# === PHASE 26: COGNITIVE DISSONANCE COMMANDS ===

def dissonance_report_command() -> str:
    """Show current dissonance stress regions and scores"""
    try:
        from agents.dissonance_tracker import get_dissonance_tracker
        
        tracker = get_dissonance_tracker()
        report = tracker.get_dissonance_report()
        
        response = "🧠 Cognitive Dissonance Report\n"
        response += "=" * 60 + "\n\n"
        
        # System summary
        summary = report['summary']
        response += f"📊 System Overview:\n"
        response += f"   • Total Dissonance Regions: {summary['total_regions']}\n"
        response += f"   • Total Pressure Score: {summary['total_pressure']:.3f}\n"
        response += f"   • Average Pressure: {summary['average_pressure']:.3f}\n"
        response += f"   • High Urgency Regions: {summary['high_urgency_regions']}\n"
        response += f"   • Recent Activity (1h): {summary['recent_events']} events\n\n"
        
        # Top stress regions
        if report['top_stress_regions']:
            response += f"🔥 Top Stress Regions:\n\n"
            for i, region in enumerate(report['top_stress_regions'][:5], 1):
                urgency_icon = "🔥" if region['urgency'] > 0.8 else "⚠️" if region['urgency'] > 0.5 else "📊"
                response += f"{i}. {urgency_icon} {region['belief_id']}\n"
                response += f"   • Cluster: {region['semantic_cluster']}\n"
                response += f"   • Pressure: {region['pressure_score']:.3f} | Urgency: {region['urgency']:.3f}\n"
                response += f"   • Confidence Erosion: {region['confidence_erosion']:.3f}\n"
                response += f"   • Emotional Volatility: {region['emotional_volatility']:.3f}\n"
                response += f"   • Duration: {region['duration_hours']:.1f}h | Contradictions: {region['contradiction_count']}\n"
                response += f"   • Conflict Sources: {', '.join(region['conflict_sources'])}\n\n"
        else:
            response += "✅ No significant dissonance regions detected\n\n"
        
        # Recent events
        if report['recent_events']:
            response += f"⏰ Recent Events (Last Hour):\n"
            for event in report['recent_events'][-5:]:  # Last 5 events
                timestamp = event['timestamp'].split('T')[1][:8]  # HH:MM:SS
                response += f"   • {timestamp} - {event['event_type']} in {event['belief_id']} (intensity: {event['intensity']:.3f})\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error generating dissonance report: {str(e)}"


def resolve_dissonance_command(belief_id: str = None, force_evolution: bool = False) -> str:
    """Trigger resolution for dissonance - either specific belief or highest pressure"""
    try:
        from agents.dissonance_tracker import get_dissonance_tracker
        
        tracker = get_dissonance_tracker()
        
        # Parse force_evolution if it's a string
        if isinstance(force_evolution, str):
            force_evolution = force_evolution.lower() in ['true', 'yes', '1', 'force']
        
        result = tracker.resolve_dissonance(belief_id=belief_id, force_evolution=force_evolution)
        
        response = "🧠 Dissonance Resolution\n"
        response += "=" * 60 + "\n\n"
        
        if result['status'] == 'no_dissonance':
            response += "✅ No dissonance regions found to resolve\n"
            return response
        elif result['status'] == 'error':
            response += f"❌ Error: {result['message']}\n"
            return response
        
        # Resolution attempt results
        response += f"🎯 Target: {result['belief_id']}\n"
        response += f"📊 Pressure Before: {result['pressure_before']:.3f}\n"
        response += f"📉 Pressure After: {result['pressure_after']:.3f}\n"
        response += f"📈 Reduction: {((result['pressure_before'] - result['pressure_after']) / result['pressure_before'] * 100):.1f}%\n\n"
        
        response += "🔧 Resolution Attempts:\n"
        for i, attempt in enumerate(result['results'], 1):
            status_icon = "✅" if attempt['success'] else "❌"
            response += f"{i}. {status_icon} {attempt['type'].title()}: {attempt['message']}\n"
        
        if force_evolution:
            response += "\n⚠️ Note: Belief evolution was requested but may require memory system integration\n"
        
        response += "\n💡 Tip: Use 'dissonance_report' to monitor progress\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error resolving dissonance: {str(e)}"


def dissonance_history_command(hours: int = 24) -> str:
    """Show dissonance activity timeline and buildup/resolution patterns"""
    try:
        from agents.dissonance_tracker import get_dissonance_tracker
        
        tracker = get_dissonance_tracker()
        
        # Parse hours if it's a string
        if isinstance(hours, str):
            try:
                hours = int(hours)
            except ValueError:
                hours = 24
        
        history = tracker.get_dissonance_history(hours=hours)
        
        response = f"🧠 Dissonance History ({hours}h)\n"
        response += "=" * 60 + "\n\n"
        
        # Activity summary
        response += f"📊 Activity Summary:\n"
        response += f"   • Total Events: {history['total_events']}\n"
        response += f"   • Event Types: {', '.join([f'{k}: {v}' for k, v in history['event_counts'].items()])}\n\n"
        
        # Pressure trend analysis
        trend = history['pressure_trend']
        response += f"📈 Pressure Trend Analysis:\n"
        response += f"   • Current Pressure: {trend['current_pressure']:.3f}\n"
        response += f"   • Estimated Baseline: {trend['estimated_baseline']:.3f}\n"
        response += f"   • Pressure Change: {trend['pressure_change']:+.3f}\n"
        response += f"   • Contradiction Rate: {trend['contradiction_rate']:.2f}/hour\n\n"
        
        # Trend interpretation
        if trend['pressure_change'] > 0.5:
            response += "🔥 Status: HIGH PRESSURE BUILDUP - Consider immediate resolution\n"
        elif trend['pressure_change'] > 0.2:
            response += "⚠️ Status: MODERATE PRESSURE - Monitor closely\n"
        elif trend['pressure_change'] < -0.2:
            response += "📉 Status: PRESSURE DECREASING - Resolutions working\n"
        else:
            response += "✅ Status: STABLE PRESSURE - Normal operations\n"
        
        response += "\n"
        
        # Event timeline
        if history['events']:
            response += f"⏰ Event Timeline (Last {min(10, len(history['events']))} events):\n"
            
            # Group events by belief_id for better readability
            events_by_belief = {}
            for event in history['events']:
                belief_id = event['belief_id']
                if belief_id not in events_by_belief:
                    events_by_belief[belief_id] = []
                events_by_belief[belief_id].append(event)
            
            for belief_id, events in list(events_by_belief.items())[:5]:  # Top 5 beliefs
                response += f"\n🎯 {belief_id}:\n"
                for event in events[-3:]:  # Last 3 events per belief
                    timestamp = event['timestamp'].split('T')[1][:8]  # HH:MM:SS
                    event_icon = {
                        'contradiction_detected': '⚡',
                        'pressure_increase': '📈',
                        'resolution_attempt': '🔧',
                        'resolution': '✅'
                    }.get(event['event_type'], '📊')
                    
                    response += f"   {timestamp} {event_icon} {event['event_type']}"
                    if event.get('intensity'):
                        response += f" (intensity: {event['intensity']:.3f})"
                    if event.get('resolution_attempts', 0) > 0:
                        response += f" [attempt #{event['resolution_attempts']}]"
                    response += "\n"
        else:
            response += "✅ No events in the specified time window\n"
        
        response += f"\n💡 Use 'dissonance_history {hours*2}' to see longer timeline\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error retrieving dissonance history: {str(e)}"


# Phase 27 pt 2: Meta-Self Agent Commands

def handle_meta_status_command() -> str:
    """
    🧠 Show cognitive system health status from MetaSelfAgent.
    
    Returns:
        System health report with component scores and anomalies
        
    Usage: /meta_status
    """
    try:
        from agents.meta_self_agent import get_meta_self_agent
        
        print(f"🧠 **Cognitive System Health Status**")
        print(f"=" * 50)
        
        # Get MetaSelfAgent instance
        meta_agent = get_meta_self_agent()
        
        # Perform health check
        health_metrics = meta_agent.perform_health_check()
        
        # Generate and display health report
        health_report = meta_agent._generate_health_report(health_metrics)
        print(health_report)
        
        # Show system advice if needed
        if health_metrics.overall_health_score < 0.6:
            print(f"\n⚠️ **System Alert**: Cognitive health is below optimal levels.")
            print(f"Consider running /meta_reflect for deeper analysis or /meta_goals to see improvement actions.")
        
        elif health_metrics.critical_issues:
            print(f"\n🚨 **Critical Alert**: {len(health_metrics.critical_issues)} critical issues detected.")
            print(f"Immediate attention recommended.")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ MetaSelfAgent not available - ensure Phase 27 pt 2 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error showing meta status: {str(e)}")
        return 'continue'


def handle_meta_reflect_command() -> str:
    """
    🔍 Trigger deep introspective analysis of the cognitive system.
    
    Returns:
        Deep analysis report with trends, patterns, and strategic insights
        
    Usage: /meta_reflect
    """
    try:
        from agents.meta_self_agent import get_meta_self_agent
        
        print(f"🔍 **Triggering Deep Cognitive Introspection**")
        print(f"=" * 50)
        
        # Get MetaSelfAgent instance
        meta_agent = get_meta_self_agent()
        
        # Trigger introspective analysis
        print(f"Analyzing cognitive patterns and system health...")
        
        introspection_report = meta_agent._generate_introspective_analysis()
        print(introspection_report)
        
        # Show follow-up recommendations
        print(f"\n💡 **Follow-up Actions**:")
        print(f"  • Check /meta_goals to see generated improvement goals")
        print(f"  • Monitor /meta_status regularly for health trends")
        print(f"  • Review specific component health if issues persist")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ MetaSelfAgent not available - ensure Phase 27 pt 2 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error during meta reflection: {str(e)}")
        return 'continue'


def handle_meta_goals_command(limit: int = 10) -> str:
    """
    🎯 Show recent meta-goals generated by the MetaSelfAgent.
    
    Args:
        limit: Maximum number of goals to display
        
    Returns:
        List of recent meta-goals with status and details
        
    Usage: /meta_goals [limit]
    """
    try:
        from agents.meta_self_agent import get_meta_self_agent
        from datetime import datetime
        
        print(f"🎯 **Meta-Goals Generated by Cognitive System**")
        print(f"=" * 50)
        
        # Get MetaSelfAgent instance
        meta_agent = get_meta_self_agent()
        
        # Get recent goals
        recent_goals = meta_agent.get_recent_meta_goals(limit)
        
        if not recent_goals:
            print(f"📭 No meta-goals have been generated yet.")
            print(f"\nThe MetaSelfAgent will generate goals automatically when:")
            print(f"  • System health drops below critical thresholds")
            print(f"  • Anomalies or critical issues are detected")
            print(f"  • Performance decline patterns are identified")
            return 'continue'
        
        print(f"📋 Showing {len(recent_goals)} most recent meta-goals:")
        print()
        
        # Group goals by status
        pending_goals = [g for g in recent_goals if not g.executed_at and not g.completed_at]
        in_progress_goals = [g for g in recent_goals if g.executed_at and not g.completed_at]
        completed_goals = [g for g in recent_goals if g.completed_at]
        
        # Display goals by status
        if pending_goals:
            print(f"⏳ **Pending Goals** ({len(pending_goals)}):")
            for goal in pending_goals[:5]:  # Show top 5 pending
                priority_emoji = "🔴" if goal.priority >= 0.8 else "🟡" if goal.priority >= 0.6 else "🟢"
                urgency_emoji = "⚡" if goal.urgency >= 0.8 else "🔵" if goal.urgency >= 0.6 else "⚪"
                
                print(f"  {priority_emoji} {urgency_emoji} **{goal.description}**")
                print(f"     Type: {goal.goal_type} | Target: {goal.target_component}")
                print(f"     Priority: {goal.priority:.2f} | Urgency: {goal.urgency:.2f}")
                print(f"     Created: {datetime.fromtimestamp(goal.created_at).strftime('%Y-%m-%d %H:%M')}")
                print(f"     Justification: {goal.justification}")
                print()
        
        if in_progress_goals:
            print(f"🔄 **In Progress Goals** ({len(in_progress_goals)}):")
            for goal in in_progress_goals[:3]:  # Show top 3 in progress
                print(f"  🔄 **{goal.description}**")
                print(f"     Type: {goal.goal_type} | Target: {goal.target_component}")
                print(f"     Started: {datetime.fromtimestamp(goal.executed_at).strftime('%Y-%m-%d %H:%M')}")
                print()
        
        if completed_goals:
            print(f"✅ **Recently Completed Goals** ({len(completed_goals)}):")
            for goal in completed_goals[:3]:  # Show top 3 completed
                duration = goal.completed_at - goal.created_at
                duration_hours = duration / 3600
                
                print(f"  ✅ **{goal.description}**")
                print(f"     Type: {goal.goal_type} | Target: {goal.target_component}")
                print(f"     Completed: {datetime.fromtimestamp(goal.completed_at).strftime('%Y-%m-%d %H:%M')}")
                print(f"     Duration: {duration_hours:.1f} hours")
                print()
        
        # Show goal effectiveness summary
        total_goals = len(meta_agent.generated_goals)
        completed_count = len([g for g in meta_agent.generated_goals.values() if g.completed_at])
        completion_rate = completed_count / total_goals if total_goals > 0 else 0
        
        print(f"📊 **Goal Effectiveness Summary**:")
        print(f"  • Total goals generated: {total_goals}")
        print(f"  • Completion rate: {completion_rate:.1%}")
        print(f"  • Active goals: {len(pending_goals) + len(in_progress_goals)}")
        
        if completion_rate > 0.7:
            print(f"  🎯 Goal completion rate is healthy")
        elif completion_rate < 0.3:
            print(f"  ⚠️ Low goal completion rate - may indicate execution issues")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ MetaSelfAgent not available - ensure Phase 27 pt 2 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error showing meta goals: {str(e)}")
        return 'continue'


# ===== PHASE 28: AUTONOMOUS PLANNING CLI HANDLERS =====

def handle_plan_status_command() -> str:
    """
    📋 Show status of autonomous action plans.
    
    Returns:
        Current planning status with active plans and progress
        
    Usage: /plan_status
    """
    try:
        from agents.action_planner import get_autonomous_planner
        
        print(f"📋 **Autonomous Planning Status**")
        print(f"=" * 50)
        
        # Get AutonomousPlanner instance
        planner = get_autonomous_planner()
        
        if not planner.enabled:
            print(f"⚠️ Autonomous planning is currently disabled")
            print(f"   Enable in config.yaml: autonomous_planner.enabled = true")
            return 'continue'
        
        # Generate status report
        status_report = planner._generate_plan_status_report()
        print(status_report)
        
        # Add planning cycle information
        if planner.last_planning_cycle:
            print(f"\n🕒 Last Planning Cycle: {planner.last_planning_cycle}")
        else:
            print(f"\n🕒 No planning cycles have run yet")
        
        # Show recent planning history
        if planner.planning_history:
            recent_cycle = planner.planning_history[-1]
            print(f"📊 Last Cycle Results:")
            print(f"   • Duration: {recent_cycle.get('cycle_duration', 0):.2f}s")
            print(f"   • System Health: {recent_cycle.get('system_health_score', 0):.2f}")
            print(f"   • Goals Enqueued: {recent_cycle.get('goals_enqueued', 0)}")
        
        # Show configuration
        print(f"\n⚙️ Configuration:")
        print(f"   • Planning Interval: {planner.planning_interval / 3600:.1f} hours")
        print(f"   • Max Active Plans: {planner.max_active_plans}")
        print(f"   • Min Plan Score: {planner.min_plan_score}")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ AutonomousPlanner not available - ensure Phase 28 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error showing plan status: {str(e)}")
        return 'continue'


def handle_plan_next_command() -> str:
    """
    ⏭️ Trigger execution of the next highest-priority plan step.
    
    Returns:
        Result of triggering the next plan step
        
    Usage: /plan_next
    """
    try:
        from agents.action_planner import get_autonomous_planner
        
        print(f"⏭️ **Triggering Next Plan Step**")
        print(f"=" * 40)
        
        # Get AutonomousPlanner instance
        planner = get_autonomous_planner()
        
        if not planner.enabled:
            print(f"⚠️ Autonomous planning is currently disabled")
            return 'continue'
        
        # Trigger next step
        result = planner._trigger_next_plan_step()
        print(result)
        
        return 'continue'
        
    except ImportError:
        print(f"❌ AutonomousPlanner not available - ensure Phase 28 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error triggering next plan step: {str(e)}")
        return 'continue'


def handle_plan_eval_command() -> str:
    """
    🔄 Trigger a full planning evaluation cycle.
    
    Returns:
        Results of the planning evaluation cycle
        
    Usage: /plan_eval
    """
    try:
        from agents.action_planner import get_autonomous_planner
        
        print(f"🔄 **Starting Planning Evaluation Cycle**")
        print(f"=" * 45)
        
        # Get AutonomousPlanner instance
        planner = get_autonomous_planner()
        
        if not planner.enabled:
            print(f"⚠️ Autonomous planning is currently disabled")
            return 'continue'
        
        print(f"🔍 Evaluating system state and updating plans...")
        
        # Trigger evaluation cycle
        result = planner._trigger_plan_evaluation()
        print(result)
        
        # Show updated status
        print(f"\n📊 Updated Planning Status:")
        status_summary = planner._generate_plan_status_report()
        # Show just the summary line
        lines = status_summary.split('\n')
        for line in lines[:5]:  # First 5 lines
            print(line)
        
        return 'continue'
        
    except ImportError:
        print(f"❌ AutonomousPlanner not available - ensure Phase 28 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error during plan evaluation: {str(e)}")
        return 'continue'


def handle_personality_evolve_command() -> str:
    """
    🧬 Force a personality evolution check and potentially trigger evolution.
    
    Returns:
        Command execution status
        
    Usage: /personality_evolve
    """
    try:
        from agents.personality_evolver import get_personality_evolver
        
        print(f"🧬 **Triggering Personality Evolution Check**")
        print(f"=" * 50)
        
        # Get PersonalityEvolver instance
        evolver = get_personality_evolver()
        
        print(f"🔍 Analyzing memory trends, contradiction stress, and belief changes...")
        
        # Force personality evolution
        trace = evolver.evolve_personality(trigger_type="manual", lookback_days=7)
        
        if trace:
            print(f"\n✅ **Personality Evolution Completed**")
            print(f"Evolution ID: {trace.evolution_id}")
            print(f"Trigger: {trace.trigger_type}")
            print(f"Total Changes: {trace.total_changes}")
            print(f"Evolution Magnitude: {trace.evolution_magnitude:.3f}")
            print(f"Duration: {trace.duration_ms:.1f}ms")
            
            print(f"\n📊 **Changes Applied**:")
            for change in trace.trait_changes:
                direction = "↗️" if change.new_value > change.old_value else "↘️" 
                print(f"  {direction} {change.trait_name.title()}: {change.old_value:.3f} → {change.new_value:.3f}")
                print(f"     Reason: {change.trigger_details}")
            
            print(f"\n📝 **Summary**: {trace.natural_language_summary}")
            
            if trace.trait_changes:
                print(f"\n💡 **Next Steps**:")
                print(f"  • Check /personality_status to see updated traits")
                print(f"  • Review /personality_history for evolution trends")
                print(f"  • Monitor system responses for personality changes")
            
        else:
            print(f"\n📋 **No Evolution Needed**")
            print(f"Current personality traits are stable and no significant")
            print(f"memory trends, contradiction stress, or belief changes detected.")
            
            # Show current status as consolation
            status = evolver.get_personality_status()
            print(f"\n📊 **Current Trait Summary**:")
            for trait, value in status['current_traits'].items():
                print(f"  • {trait.title()}: {value:.3f}")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ PersonalityEvolver not available - ensure Phase 29 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error during personality evolution: {str(e)}")
        return 'continue'


def handle_personality_history_command(limit: int = 10) -> str:
    """
    📚 Show personality evolution history with timestamps and causes.
    
    Args:
        limit: Maximum number of evolution traces to display
        
    Returns:
        Command execution status
        
    Usage: /personality_history [limit]
    """
    try:
        from agents.personality_evolver import get_personality_evolver
        from datetime import datetime
        
        print(f"📚 **Personality Evolution History**")
        print(f"=" * 50)
        
        # Get PersonalityEvolver instance
        evolver = get_personality_evolver()
        
        # Get evolution history
        history = evolver.get_evolution_history(limit)
        
        if not history:
            print(f"📭 No personality evolution history found.")
            print(f"\nPersonality evolution history will be recorded when:")
            print(f"  • Weekly automatic evolution checks occur")
            print(f"  • Major belief conflicts trigger evolution")
            print(f"  • Manual evolution is triggered with /personality_evolve")
            return 'continue'
        
        print(f"📋 Showing {len(history)} most recent evolution events:")
        print()
        
        for i, trace in enumerate(history, 1):
            evolution_id = trace.get('evolution_id', 'unknown')
            timestamp = trace.get('timestamp', 0)
            trigger_type = trace.get('trigger_type', 'unknown')
            total_changes = trace.get('total_changes', 0)
            magnitude = trace.get('evolution_magnitude', 0.0)
            summary = trace.get('natural_language_summary', 'No summary available')
            
            # Format timestamp
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = 'Unknown time'
            
            # Choose emoji based on trigger type
            trigger_emoji = {
                'weekly_cadence': '📅',
                'major_conflict': '⚡',
                'system_health': '🏥',
                'manual': '🎛️'
            }.get(trigger_type, '🔄')
            
            # Choose magnitude color
            if magnitude >= 0.3:
                magnitude_desc = "Major"
                magnitude_emoji = "🔴"
            elif magnitude >= 0.1:
                magnitude_desc = "Moderate" 
                magnitude_emoji = "🟡"
            else:
                magnitude_desc = "Minor"
                magnitude_emoji = "🟢"
            
            print(f"{i:2d}. {trigger_emoji} **Evolution {evolution_id}**")
            print(f"     📅 When: {time_str}")
            print(f"     🎯 Trigger: {trigger_type}")
            print(f"     {magnitude_emoji} Impact: {magnitude_desc} ({magnitude:.3f})")
            print(f"     🔧 Changes: {total_changes} traits modified")
            print(f"     📝 Summary: {summary}")
            
            # Show trait changes if available
            trait_changes = trace.get('trait_changes', [])
            if trait_changes and len(trait_changes) <= 3:  # Only show details for small changes
                print(f"     📊 Details:")
                for change in trait_changes:
                    if isinstance(change, dict):
                        trait_name = change.get('trait_name', 'unknown')
                        old_val = change.get('old_value', 0)
                        new_val = change.get('new_value', 0)
                        direction = "↗️" if new_val > old_val else "↘️"
                        print(f"        {direction} {trait_name.title()}: {old_val:.3f} → {new_val:.3f}")
            
            print()  # Empty line between entries
        
        # Show summary statistics
        total_evolutions = len(history)
        avg_magnitude = sum(trace.get('evolution_magnitude', 0) for trace in history) / max(1, total_evolutions)
        
        trigger_counts = {}
        for trace in history:
            trigger = trace.get('trigger_type', 'unknown')
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        print(f"📈 **Evolution Statistics**:")
        print(f"  • Total Evolutions: {total_evolutions}")
        print(f"  • Average Magnitude: {avg_magnitude:.3f}")
        print(f"  • Trigger Breakdown:")
        for trigger, count in sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_evolutions) * 100
            print(f"    - {trigger}: {count} ({percentage:.1f}%)")
        
        print(f"\n💡 **Tips**:")
        print(f"  • Use /personality_status to see current trait values")
        print(f"  • Use /personality_evolve to manually trigger evolution")
        print(f"  • Weekly automatic evolution helps maintain personality balance")
        
        return 'continue'
        
    except ImportError:
        print(f"❌ PersonalityEvolver not available - ensure Phase 29 is properly configured")
        return 'continue'
    except Exception as e:
        print(f"❌ Error retrieving personality history: {str(e)}")
        return 'continue'


# ===== PHASE 31: FAST REFLEX AGENT CLI HANDLERS =====

def handle_reflex_mode_command(mode: str) -> str:
    """Handle /reflex_mode <on|off> command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        reflex_agent = registry.get_agent('fast_reflex')
        
        if not reflex_agent:
            print("❌ Fast Reflex Agent not available")
            return 'continue'
        
        if mode in ['on', 'enable', 'true']:
            result = reflex_agent.toggle_reflex_mode(True)
            print(f"✅ {result['message']}")
            print(f"🚀 Fast Reflex Agent is now {'active' if result['current_state'] else 'inactive'}")
            
        elif mode in ['off', 'disable', 'false']:
            result = reflex_agent.toggle_reflex_mode(False)
            print(f"✅ {result['message']}")
            print(f"⏸️ Fast Reflex Agent is now {'active' if result['current_state'] else 'inactive'}")
            
        else:
            print("❌ Invalid mode. Use 'on' or 'off'")
            print("💡 Usage: reflex_mode on|off")
            
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error toggling reflex mode: {e}")
        return 'continue'

def handle_reflex_status_command() -> str:
    """Handle /reflex_status command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        reflex_agent = registry.get_agent('fast_reflex')
        
        if not reflex_agent:
            print("❌ Fast Reflex Agent not available")
            return 'continue'
        
        status = reflex_agent.get_reflex_status()
        
        print(f"\n⚡ **Fast Reflex Agent Status**")
        print(f"{'='*50}")
        
        # Main status
        reflex_enabled = status['reflex_mode_enabled']
        auto_trigger = status['auto_trigger_enabled']
        print(f"🎯 **Mode:** {'🟢 ENABLED' if reflex_enabled else '🔴 DISABLED'}")
        print(f"🤖 **Auto-Trigger:** {'🟢 ON' if auto_trigger else '🔴 OFF'}")
        
        # Cognitive load metrics
        load = status['cognitive_load']
        print(f"\n🧠 **Cognitive Load Metrics:**")
        print(f"  • Active Agents: {load['active_agents']}")
        print(f"  • Pending Tasks: {load['pending_tasks']}")
        print(f"  • Memory Usage: {load['memory_usage_percent']:.1f}%")
        print(f"  • Response Latency: {load['response_latency_ms']:.1f}ms")
        print(f"  • Error Rate: {load['error_rate']:.1%}")
        print(f"  • Complexity Score: {load['complexity_score']:.2f}")
        
        # Performance metrics
        perf = status['performance']
        print(f"\n📊 **Performance Metrics:**")
        print(f"  • Total Responses: {perf['total_responses']}")
        print(f"  • Avg Response Time: {perf['avg_response_time_ms']:.1f}ms")
        print(f"  • Defer Rate: {perf['defer_rate']:.1%}")
        print(f"  • Cached Patterns: {perf['cached_patterns']}")
        print(f"  • Cache Size: {perf['cache_size']}")
        
        # Thresholds
        thresh = status['thresholds']
        print(f"\n⚙️ **Trigger Thresholds:**")
        print(f"  • Cognitive Load: {thresh['cognitive_load_threshold']}")
        print(f"  • Timeout: {thresh['timeout_threshold_ms']}ms")
        print(f"  • Defer Threshold: {thresh['defer_threshold']}")
        print(f"  • Max Reflex Time: {thresh['max_reflex_time_ms']}ms")
        
        # Usage tips
        print(f"\n💡 **Usage Tips:**")
        if not reflex_enabled:
            print(f"  • Use 'reflex_mode on' to enable fast responses")
        else:
            print(f"  • Use 'reflex_mode off' to disable fast responses")
        print(f"  • Fast reflex handles simple queries automatically")
        print(f"  • Complex tasks are deferred to specialized agents")
        print(f"  • Cognitive load triggers automatic activation")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error getting reflex status: {e}")
        return 'continue'


# ===== PHASE 32: DISTRIBUTED AGENT MESH CLI HANDLERS =====

def handle_mesh_status_command() -> str:
    """Handle /mesh_status command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        mesh_manager = registry.get_agent('mesh_manager')
        
        if not mesh_manager:
            print("❌ Agent Mesh Manager not available")
            return 'continue'
        
        status = mesh_manager.get_mesh_status()
        
        if not status['mesh_enabled']:
            print(f"\n🌐 **Agent Mesh Status: 🔴 DISABLED**")
            print(f"{'='*50}")
            print(f"💡 **Usage:**")
            print(f"  • Use 'mesh_enable' to start mesh networking")
            print(f"  • Mesh enables distributed coordination across nodes")
            print(f"  • Supports SQLite (local), Redis (distributed), gRPC (cross-device)")
            return 'continue'
        
        # Display comprehensive mesh status
        local_node = status['local_node']
        stats = status['mesh_statistics']
        
        print(f"\n🌐 **Agent Mesh Status: 🟢 ENABLED**")
        print(f"{'='*60}")
        
        # Local node info
        print(f"🏠 **Local Node:**")
        print(f"  • Node ID: {local_node['node_id'][:16]}...")
        print(f"  • Type: {local_node['node_type']}")
        print(f"  • Status: {'🟢' if local_node['status'] == 'active' else '🟡'} {local_node['status'].upper()}")
        print(f"  • Protocol: {status['protocol']}")
        print(f"  • Load: {local_node['current_load']:.1%} ({local_node['current_load']*local_node['max_capacity']:.0f}/{local_node['max_capacity']})")
        
        # Capabilities and specializations
        if local_node['capabilities']:
            print(f"  • Capabilities: {', '.join(local_node['capabilities'][:5])}")
        if local_node['specializations']:
            print(f"  • Specializations: {', '.join(local_node['specializations'])}")
        
        # Mesh statistics
        print(f"\n📊 **Mesh Statistics:**")
        print(f"  • Total Nodes: {stats['total_nodes']}")
        print(f"  • Active Nodes: {stats['active_nodes']}")
        print(f"  • Pending Tasks: {stats['pending_tasks']}")
        print(f"  • Running Tasks: {stats['running_tasks']}")
        print(f"  • Completed Tasks: {stats['completed_tasks']}")
        print(f"  • Memory Sync Items: {stats['memory_sync_items']}")
        
        # Known nodes
        if status['known_nodes']:
            print(f"\n🤖 **Known Nodes ({len(status['known_nodes'])}):**")
            for node in status['known_nodes'][:5]:  # Show first 5
                node_status = '🟢' if node['status'] == 'active' else '🟡' if node['status'] == 'busy' else '🔴'
                print(f"  • {node_status} {node['node_id'][:12]}... ({node['node_type']}) - Load: {node['current_load']:.1%}")
            
            if len(status['known_nodes']) > 5:
                print(f"  • ... and {len(status['known_nodes']) - 5} more nodes")
        
        # Backends status
        backends = status['backends']
        print(f"\n⚙️ **Communication Backends:**")
        print(f"  • SQLite: {'🟢 Active' if backends['sqlite'] else '🔴 Inactive'}")
        print(f"  • Redis: {'🟢 Active' if backends['redis'] else '🔴 Inactive'}")
        print(f"  • gRPC: {'🟢 Active' if backends['grpc'] else '🔴 Inactive'}")
        
        # Recent tasks
        if status['recent_tasks']:
            print(f"\n📋 **Recent Tasks:**")
            for task in status['recent_tasks']:
                task_status = '✅' if task.get('completed_at') else '🏃' if task.get('started_at') else '⏳'
                print(f"  • {task_status} {task['task_type']} (Priority: {task['priority']})")
        
        # Usage tips
        print(f"\n💡 **Commands:**")
        print(f"  • 'mesh_agents' - Show detailed node information")
        print(f"  • 'mesh_sync' - Trigger memory synchronization")
        print(f"  • 'mesh_disable' - Disable mesh networking")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error getting mesh status: {e}")
        return 'continue'

def handle_mesh_enable_command() -> str:
    """Handle /mesh_enable command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        mesh_manager = registry.get_agent('mesh_manager')
        
        if not mesh_manager:
            print("❌ Agent Mesh Manager not available")
            return 'continue'
        
        result = mesh_manager.enable_mesh()
        
        if result['status'] == 'enabled':
            print(f"✅ Mesh networking enabled successfully!")
            print(f"🏠 **Node Details:**")
            print(f"  • Node ID: {result['node_id'][:16]}...")
            print(f"  • Type: {result['node_type']}")
            print(f"  • Protocol: {result['protocol']}")
            print(f"  • Capabilities: {', '.join(result.get('capabilities', [])[:3])}")
            if result.get('specializations'):
                print(f"  • Specializations: {', '.join(result['specializations'])}")
            print(f"\n🌐 Mesh is now coordinating with other nodes...")
            
        elif result['status'] == 'already_enabled':
            print(f"ℹ️ Mesh networking is already enabled")
            print(f"🏠 Node ID: {result['node_id'][:16]}...")
            
        else:
            print(f"❌ Failed to enable mesh: {result.get('message', 'Unknown error')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error enabling mesh: {e}")
        return 'continue'

def handle_mesh_disable_command() -> str:
    """Handle /mesh_disable command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        mesh_manager = registry.get_agent('mesh_manager')
        
        if not mesh_manager:
            print("❌ Agent Mesh Manager not available")
            return 'continue'
        
        result = mesh_manager.disable_mesh()
        
        if result['status'] == 'disabled':
            print(f"✅ Mesh networking disabled successfully")
            print(f"🏠 Node {result['node_id'][:16]}... has left the mesh")
            print(f"💡 Use 'mesh_enable' to rejoin the mesh")
            
        elif result['status'] == 'already_disabled':
            print(f"ℹ️ Mesh networking is already disabled")
            
        else:
            print(f"❌ Failed to disable mesh: {result.get('message', 'Unknown error')}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error disabling mesh: {e}")
        return 'continue'

def handle_mesh_sync_command() -> str:
    """Handle /mesh_sync command"""
    try:
        from agents.registry import AgentRegistry
        
        registry = AgentRegistry()
        mesh_manager = registry.get_agent('mesh_manager')
        
        if not mesh_manager:
            print("❌ Agent Mesh Manager not available")
            return 'continue'
        
        if not mesh_manager.mesh_enabled:
            print("❌ Mesh networking is not enabled")
            print("💡 Use 'mesh_enable' first to start mesh networking")
            return 'continue'
        
        # Trigger manual memory sync
        print("🔄 Initiating mesh memory synchronization...")
        
        # Example: sync recent memory items
        try:
            sync_id = mesh_manager.sync_memory_item(
                memory_type="manual_sync",
                memory_data={"sync_trigger": "manual", "timestamp": str(datetime.now())},
                priority=8
            )
            
            print(f"✅ Memory sync initiated")
            print(f"🆔 Sync ID: {sync_id[:16]}...")
            print(f"📡 Broadcasting to mesh nodes...")
            
            # Get current status after sync
            status = mesh_manager.get_mesh_status()
            if status['mesh_enabled']:
                nodes = status['mesh_statistics']['total_nodes']
                print(f"🌐 Synchronized with {nodes} total nodes in mesh")
            
        except Exception as e:
            print(f"❌ Failed to initiate sync: {e}")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error in mesh sync: {e}")
        return 'continue'

def handle_mesh_agents_command() -> str:
    """Handle /mesh_agents command"""
    try:
        from agents.registry import AgentRegistry
        from datetime import datetime, timedelta
        
        registry = AgentRegistry()
        mesh_manager = registry.get_agent('mesh_manager')
        
        if not mesh_manager:
            print("❌ Agent Mesh Manager not available")
            return 'continue'
        
        status = mesh_manager.get_mesh_status()
        
        if not status['mesh_enabled']:
            print("❌ Mesh networking is not enabled")
            print("💡 Use 'mesh_enable' first to start mesh networking")
            return 'continue'
        
        print(f"\n🤖 **Mesh Agents Overview**")
        print(f"{'='*60}")
        
        # Local node detailed info
        local_node = status['local_node']
        print(f"🏠 **Local Node (This Instance):**")
        print(f"  📧 ID: {local_node['node_id']}")
        print(f"  🏷️ Type: {local_node['node_type']}")
        print(f"  📊 Status: {'🟢' if local_node['status'] == 'active' else '🟡'} {local_node['status'].upper()}")
        print(f"  🖥️ Host: {local_node['hostname']}:{local_node['port']}")
        print(f"  ⚡ Load: {local_node['current_load']:.1%} ({local_node['current_load']*local_node['max_capacity']:.0f}/{local_node['max_capacity']})")
        print(f"  💾 Memory: {local_node.get('memory_usage_mb', 0):.1f}MB")
        print(f"  🔧 CPU: {local_node.get('cpu_usage_percent', 0):.1f}%")
        print(f"  ⏱️ Avg Response: {local_node.get('average_response_time', 0):.1f}ms")
        print(f"  ✅ Tasks Done: {local_node.get('total_tasks_completed', 0)}")
        print(f"  ❌ Error Rate: {local_node.get('error_rate', 0):.1%}")
        
        if local_node['capabilities']:
            print(f"  🛠️ Capabilities: {', '.join(local_node['capabilities'])}")
        if local_node['specializations']:
            print(f"  🎯 Specializations: {', '.join(local_node['specializations'])}")
        
        # Remote nodes
        known_nodes = status['known_nodes']
        if known_nodes:
            print(f"\n🌐 **Remote Nodes ({len(known_nodes)}):**")
            
            for i, node in enumerate(known_nodes, 1):
                # Calculate time since last heartbeat
                try:
                    last_hb = datetime.fromisoformat(node['last_heartbeat'])
                    time_diff = datetime.now() - last_hb
                    hb_status = "🟢 Active" if time_diff.seconds < 120 else "🟡 Stale" if time_diff.seconds < 300 else "🔴 Offline"
                except:
                    hb_status = "❓ Unknown"
                
                # Status indicators
                status_icon = '🟢' if node['status'] == 'active' else '🟡' if node['status'] == 'busy' else '🔴'
                
                print(f"\n  {i}. {status_icon} **{node['node_type'].replace('_', ' ').title()} Node**")
                print(f"     📧 ID: {node['node_id'][:24]}...")
                print(f"     🖥️ Host: {node['hostname']}:{node['port']}")
                print(f"     📊 Status: {node['status']} ({hb_status})")
                print(f"     ⚡ Load: {node['current_load']:.1%}")
                print(f"     💾 Memory: {node.get('memory_usage_mb', 0):.1f}MB")
                print(f"     ✅ Tasks: {node.get('total_tasks_completed', 0)}")
                
                if node['specializations']:
                    print(f"     🎯 Specializes in: {', '.join(node['specializations'][:3])}")
        else:
            print(f"\n🌐 **Remote Nodes:** None discovered yet")
            print(f"   💡 Other nodes will appear here as they join the mesh")
        
        # Mesh topology summary
        stats = status['mesh_statistics']
        print(f"\n📈 **Mesh Summary:**")
        print(f"  • Total Network Size: {stats['total_nodes']} nodes")
        print(f"  • Active Participants: {stats['active_nodes']} nodes")
        print(f"  • Workload Distribution: {stats['pending_tasks']} pending, {stats['running_tasks']} running")
        print(f"  • Memory Sync: {stats['memory_sync_items']} items in queue")
        
        # Communication status
        backends = status['backends']
        active_backends = [name for name, active in backends.items() if active]
        print(f"  • Communication: {', '.join(active_backends) if active_backends else 'None'}")
        
        print(f"\n💡 **Mesh Commands:**")
        print(f"  • 'mesh_status' - Quick mesh overview")
        print(f"  • 'mesh_sync' - Force memory synchronization")
        print(f"  • 'mesh_disable' - Leave the mesh network")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error showing mesh agents: {e}")
        return 'continue'


# ===== PHASE 33: NEURAL CONTROL HOOKS CLI HANDLERS =====

def handle_arbiter_trace_command(limit: int = 10) -> str:
    """Handle /arbiter_trace [limit] command - shows cognitive arbitration decision history"""
    try:
        from agents.registry import get_agent_registry
        
        registry = get_agent_registry()
        arbiter = registry.get_agent('cognitive_arbiter')
        
        if not arbiter:
            print("❌ CognitiveArbiter not available")
            return 'continue'
        
        # Get arbitration traces
        traces = arbiter.get_arbitration_trace(limit)
        
        if not traces:
            print("📊 No arbitration traces found")
            return 'continue'
        
        print(f"\n🧠 **Cognitive Arbitration Trace (Last {len(traces)} decisions)**")
        print("=" * 70)
        
        for i, trace in enumerate(reversed(traces), 1):
            print(f"\n**{i}. Decision {trace['decision_id']}**")
            print(f"🕐 Time: {trace['timestamp']}")
            print(f"🎯 Chosen Mode: **{trace['chosen_mode']}**")
            print(f"📍 Primary Reason: {trace['primary_reason']}")
            print(f"📊 Confidence: {trace['confidence']:.2f}")
            print(f"⚡ Agent: {trace['execution_agent']}")
            
            if trace['duration']:
                print(f"⏱️  Duration: {trace['duration']:.2f}s")
            
            if trace['success'] is not None:
                status = "✅ Success" if trace['success'] else "❌ Failed"
                print(f"📈 Status: {status}")
            
            # Input summary
            input_summary = trace['input_summary']
            print(f"\n📊 **Input Factors:**")
            print(f"  • Dissonance Pressure: {input_summary['dissonance_pressure']:.2f}")
            print(f"  • Confidence Score: {input_summary['confidence_score']:.2f}")
            print(f"  • Time Budget: {input_summary['time_budget']:.1f}s")
            print(f"  • Task Complexity: {input_summary['task_complexity']:.2f}")
            
            # Reasoning chain
            if trace['reasoning_chain']:
                print(f"\n🔍 **Decision Reasoning:**")
                for step in trace['reasoning_chain']:
                    print(f"  • {step}")
            
            if i < len(traces):
                print("-" * 50)
        
        # Performance summary
        print(f"\n📈 **Performance Summary:**")
        mode_counts = {}
        success_counts = {}
        
        for trace in traces:
            mode = trace['chosen_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            if trace['success'] is not None:
                if mode not in success_counts:
                    success_counts[mode] = {'total': 0, 'successful': 0}
                success_counts[mode]['total'] += 1
                if trace['success']:
                    success_counts[mode]['successful'] += 1
        
        print("\n**Mode Usage:**")
        for mode, count in mode_counts.items():
            percentage = (count / len(traces)) * 100
            print(f"  • {mode}: {count} times ({percentage:.1f}%)")
            
            if mode in success_counts and success_counts[mode]['total'] > 0:
                success_rate = (success_counts[mode]['successful'] / success_counts[mode]['total']) * 100
                print(f"    └─ Success rate: {success_rate:.1f}%")
        
        print(f"\n💡 **Usage:** /arbiter_trace [limit] to see more/fewer traces")
        
        return 'continue'
        
    except Exception as e:
        print(f"❌ Error showing arbitration trace: {e}")
        return 'continue'


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "start_dashboard":
        start_dashboard()
    # Add other CLI commands here as needed 