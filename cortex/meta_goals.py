import json
import os
from datetime import datetime
from storage.auto_reconciliation import AutoReconciliationEngine
from storage.memory_compression import MemoryCompressionEngine

def execute_meta_goals(memory_log):
    """
    Execute suggested meta-goals from generate_meta_goals().
    
    Args:
        memory_log: MemoryLog instance
        
    Returns:
        List of executed goals and their results
    """
    goals = memory_log.generate_meta_goals()
    executed = []
    
    for goal in goals:
        try:
            result = "success"
            
            if "run memory reconciliation" in goal.lower():
                auto_recon = AutoReconciliationEngine(memory_log)
                auto_recon.trigger_check()
                print(f"🔄 Executed: {goal}")
                
            elif "compress_cluster" in goal.lower():
                # Extract subject from goal
                if "for subject" in goal:
                    subject = goal.split("'")[1] if "'" in goal else goal.split('"')[1]
                    compression_engine = MemoryCompressionEngine(memory_log)
                    compression_engine.compress_subject_clusters(subject)
                    print(f"🗜️ Executed: {goal}")
                    
            elif "reconcile contradictions" in goal.lower():
                auto_recon = AutoReconciliationEngine(memory_log)
                auto_recon.trigger_check()
                print(f"⚖️ Executed: {goal}")
                
            else:
                result = "unknown_action"
                print(f"❓ Unknown meta-goal: {goal}")
                
            # Log the execution
            log_entry = {
                "type": "meta_goal_exec",
                "goal": goal,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            
            # Append to trace log
            with open("logs/trace.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            executed.append({"goal": goal, "result": result})
            
        except Exception as e:
            print(f"❌ Failed to execute meta-goal '{goal}': {e}")
            executed.append({"goal": goal, "result": "error", "error": str(e)})
    
    return executed 