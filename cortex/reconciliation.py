"""
Reconciliation module for MeRNSTA cortex package.
Handles conflict summarization and resolution.
"""

from typing import List, Dict, Any
from storage.memory_log import MemoryLog
from storage.memory_utils import TripletFact

def summarize_conflicts(memory_log: MemoryLog, limit: int = 10) -> str:
    """
    Summarize recent conflicts/contradictions in memory.
    
    Args:
        memory_log: MemoryLog instance
        limit: Maximum number of conflicts to summarize
        
    Returns:
        String summary of conflicts
    """
    try:
        contradictions = memory_log.get_contradictions(resolved=False)[:limit]
        
        if not contradictions:
            return "No unresolved conflicts found in memory."
        
        summary_parts = [f"Found {len(contradictions)} unresolved conflicts:\n"]
        
        for i, contradiction in enumerate(contradictions, 1):
            summary_parts.append(
                f"{i}. '{contradiction['fact_a_text']}' vs '{contradiction['fact_b_text']}' "
                f"(confidence: {contradiction['confidence']:.2f})"
            )
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Error summarizing conflicts: {str(e)}" 