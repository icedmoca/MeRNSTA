from difflib import get_close_matches
import sys

# Available commands for fuzzy matching
AVAILABLE_COMMANDS = [
    "quit", "exit", "help", "list_facts", "delete_fact", "show_contradictions", 
    "resolve_contradiction", "resolve_contradiction_interactive", "list_facts_decayed", "reinforce_fact", "summarize_all",
    "summarize_subject", "show_summaries", "memory_mode", "set_memory_mode",
    "list_episodes", "show_episode", "delete_episode", "prune_memory", "forget_subject",
    "personality", "set_personality", "summarize_contradictions", "highlight_conflicts",
    "contradiction_clusters", "evolve_file_with_context", "generate_meta_goals",
    "health_check", "test_patterns", "show_emotive_facts", "reconcile",
    "embedding_cache_stats", "clear_embedding_cache", "test_agreement",
    "execute_meta_goals", "dump_memory", "sanitize_facts",
    "sentiment_trajectory", "opinion_summary", "volatility_report", "belief_context",
    "resolve_from_trajectory", "apply_decay", "trajectory_reflect"
]

def suggest_command(user_input: str) -> str:
    """
    Suggest the most similar command for fuzzy matching.
    
    Args:
        user_input: User's input command
        
    Returns:
        Suggested command or empty string if no good match
    """
    if not user_input:
        return ""
    
    # Try exact match first
    if user_input.lower() in AVAILABLE_COMMANDS:
        return ""
    
    # Find close matches
    matches = get_close_matches(user_input.lower(), AVAILABLE_COMMANDS, n=1, cutoff=0.6)
    
    if matches:
        return matches[0]
    
    return ""

def handle_fuzzy_command(user_input: str) -> str:
    """
    Handle fuzzy command matching and suggest corrections.
    
    Args:
        user_input: User's input command
        
    Returns:
        Corrected command or original input if no correction needed
    """
    suggestion = suggest_command(user_input)
    
    if suggestion:
        print(f"💡 Did you mean '{suggestion}'? (Type 'y' to use, or press Enter to continue)")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                return suggestion
        except (EOFError, KeyboardInterrupt):
            pass
    
    return user_input 