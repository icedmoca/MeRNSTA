"""
Response generation module for MeRNSTA cortex package.
Handles LLM response generation and reflection prompts.
"""

import requests
import json
import logging
from datetime import datetime
from storage.memory_log import MemoryLog
from storage.memory_utils import _normalize_subject
from config.settings import REFLECTIVE_PROMPTING
from storage.errors import safe_api_call, ExternalServiceError

@safe_api_call
def generate_response(prompt: str, context: str = None) -> str:
    """Generate response using Ollama API with memory validation"""
    from config.settings import ollama_host
    from storage.memory_utils import (
        validate_memory_response, 
        generate_uncertainty_response, 
        enhance_system_prompt_with_validation
    )
    
    url = f"{ollama_host}/api/generate"

    # Fast availability check + simple circuit breaker to avoid repeated timeouts
    try:
        from utils.ollama_checker import check_ollama_health
        if not check_ollama_health():
            raise requests.RequestException("Ollama unavailable")
    except Exception:
        # Treat as unavailable and go straight to fallback
        logging.warning("[API WARN] Skipping Ollama call: unavailable")
        from tools.llm_fallback import LLMFallbackAgent
        return LLMFallbackAgent()._simple_fallback_response(prompt)
    
    # Enhanced system prompt with validation instructions
    base_system_prompt = """You are a helpful AI assistant with access to a memory system. 
Keep your responses concise, clear, and directly relevant to the user's input. 
Avoid unnecessary verbosity and focus on providing helpful, actionable information."""
    
    system_prompt = enhance_system_prompt_with_validation(base_system_prompt)
    
    # Combine context and prompt with memory source attribution
    full_prompt = prompt
    if context:
        # Check for memory source context in the context data
        memory_source_context = ""
        try:
            # Extract memory source information if available
            from vector_memory import memory_source_logger
            if hasattr(memory_source_logger, 'get_recent_sources'):
                recent_sources = memory_source_logger.get_recent_sources(limit=1)
                if recent_sources:
                    memory_source_context = f"\n\n{recent_sources[0].get('formatted_sources', '')}"
        except:
            pass
        
        full_prompt = f"Context: {context}{memory_source_context}\n\nUser: {prompt}\n\nAssistant:"
    
    # Apply prompt budget using tokenizer if configured
    try:
        from config.settings import get_token_budget
        from utils.prompt_budget import budget_join
        budget = max(16, int(get_token_budget()))
        # Compose system + context + prompt within budget
        composed = budget_join([
            f"System:\n{system_prompt}",
            f"Context:\n{context}" if context else "",
            f"User:\n{prompt}\nAssistant:"
        ], budget=budget, sep="\n\n")
        full_prompt = composed
    except Exception:
        pass

    body = {
        "model": "mistral",
        "prompt": full_prompt,
        "system": system_prompt,
        "stream": False
    }
    try:
        # Shorter connect timeout and overall read timeout to prevent long hangs
        response = requests.post(url, json=body, timeout=(3.5, 10))
        response.raise_for_status()
        data = response.json()
        candidate_response = data.get("response", "")
        if not candidate_response:
            # Degrade if empty
            raise ExternalServiceError("Empty response from Ollama")
        
        # CRITICAL: Validate response before returning to prevent fabrication
        validation_result = validate_memory_response(
            user_query=prompt,
            memory_context=context or "",
            candidate_response=candidate_response,
            confidence_threshold=0.6
        )
        
        logging.info(f"[RESPONSE VALIDATION] Query: '{prompt}' | Valid: {validation_result['is_valid']} | Issues: {validation_result['issues']}")
        
        # If validation fails, use uncertainty response instead
        if not validation_result['is_valid'] and validation_result['recommended_response']:
            logging.warning(f"[FABRICATION PREVENTED] Replacing response due to validation issues: {validation_result['issues']}")
            base_response = validation_result['recommended_response']
        else:
            base_response = candidate_response
        
        # Add reflective prompting based on sentiment analysis
        # Check if reflective prompting is enabled
        if REFLECTIVE_PROMPTING:
            reflection = generate_conversational_reflection(prompt, context)
            if reflection:
                base_response += f"\n\n{reflection}"
        
        # Phase 9: Apply personality styling to response
        try:
            from agents.personality_engine import get_personality_engine
            
            personality_engine = get_personality_engine()
            
            # Create context for personality engine
            personality_context = {
                "user_query": prompt,
                "has_context": bool(context),
                "reflection_added": bool(REFLECTIVE_PROMPTING and reflection)
            }
            
            # Apply personality styling
            styled_response = personality_engine.style_response(base_response, personality_context)
            
            return styled_response
            
        except Exception as e:
            logging.warning(f"[PERSONALITY] Failed to apply personality styling: {e}")
            return base_response  # Return original response if personality styling fails
    except requests.RequestException as e:
        logging.warning(f"[API WARN] Ollama request failed: {e}")
        # Soft-degrade to LLMFallback simple response
        from tools.llm_fallback import LLMFallbackAgent
        simple = LLMFallbackAgent()._simple_fallback_response(prompt)
        return simple
    except Exception as e:
        logging.error(f"[API ERROR] generate_response failed: {e}", exc_info=True)
        from tools.llm_fallback import LLMFallbackAgent
        return LLMFallbackAgent()._simple_fallback_response(prompt)

def generate_reflective_prompt(user_input: str, context: str = None) -> str:
    """
    Generate reflective prompts based on sentiment trajectory and volatility analysis.
    
    Args:
        user_input: User's input
        context: Memory context
        
    Returns:
        Reflective prompt string or empty string if no reflection needed
    """
    try:
        from config.settings import DATABASE_CONFIG
        db_path = DATABASE_CONFIG.get("default_path", "memory.db")
        memory_log = MemoryLog(db_path)
        # Extract potential subjects from user input
        words = user_input.lower().split()
        potential_subjects = []
        
        for word in words:
            normalized = _normalize_subject(word)
            if normalized and len(normalized) > 2:  # Skip very short words
                potential_subjects.append(normalized)
        
        if not potential_subjects:
            return ""
        
        # Check sentiment trajectory for each potential subject
        reflections = []
        
        for subject in potential_subjects[:3]:  # Check top 3 subjects
            try:
                trajectory = memory_log.get_sentiment_trajectory(subject)
                
                if trajectory["fact_count"] >= 3:  # Need enough facts for analysis
                    slope = trajectory["slope"]
                    volatility = trajectory["volatility"]
                    recent_sentiment = trajectory["recent_sentiment"]
                    
                    # Generate reflection based on patterns
                    if abs(slope) > 0.4:  # Significant trend
                        if slope > 0.4:
                            reflections.append(f"You've been expressing more positive feelings about {subject} recently. Has something changed?")
                        elif slope < -0.4:
                            reflections.append(f"Your feelings about {subject} seem to have become more negative over time. What's been happening?")
                    
                    elif volatility > 0.5:  # High volatility
                        reflections.append(f"Your feelings about {subject} seem to fluctuate quite a bit. Is there something specific causing this uncertainty?")
                    
                    elif abs(recent_sentiment) > 0.7:  # Strong recent sentiment
                        if recent_sentiment > 0.7:
                            reflections.append(f"You've been quite positive about {subject} lately. What's driving this enthusiasm?")
                        elif recent_sentiment < -0.7:
                            reflections.append(f"You've been quite negative about {subject} recently. Is there a specific reason?")
                            
            except Exception as e:
                logging.debug(f"Error analyzing sentiment for {subject}: {e}")
                continue
        
        # Return the most relevant reflection
        if reflections:
            return reflections[0]  # Return the first reflection found
        
        return ""
        
    except Exception as e:
        logging.debug(f"Error generating reflective prompt: {e}")
        return ""

def generate_conversational_reflection(user_input: str, context: str = None) -> str:
    """
    Enhanced conversational reflection agent that notices emotional volatility and changing views.
    
    Args:
        user_input: User's input
        context: Memory context
        
    Returns:
        Natural reflection prompt or empty string
    """
    try:
        from config.settings import DATABASE_CONFIG
        db_path = DATABASE_CONFIG.get("default_path", "memory.db")
        memory_log = MemoryLog(db_path)
        # Extract potential subjects and objects from user input
        words = user_input.lower().split()
        potential_subjects = []
        potential_objects = []
        
        # Simple extraction - in practice, you might use more sophisticated NLP
        for i, word in enumerate(words):
            normalized = _normalize_subject(word)
            if normalized and len(normalized) > 2:
                potential_subjects.append(normalized)
                # Check if next word might be an object
                if i + 1 < len(words) and len(words[i + 1]) > 2:
                    potential_objects.append(words[i + 1])
        
        if not potential_subjects:
            return ""
        
        reflections = []
        
        # Analyze each subject-object pair for conversational reflection
        for subject in potential_subjects[:2]:  # Check top 2 subjects
            try:
                # Get trajectory for subject
                trajectory = memory_log.get_sentiment_trajectory(subject)
                
                if trajectory["fact_count"] >= 3:
                    slope = trajectory["slope"]
                    volatility = trajectory["volatility"]
                    recent_sentiment = trajectory["recent_sentiment"]
                    
                    # Check for rapid opinion changes (high volatility)
                    if volatility > 0.5:
                        reflections.append(f"You've changed your view on {subject} a few times lately. Is something shifting for you?")
                    
                    # Check for strong trends
                    elif abs(slope) > 0.5:
                        if slope > 0.5:
                            reflections.append(f"Your feelings about {subject} seem to be getting more positive over time. What's behind this change?")
                        else:
                            reflections.append(f"Your feelings about {subject} seem to be getting more negative. Has something changed?")
                    
                    # Check for emotional intensity
                    elif abs(recent_sentiment) > 0.8:
                        if recent_sentiment > 0.8:
                            reflections.append(f"You seem really passionate about {subject} lately. What's got you so excited?")
                        else:
                            reflections.append(f"You seem really frustrated with {subject} recently. What's been bothering you?")
                    
                    # Check for mixed feelings
                    elif abs(recent_sentiment) < 0.2 and volatility > 0.3:
                        reflections.append(f"Your feelings about {subject} seem a bit mixed. Are you still figuring out how you feel?")
                        
            except Exception as e:
                logging.debug(f"Error in conversational reflection for {subject}: {e}")
                continue
        
        # Also check for subject-object pairs if we have both
        for subject in potential_subjects[:1]:
            for object_ in potential_objects[:1]:
                try:
                    trajectory = memory_log.get_sentiment_trajectory(subject, object_)
                    
                    if trajectory["fact_count"] >= 2:
                        slope = trajectory["slope"]
                        volatility = trajectory["volatility"]
                        
                        # Check for specific subject-object volatility
                        if volatility > 0.6:
                            reflections.append(f"Your feelings about {subject} and {object_} seem to be all over the place. What's going on there?")
                        
                        # Check for strong subject-object trends
                        elif abs(slope) > 0.6:
                            if slope > 0.6:
                                reflections.append(f"You seem to be warming up to {subject} and {object_}. What changed?")
                            else:
                                reflections.append(f"You seem to be cooling on {subject} and {object_}. What happened?")
                                
                except Exception as e:
                    logging.debug(f"Error in subject-object reflection: {e}")
                    continue
        
        # Return the most relevant reflection
        if reflections:
            return reflections[0]
        
        return ""
        
    except Exception as e:
        logging.debug(f"Error in conversational reflection: {e}")
        return ""

def estimate_tokens(text: str) -> int:
    """Estimate token count for text using Ollama tokenizer."""
    try:
        from utils.ollama_tokenizer import count_tokens
        return count_tokens(text)
    except ImportError:
        # Fallback to character-based estimation
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

def ollama_stream(prompt: str, system_prompt: str = None):
    """Stream tokens from Ollama"""
    from config.settings import ollama_host
    url = f"{ollama_host}/api/generate"
    body = {"model": "mistral", "prompt": prompt, "stream": True}
    if system_prompt:
        body["system"] = system_prompt
    
    with requests.post(url, json=body, stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                yield json.loads(line)["response"] 