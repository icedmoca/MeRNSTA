#!/usr/bin/env python3
"""
LLM Fallback Agent for MeRNSTA

Provides conversational AI fallback for queries that cannot be answered
through memory search or symbolic reasoning. Uses existing LLM infrastructure.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from config.settings import get_config

class LLMFallbackAgent:
    """
    LLM-powered fallback agent for general conversation and reasoning.
    Integrates with existing response generation infrastructure.
    """
    
    def __init__(self):
        self.config = get_config().get('reasoning', {})
        self.enabled = self.config.get('enable_llm_fallback', True)
        self.model = self.config.get('llm_model', 'mistral')
        self.fallback_threshold = self.config.get('fallback_threshold', 0.5)
        
        # Conversational patterns that should use LLM
        self.conversational_patterns = [
            r'\b(?:how\s+are\s+you|hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b',
            r'\b(?:thank\s+you|thanks|goodbye|bye|see\s+you)\b',
            r'\b(?:what\s+do\s+you\s+think|your\s+opinion|how\s+do\s+you\s+feel)\b',
            r'\b(?:tell\s+me\s+about|explain|describe|what\s+is)\b',
            r'\b(?:can\s+you\s+help|need\s+help|assist\s+me)\b',
            r'^\s*(?:why|what|how|when|where|who)\s+.*\??\s*$'
        ]
        
        # Initialize response context
        self.conversation_context = []
        self.max_context_length = 5
        
    def should_use_llm_fallback(self, query: str, memory_results: List = None, confidence_score: float = 0.0) -> bool:
        """
        Determine if LLM fallback should be used based on query type and memory results.
        """
        if not self.enabled:
            return False
            
        # Use LLM if memory confidence is low
        if confidence_score < self.fallback_threshold:
            logging.debug(f"[LLMFallback] Low confidence ({confidence_score}), using LLM")
            return True
            
        # Use LLM if no memory results found
        if not memory_results or len(memory_results) == 0:
            logging.debug("[LLMFallback] No memory results, using LLM")
            return True
            
        # Check for conversational patterns
        import re
        query_lower = query.lower()
        for pattern in self.conversational_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logging.debug(f"[LLMFallback] Conversational pattern matched: {pattern}")
                return True
                
        return False
        
    def generate_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a conversational response using LLM.
        """
        if not self.enabled:
            return {
                "response": "I don't have any information about that in my memory.",
                "confidence": 0.0,
                "method": "disabled",
                "success": False
            }
            
        try:
            # Build context-aware prompt
            prompt = self._build_contextual_prompt(query, context)
            
            # Generate response using existing infrastructure
            response_text = self._call_llm(prompt)
            
            # Update conversation context
            self._update_context(query, response_text)
            
            return {
                "response": response_text,
                "confidence": 0.8,  # LLM responses have moderate confidence
                "method": "llm_fallback",
                "model": self.model,
                "success": True,
                "original_query": query
            }
            
        except Exception as e:
            # Robust degradation: fall back to simple pattern-based response
            logging.error(f"[LLMFallback] Error generating response: {e}")
            simple = self._simple_fallback_response(query)
            # Update context with simple response as well
            try:
                self._update_context(query, simple)
            except Exception:
                pass
            return {
                "response": simple,
                "confidence": 0.5,
                "method": "simple_fallback",
                "model": self.model,
                "success": True,
                "original_query": query,
                "error": str(e)
            }
            
    def _build_contextual_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Build a contextual prompt for the LLM that includes conversation history
        and any relevant memory context.
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(
            "You are MeRNSTA, an AI assistant with memory capabilities. "
            "Respond naturally and conversationally. Be helpful, friendly, and concise."
        )
        
        # Add memory context if available
        if context and context.get('memory_facts'):
            memory_facts = context['memory_facts']
            if memory_facts:
                prompt_parts.append(f"\nRelevant memory context: {'; '.join(memory_facts[:3])}")
                
        # Add conversation history
        if self.conversation_context:
            prompt_parts.append("\nRecent conversation:")
            for turn in self.conversation_context[-2:]:  # Last 2 turns
                prompt_parts.append(f"User: {turn['user']}")
                prompt_parts.append(f"Assistant: {turn['assistant']}")
                
        # Add current query
        prompt_parts.append(f"\nUser: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
        
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM using existing response generation infrastructure.
        """
        try:
            # Import here to avoid circular dependencies
            from cortex.response_generation import generate_response
            
            response = generate_response(prompt)
            
            # Clean up the response
            if response:
                # Remove any "Assistant:" prefix if it exists
                response = response.strip()
                if response.lower().startswith("assistant:"):
                    response = response[10:].strip()
                    
                return response
            else:
                return "I'm not sure how to respond to that."
                
        except ImportError:
            # Fallback if response_generation is not available
            logging.warning("[LLMFallback] response_generation not available, using simple fallback")
            return self._simple_fallback_response(prompt)
            
    def _simple_fallback_response(self, prompt: str) -> str:
        """
        Simple pattern-based responses when LLM is not available.
        """
        query_lower = prompt.lower()
        
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif any(farewell in query_lower for farewell in ['goodbye', 'bye', 'see you']):
            return "Goodbye! Feel free to come back anytime."
        elif 'how are you' in query_lower:
            return "I'm doing well, thank you for asking! How are you?"
        elif any(thanks in query_lower for thanks in ['thank you', 'thanks']):
            return "You're welcome! Happy to help."
        elif any(help_word in query_lower for help_word in ['help', 'assist']):
            return "I'm here to help! What would you like to know?"
        else:
            return "I don't have specific information about that, but I'm here to help with whatever I can!"
            
    def _update_context(self, user_query: str, assistant_response: str):
        """Update conversation context for future reference."""
        self.conversation_context.append({
            'user': user_query,
            'assistant': assistant_response,
            'timestamp': time.time()
        })
        
        # Keep only recent context
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
            
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation context."""
        return {
            "turns": len(self.conversation_context),
            "enabled": self.enabled,
            "model": self.model,
            "recent_topics": [turn['user'][:50] + "..." if len(turn['user']) > 50 else turn['user'] 
                             for turn in self.conversation_context[-3:]]
        }
        
    def clear_context(self):
        """Clear conversation context."""
        self.conversation_context = []
        logging.info("[LLMFallback] Conversation context cleared")
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about LLM fallback capabilities."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "fallback_threshold": self.fallback_threshold,
            "conversational_patterns": len(self.conversational_patterns),
            "context_length": len(self.conversation_context),
            "max_context": self.max_context_length
        } 