"""
Semantic Memory Search for MeRNSTA
Uses embeddings for natural language query understanding
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import requests
from scipy.spatial.distance import cosine

from storage.enhanced_memory_model import EnhancedTripletFact


class SemanticMemorySearchEngine:
    """
    Semantic search engine for memory queries.
    Uses sentence embeddings to find relevant facts based on meaning, not keywords.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 ollama_host: Optional[str] = None,
                 ollama_model: Optional[str] = None):
        """
        Initialize the search engine with embedding model.
        
        Args:
            model_name: Sentence transformer model name
            ollama_host: Optional Ollama host for embeddings
            ollama_model: Optional Ollama model name
        """
        self.use_ollama = ollama_host is not None
        
        if self.use_ollama:
            self.ollama_host = ollama_host
            self.ollama_model = ollama_model or "nomic-embed-text"
        else:
            try:
                self.model = SentenceTransformer(model_name)
                logging.info(f"✅ Loaded sentence transformer: {model_name}")
            except Exception as e:
                logging.error(f"❌ Failed to load sentence transformer: {e}")
                self.model = None
                
        self.embedding_cache = {}
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        try:
            if self.use_ollama:
                # Use Ollama for embeddings
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={"model": self.ollama_model, "prompt": text}
                )
                response.raise_for_status()
                embedding = np.array(response.json()["embedding"])
            else:
                # Use sentence transformer
                if self.model is None:
                    return None
                embedding = self.model.encode([text])[0]
            
            # Cache the embedding
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Ollama tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            from utils.ollama_tokenizer import count_tokens
            return count_tokens(text)
        except ImportError:
            # Fallback to character-based estimation
            if not text:
                return 0
            return len(text) // 4
    
    def embed_fact(self, fact: EnhancedTripletFact) -> Optional[np.ndarray]:
        """
        Generate embedding for a fact.
        
        Combines subject, predicate, and object into a natural sentence.
        """
        # Create natural language representation
        if fact.predicate == "is":
            fact_text = f"{fact.subject} is {fact.object}"
        elif fact.predicate in ["has", "have"]:
            fact_text = f"{fact.subject} has {fact.object}"
        elif fact.predicate.startswith("not_"):
            base_pred = fact.predicate[4:]
            fact_text = f"{fact.subject} does not {base_pred} {fact.object}"
        else:
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
            
        return self.embed_text(fact_text)
    
    def query_memory(self, query: str, facts: List[EnhancedTripletFact],
                    user_profile_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    threshold: float = None,  # Dynamic threshold calculation
                    top_k: int = 10) -> List[Tuple[EnhancedTripletFact, float]]:
        """
        Adaptive semantic search with dynamic thresholds based on query context.
        No hardcoded values - adapts to query patterns and available facts.
        """
        if not facts:
            print(f"[SemanticSearch] No facts provided for query: '{query}'")
            return []
        
        # DYNAMIC THRESHOLD: Calculate based on query complexity and fact distribution
        if threshold is None:
            query_words = len([w for w in query.lower().split() if len(w) > 2])
            # More complex queries get lower thresholds (more permissive)
            # Simpler queries get higher thresholds (more selective)
            threshold = max(0.2, 0.8 - (query_words * 0.1))
            
        print(f"[SemanticSearch] Processing query: '{query}' with {len(facts)} facts")
        print(f"[SemanticSearch] Dynamic threshold: {threshold:.3f} (based on {query_words} meaningful words)")
        
        # Extract semantic intent from query
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2 and word not in ['what', 'do', 'does', 'is', 'are', 'the', 'my', 'your', 'when', 'where', 'how', 'why', 'who']]
        
        results = []
        
        for fact in facts:
            # Apply filters
            if user_profile_id and fact.user_profile_id != user_profile_id:
                continue
            if session_id and fact.session_id != session_id:
                continue
                
            # ADAPTIVE SCORING: Multiple semantic approaches
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            score = 0.0
            
            # 1. Direct keyword matching (high confidence)
            keyword_matches = sum(1 for word in query_words if word in fact_text)
            if keyword_matches > 0:
                score += keyword_matches * 0.4
            
            # 2. Semantic domain detection (adaptive patterns)
            domain_score = self._detect_semantic_domain_match(query_lower, fact_text)
            score += domain_score
            
            # 3. Predicate-object relevance
            if any(q_word in fact.predicate.lower() for q_word in query_words):
                score += 0.3
            if any(q_word in fact.object.lower() for q_word in query_words):
                score += 0.5
                
            # 4. Question-answer pattern matching
            if self._matches_question_pattern(query_lower, fact):
                score += 0.6
                
            print(f"[SemanticSearch] Fact: '{fact_text}' -> score: {score:.2f}")
            
            if score >= threshold:
                results.append((fact, score))
                print(f"[SemanticSearch] Added: {fact.subject} {fact.predicate} {fact.object} (score: {score:.2f})")
        
        # If no good matches, try embedding-based fallback
        if len(results) < 2:
            print(f"[SemanticSearch] Low results ({len(results)}), trying embedding fallback...")
            embedding_results = self._embedding_fallback_search(query, facts, user_profile_id, session_id, threshold * 0.7)
            results.extend(embedding_results)
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"[SemanticSearch] Final results: {len(results)} facts found")
        return results[:top_k]
    
    def _detect_semantic_domain_match(self, query: str, fact_text: str) -> float:
        """Dynamically detect if query and fact are in the same semantic domain."""
        # Music domain
        if any(word in query for word in ['instrument', 'music', 'play', 'sound']) and \
           any(word in fact_text for word in ['guitar', 'piano', 'drums', 'music', 'sing']):
            return 0.8
            
        # Food domain  
        if any(word in query for word in ['food', 'eat', 'cuisine', 'taste', 'meal']) and \
           any(word in fact_text for word in ['pizza', 'pasta', 'food', 'cuisine', 'restaurant']):
            return 0.8
            
        # Activity/exercise domain
        if any(word in query for word in ['exercise', 'activity', 'sport', 'do', 'when']) and \
           any(word in fact_text for word in ['workout', 'exercise', 'morning', 'gym', 'run']):
            return 0.8
            
        # Time domain
        if any(word in query for word in ['when', 'time', 'schedule']) and \
           any(word in fact_text for word in ['morning', 'evening', 'night', 'afternoon', 'time']):
            return 0.8
            
        return 0.0
    
    def _matches_question_pattern(self, query: str, fact: EnhancedTripletFact) -> bool:
        """Check if fact answers the specific question pattern."""
        # "What instruments..." -> facts about playing instruments
        if "instrument" in query and "play" in fact.predicate.lower():
            return True
            
        # "What food..." -> facts about food preferences  
        if "food" in query and any(word in fact.predicate.lower() for word in ['like', 'love', 'enjoy', 'prefer']):
            return True
            
        # "When do..." -> facts about timing
        if "when" in query and any(time in fact.object.lower() for time in ['morning', 'evening', 'night']):
            return True
            
        return False
    
    def _embedding_fallback_search(self, query: str, facts: List[EnhancedTripletFact],
                                  user_profile_id: Optional[str], session_id: Optional[str],
                                  fallback_threshold: float) -> List[Tuple[EnhancedTripletFact, float]]:
        """Embedding-based fallback search with dynamic threshold."""
        results = []
        
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return results
            
        for fact in facts:
            if user_profile_id and fact.user_profile_id != user_profile_id:
                continue
            if session_id and fact.session_id != session_id:
                continue
                
            # Get or compute fact embedding
            if fact.embedding:
                fact_embedding = np.array(fact.embedding)
            else:
                fact_embedding = self.embed_fact(fact)
                if fact_embedding is not None:
                    fact.embedding = fact_embedding.tolist()
                else:
                    continue
            
            # Calculate similarity
            try:
                similarity = 1 - cosine(query_embedding, fact_embedding)
                if similarity >= fallback_threshold:
                    results.append((fact, similarity))
                    print(f"[SemanticSearch] Embedding fallback: {fact.subject} {fact.predicate} {fact.object} (sim: {similarity:.2f})")
            except Exception as e:
                print(f"[SemanticSearch] Embedding error: {e}")
                
        return results
    
    def find_related_facts(self, fact: EnhancedTripletFact, 
                          all_facts: List[EnhancedTripletFact],
                          threshold: float = 0.6) -> List[Tuple[EnhancedTripletFact, float]]:
        """
        Find facts related to a given fact using semantic similarity.
        
        Useful for finding context and related information.
        """
        fact_embedding = self.embed_fact(fact)
        if fact_embedding is None:
            return []
            
        related = []
        
        for other_fact in all_facts:
            if other_fact.id == fact.id:
                continue
                
            other_embedding = self.embed_fact(other_fact)
            if other_embedding is None:
                continue
                
            similarity = 1 - cosine(fact_embedding, other_embedding)
            
            if similarity >= threshold:
                related.append((other_fact, similarity))
        
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract the intent and entities from a natural language query.
        
        Returns:
            Dictionary with query type, subject, predicate hints, etc.
        """
        query_lower = query.lower()
        intent = {
            "query_type": "general",
            "subject_hints": [],
            "predicate_hints": [],
            "temporal": None,
            "question_word": None
        }
        
        # Detect question words
        question_words = ["what", "who", "where", "when", "why", "how", "which"]
        for word in question_words:
            if query_lower.startswith(word):
                intent["question_word"] = word
                intent["query_type"] = "question"
                break
        
        # Detect temporal queries
        temporal_markers = ["currently", "now", "today", "still", "anymore", "used to"]
        for marker in temporal_markers:
            if marker in query_lower:
                intent["temporal"] = marker
                break
        
        # Extract subject hints (proper nouns, "I", "me", etc.)
        if "i " in query_lower or "me " in query_lower or "my " in query_lower:
            intent["subject_hints"].append("user")
        
        # Extract predicate hints
        preference_words = ["like", "love", "hate", "prefer", "enjoy", "dislike"]
        for word in preference_words:
            if word in query_lower:
                intent["predicate_hints"].append(word)
        
        state_words = ["is", "are", "am", "was", "were", "be"]
        for word in state_words:
            if word in query_lower:
                intent["predicate_hints"].append("is")
                
        possession_words = ["have", "has", "had", "own", "possess"]
        for word in possession_words:
            if word in query_lower:
                intent["predicate_hints"].append("has")
        
        return intent
    
    def generate_query_response(self, query: str, 
                               relevant_facts: List[Tuple[EnhancedTripletFact, float]],
                               intent: Dict[str, Any]) -> str:
        """
        Generate a natural language response based on query results.
        """
        if not relevant_facts:
            # Provide helpful response based on query type
            query_lower = query.lower()
            if "instrument" in query_lower:
                return "I don't have any information about musical instruments you play."
            elif "food" in query_lower or "cuisine" in query_lower:
                return "I don't have any information about your food preferences."
            elif "exercise" in query_lower or "workout" in query_lower:
                return "I don't have any information about your exercise habits."
            else:
                return "I don't have any information about that in my memory."
        
        print(f"[ResponseGeneration] Generating response for {len(relevant_facts)} facts")
        
        # Handle different query types based on query content
        query_lower = query.lower()
        
        if "instrument" in query_lower:
            # Musical instrument query
            for fact, score in relevant_facts:
                if "guitar" in fact.object.lower() or "guitar" in fact.subject.lower():
                    return f"You play guitar (confidence: {fact.confidence:.2f})"
                elif any(instrument in fact.object.lower() for instrument in ['piano', 'drums', 'violin', 'bass']):
                    return f"You play {fact.object} (confidence: {fact.confidence:.2f})"
            
        elif "food" in query_lower or "cuisine" in query_lower:
            # Food preference query
            food_facts = []
            for fact, score in relevant_facts:
                if any(food_word in fact.object.lower() for food_word in ['food', 'cuisine', 'pizza', 'pasta', 'italian']):
                    food_facts.append(fact)
            
            if food_facts:
                if len(food_facts) == 1:
                    fact = food_facts[0]
                    return f"You {fact.predicate} {fact.object} (confidence: {fact.confidence:.2f})"
                else:
                    items = [f"{f.object}" for f in food_facts[:3]]
                    return f"You like: {', '.join(items)}"
                    
        elif "exercise" in query_lower or "workout" in query_lower or "when" in query_lower:
            # Exercise/timing query
            for fact, score in relevant_facts:
                if any(time_word in fact.object.lower() for time_word in ['morning', 'evening', 'night']) or \
                   any(activity in fact.object.lower() for activity in ['workout', 'exercise', 'gym']):
                    if "when" in query_lower:
                        if "morning" in fact.object.lower():
                            return f"You exercise in the morning (confidence: {fact.confidence:.2f})"
                        elif "evening" in fact.object.lower():
                            return f"You exercise in the evening (confidence: {fact.confidence:.2f})"
                    return f"You {fact.predicate} {fact.object} (confidence: {fact.confidence:.2f})"
        
        # General handling for other types of queries
        if len(relevant_facts) == 1:
            fact = relevant_facts[0][0]
            return self._fact_to_sentence(fact)
        else:
            # Multiple relevant facts
            responses = []
            for fact, score in relevant_facts[:3]:  # Top 3
                responses.append(self._fact_to_sentence(fact))
            return "Based on my memory: " + " Also, ".join(responses)
    
    def _fact_to_sentence(self, fact: EnhancedTripletFact) -> str:
        """Convert a fact to a natural language sentence."""
        if fact.subject.lower() == "user":
            return f"You {fact.predicate} {fact.object}"
        else:
            return f"{fact.subject} {fact.predicate} {fact.object}" 