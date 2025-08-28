"""
Enhanced memory search functionality for MeRNSTA.
Handles natural language queries and returns structured results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from storage.memory_log import MemoryLog, TripletFact
from storage.memory_utils import normalize_question_to_subject, generate_uncertainty_response
from config.environment import get_settings
from dataclasses import dataclass
from config.settings import DEFAULT_VALUES
from storage.fact_manager import TripletFact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySearchEngine:
    """
    Advanced memory search engine that handles different types of user queries:
    - Fact retrieval ("What do I like?")
    - Specific queries ("What color cats do I like?")
    - Summarization requests ("Summarize our conversation")
    - Command detection (personality changes, etc.)
    """
    
    def __init__(self, memory_log: MemoryLog):
        self.memory_log = memory_log
        self.settings = get_settings()
        
    def search_facts(self, query: str, user_profile_id: str = "default_user", 
                    personality: str = "neutral", max_facts: int = 10, session_id: str = None) -> Dict[str, Any]:
        """
        Main search function that determines query type and returns appropriate results.
        
        Args:
            query: Natural language query
            user_profile_id: User profile for scoping results
            personality: Current personality for confidence adjustments
            max_facts: Maximum number of facts to return
            
        Returns:
            Dictionary with search results, query type, and metadata
        """
        query_lower = query.lower().strip()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        if query_type == "summarization":
            return self._handle_summarization(query, user_profile_id, session_id)
        elif query_type == "fact_retrieval":
            return self._handle_fact_retrieval(query, user_profile_id, personality, max_facts, session_id)
        elif query_type == "command":
            return self._handle_command(query)
        else:
            return self._handle_general_query(query, user_profile_id, personality, max_facts, session_id)
    
    def _detect_query_type(self, query_lower: str) -> str:
        """Detect the type of query to determine how to handle it."""
        
        # Summarization requests
        summarization_keywords = [
            "summarize", "summary", "recap", "overview", "conversation so far",
            "what have we talked about", "what did we discuss"
        ]
        if any(keyword in query_lower for keyword in summarization_keywords):
            return "summarization"
        
        # Command patterns
        command_keywords = [
            "switch to", "change personality", "set personality", "use personality",
            "clear memory", "reset", "help"
        ]
        if any(keyword in query_lower for keyword in command_keywords):
            return "command"
        
        # Fact retrieval patterns
        fact_keywords = [
            "what do i", "what are all", "tell me about", "what color", "what kind",
            "what is my", "what's my", "favorite", "like", "prefer", "enjoy"
        ]
        if any(keyword in query_lower for keyword in fact_keywords):
            return "fact_retrieval"
        
        return "general_query"
    
    def _handle_summarization(self, query: str, user_profile_id: str, session_id: str = None) -> Dict[str, Any]:
        """Handle summarization requests."""
        try:
            # Get recent facts for this user
            all_facts = self.memory_log.get_all_facts()
            user_facts = [f for f in all_facts if hasattr(f, 'user_profile_id') and 
                         getattr(f, 'user_profile_id', None) == user_profile_id]
            
            if not user_facts:
                user_facts = [f for f in all_facts if f.subject.lower() in ['user', 'i', 'me']]
            
            if not user_facts:
                return {
                    "type": "summarization",
                    "response": "No facts found to summarize. Our conversation is just beginning!",
                    "facts": [],
                    "success": True
                }
            
            # Group facts by category
            likes = [f for f in user_facts if f.predicate.lower() in ['like', 'love', 'enjoy']]
            dislikes = [f for f in user_facts if f.predicate.lower() in ['hate', 'dislike']]
            facts_about = [f for f in user_facts if f.predicate.lower() in ['is', 'am', 'are', 'have', 'has']]
            
            summary_parts = []
            if likes:
                liked_items = [f.object for f in likes[-3:]]  # Most recent 3
                summary_parts.append(f"You mentioned liking: {', '.join(liked_items)}")
            
            if dislikes:
                disliked_items = [f.object for f in dislikes[-3:]]
                summary_parts.append(f"You mentioned disliking: {', '.join(disliked_items)}")
            
            if facts_about:
                personal_facts = [f"{f.predicate} {f.object}" for f in facts_about[-3:]]
                summary_parts.append(f"About you: {', '.join(personal_facts)}")
            
            if summary_parts:
                response = "In our conversation, " + ". ".join(summary_parts) + "."
            else:
                response = f"We've discussed {len(user_facts)} topics, but I need to organize them better. What would you like to know about our conversation?"
            
            return {
                "type": "summarization",
                "response": response,
                "facts": user_facts[:10],  # Return some facts for context
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error in summarization: {e}")
            return {
                "type": "summarization",
                "response": "I had trouble summarizing our conversation. What specific topic would you like me to recall?",
                "facts": [],
                "success": False,
                "error": str(e)
            }
    
    def _handle_fact_retrieval(self, query: str, user_profile_id: str, 
                             personality: str, max_facts: int, session_id: str = None) -> Dict[str, Any]:
        """Handle fact retrieval queries like 'What do I like?'"""
        
        query_lower = query.lower()
        
        # Use linguistic analysis to understand the query intent
        query_intent = self._analyze_query_intent(query)
        
        if query_intent == "preference":
            return self._handle_preference_query(query, user_profile_id, personality)
        elif query_intent == "comparison":
            return self._handle_comparison_query(query, user_profile_id, personality)
        elif query_intent == "specific_property":
            return self._handle_specific_property_query(query, user_profile_id, personality)
        elif query_intent == "favorite":
            return self._handle_favorite_query(query, user_profile_id, personality)
        else:
            # General semantic search
            return self._semantic_search_facts(query, user_profile_id, personality, max_facts, session_id)
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query intent using linguistic features."""
        
        query_lower = query.lower()
        
        # Try spaCy analysis first
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                doc = nlp(query)
                
                # Look for comparison structures (X or Y)
                has_or = any(token.text.lower() == 'or' for token in doc)
                has_preference_verb = any(token.lemma_ in ['like', 'love', 'hate', 'prefer'] for token in doc)
                
                if has_or and has_preference_verb:
                    return "comparison"
                
                # Look for "favorite" or "fav"
                has_favorite = any(token.text.lower() in ['favorite', 'fav'] for token in doc)
                if has_favorite:
                    return "favorite"
                
                # Look for property queries (color, type, etc.)
                property_nouns = ['color', 'type', 'kind', 'style', 'size']
                has_property = any(token.text.lower() in property_nouns for token in doc)
                has_wh_word = any(token.tag_ in ['WDT', 'WP', 'WRB'] for token in doc)
                
                if has_property and has_wh_word:
                    return "specific_property"
                
                # Look for opinion/thought queries ("think about", "feel about", "opinion on")
                for i, token in enumerate(doc):
                    if token.lemma_ in ['think', 'feel', 'believe']:
                        # Check if followed by "about" or similar preposition
                        if i + 1 < len(doc) and doc[i + 1].text in ['about', 'of', 'on']:
                            return "preference"
                
                # General preference query
                if has_preference_verb and has_wh_word:
                    return "preference"
        except:
            pass
        
        # Fallback to basic pattern matching
        if " or " in query_lower and any(v in query_lower for v in ['like', 'love', 'hate']):
            return "comparison"
        elif 'favorite' in query_lower or 'fav' in query_lower:
            return "favorite"
        elif 'color' in query_lower and '?' in query:
            return "specific_property"
        elif any(phrase in query_lower for phrase in ['what do i like', 'what do i hate']):
            return "preference"
        # Add pattern for "think about" queries
        elif 'think about' in query_lower or 'feel about' in query_lower or 'opinion' in query_lower:
            return "preference"
        
        return "general"
    
    def _generate_natural_response(self, facts: List[Any], query: str, is_negative: bool = False) -> str:
        """Generate a natural language response from facts using LLM."""
        
        if not facts:
            return ""
            
        # Check if natural language responses are enabled
        try:
            from config.settings import _cfg
            use_natural = _cfg.get('system', {}).get('use_natural_language_responses', True)
            if not use_natural:
                return ""  # Let caller use fallback
        except:
            pass  # Default to trying natural language
            
        # Import here to avoid circular dependencies
        from cortex.response_generation import generate_response
        
        # Build context for LLM
        fact_summaries = []
        for fact in facts:
            if hasattr(fact, 'predicate'):
                predicate = fact.predicate
                obj = fact.object
                confidence = getattr(fact, 'confidence', 1.0)
            else:
                # Handle dict format
                predicate = fact.get('predicate', 'like')
                obj = fact.get('object', '')
                confidence = fact.get('confidence', 1.0)
            
            # Include confidence info for transparency
            fact_summaries.append(f"{predicate} {obj} (confidence: {confidence:.2f})")
        
        # Group facts by predicate type
        positive_facts = []
        negative_facts = []
        
        for fact in facts:
            pred_lower = fact.predicate.lower()
            if any(neg in pred_lower for neg in ['hate', 'dislike', 'detest']):
                negative_facts.append(fact)
            else:
                positive_facts.append(fact)
        
        # Create a prompt for natural response generation
        if len(facts) == 1:
            context = f"The user asked: '{query}'. Based on memory, they {fact_summaries[0]}."
            prompt = f"{context}\n\nGenerate a brief, natural response confirming this preference. Keep it conversational and include the confidence level naturally."
        else:
            # Handle mixed positive/negative facts
            if positive_facts and negative_facts:
                pos_items = [f"{f.object}" for f in positive_facts[:3]]
                neg_items = [f"{f.object}" for f in negative_facts[:3]]
                
                # Build positive part
                if len(pos_items) == 1:
                    pos_text = f"You like {pos_items[0]}"
                else:
                    pos_text = f"You like {', '.join(pos_items[:-1])} and {pos_items[-1]}"
                
                # Build negative part
                if len(neg_items) == 1:
                    neg_text = f"You dislike {neg_items[0]}"
                else:
                    neg_text = f"You dislike {', '.join(neg_items[:-1])} and {neg_items[-1]}"
                
                # If asking what they like, prioritize positive facts
                if not is_negative and 'like' in query.lower():
                    return f"{pos_text}. {neg_text}."
                elif is_negative and ('hate' in query.lower() or 'dislike' in query.lower()):
                    return f"{neg_text}. {pos_text}."
                else:
                    return f"{pos_text}. {neg_text}."
            elif positive_facts:
                items = [f"{f.object}" for f in positive_facts[:3]]
                return f"You like {', '.join(items[:-1])} and {items[-1]}." if len(items) > 1 else f"You like {items[0]}."
            else:
                items = [f"{f.object}" for f in negative_facts[:3]]
                return f"You dislike {', '.join(items[:-1])} and {items[-1]}." if len(items) > 1 else f"You dislike {items[0]}."
        
        try:
            # Generate natural response
            response = generate_response(prompt, max_tokens=150, temperature=0.7)
            
            # Fallback if generation fails
            if not response or response.strip() == "":
                raise Exception("Empty response from LLM")
                
            return response.strip()
            
        except Exception as e:
            logging.warning(f"Failed to generate natural response: {e}, using fallback")
            # Fallback to simple format
            if len(facts) == 1:
                fact = facts[0]
                return f"You {fact.predicate} {fact.object}."
            else:
                items = [f.object for f in facts[:3]]
                if is_negative:
                    return f"You dislike {', '.join(items[:-1])} and {items[-1]}." if len(items) > 1 else f"You dislike {items[0]}."
                else:
                    return f"You like {', '.join(items[:-1])} and {items[-1]}." if len(items) > 1 else f"You like {items[0]}."
    
    def _handle_preference_query(self, query: str, user_profile_id: str, personality: str) -> Dict[str, Any]:
        """Handle queries about what the user likes or dislikes."""
        
        # Extract what they're asking about (if specific)
        query_object = None
        
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                doc = nlp(query.lower())
                
                # Look for the object of prepositions like "about", "of", "on"
                for token in doc:
                    if token.text in ['about', 'of', 'on', 'regarding']:
                        # Get the object of the preposition
                        for child in token.children:
                            if child.dep_ == 'pobj':
                                query_object = child.text
                                break
                        break
        except:
            pass
        
        # Use linguistic analysis to determine preference type
        is_negative_query = False
        
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                doc = nlp(query.lower())
                # Look for negative sentiment verbs
                for token in doc:
                    if token.lemma_ in ['hate', 'dislike'] or (token.dep_ == 'neg' and token.head.lemma_ == 'like'):
                        is_negative_query = True
                        break
        except:
            # Basic fallback
            query_lower = query.lower()
            if any(word in query_lower for word in ['hate', 'dislike', "don't like", 'do not like']):
                is_negative_query = True
        
        # Get all facts about user preferences
        all_facts = self.memory_log.get_all_facts()
        preference_facts = []
        
        for fact in all_facts:
            if fact.subject.lower().startswith('user') or fact.subject.lower() == 'i':
                # If asking about a specific object, filter for it
                if query_object and query_object.lower() not in fact.object.lower():
                    continue
                    
                # Check if this is a preference fact based on the predicate
                # Using linguistic analysis instead of hardcoded lists
                pred_lower = fact.predicate.lower()
                
                # Determine if predicate expresses preference/sentiment
                is_preference_predicate = False
                
                # For single-word predicates, use a simpler check
                # as spaCy doesn't always correctly identify POS for single words
                if pred_lower in ['like', 'love', 'hate', 'dislike', 'prefer', 'enjoy', 'detest']:
                    is_preference_predicate = True
                else:
                    try:
                        from storage.spacy_extractor import nlp
                        if nlp:
                            # Analyze the predicate for more complex cases
                            pred_doc = nlp(pred_lower)
                            for token in pred_doc:
                                # Check if it contains preference-related words
                                if any(x in token.text for x in ['like', 'love', 'hate', 'prefer', 'enjoy']):
                                    is_preference_predicate = True
                                    break
                    except:
                        pass
                
                if is_preference_predicate:
                    # Check sentiment alignment
                    if is_negative_query:
                        # For negative queries, include negative predicates
                        if any(neg in pred_lower for neg in ['hate', 'dislike', 'detest']):
                            preference_facts.append(fact)
                    else:
                        # For general queries, include all preference facts
                        preference_facts.append(fact)
        
        if not preference_facts:
            if query_object:
                response_msg = f"I don't have information about your feelings toward {query_object}."
            elif is_negative_query:
                response_msg = "I don't have information about things you dislike."
            else:
                response_msg = "I don't have information about your preferences yet. Tell me what you like!"
            return {
                "type": "fact_retrieval",
                "response": response_msg,
                "facts": [],
                "success": True
            }
        
        # Format the response
        fact_count = len(preference_facts)
        formatted_facts = []
        
        for fact in preference_facts[:10]:  # Limit to 10 facts
            formatted_facts.append({
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": getattr(fact, 'confidence', 1.0),
                "id": fact.id
            })
        
        # Adjust confidence based on personality
        if personality == "skeptical":
            for f in formatted_facts:
                f["confidence"] *= 0.7
        elif personality == "loyal":
            for f in formatted_facts:
                f["confidence"] = min(f["confidence"] * 1.2, 1.0)
        
        # Build response
        if fact_count == 1:
            fact = preference_facts[0]
            confidence = getattr(fact, 'confidence', 1.0)
            if personality == "skeptical":
                confidence *= 0.7
            response = f"You {fact.predicate} {fact.object} (confidence: {confidence:.2f})"
        else: # fact_count > 1 or natural language enabled
            # Try to generate natural language response
            natural_response = self._generate_natural_response(preference_facts, query, is_negative_query)
            
            if natural_response:
                response = natural_response
            else:
                # Fallback to simple format if natural generation fails
                if fact_count == 1:
                    fact = preference_facts[0]
                    confidence = getattr(fact, 'confidence', 1.0)
                    response = f"You {fact.predicate} {fact.object} (confidence: {confidence:.2f})"
                else:
                    items = []
                    for fact in preference_facts:
                        confidence = getattr(fact, 'confidence', 1.0)
                        items.append(f"{fact.object} (confidence: {confidence:.2f})")
                    if is_negative_query:
                        response = f"You dislike: {', '.join(items)}"
                    else:
                        response = f"You like: {', '.join(items)}"
        
        return {
            "type": "fact_retrieval",
            "response": response,
            "facts": formatted_facts,
            "success": True,
            "personality_applied": personality,
            "query_type": "preference"
        }
    
    def _handle_specific_property_query(self, query: str, user_profile_id: str, personality: str) -> Dict[str, Any]:
        """Handle queries about specific properties like 'what color cats do i like'"""
        
        # Use NLP to extract the property and object
        property_type = None
        target_object = None
        
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                doc = nlp(query.lower())
                
                # Find property words (color, type, etc.)
                property_words = []
                object_words = []
                
                for token in doc:
                    if token.pos_ == 'NOUN':
                        if token.text in ['color', 'type', 'kind', 'style', 'size']:
                            property_words.append(token.text)
                        else:
                            object_words.append(token.text)
                
                if property_words:
                    property_type = property_words[0]
                if object_words:
                    target_object = object_words[-1]  # Usually the last noun is the main object
        except:
            # Basic fallback
            words = query.lower().split()
            for i, word in enumerate(words):
                if word == 'color':
                    property_type = 'color'
                    # Look for the next noun
                    for j in range(i+1, len(words)):
                        if words[j] not in ['do', 'i', 'like', 'hate', 'prefer']:
                            target_object = words[j].rstrip('s?')
                            break
        
        if not property_type or not target_object:
            return self._semantic_search_facts(query, user_profile_id, personality, 10)
        
        # Find facts about the object with that property
        all_facts = self.memory_log.get_all_facts()
        matching_facts = []
        
        for fact in all_facts:
            if (fact.subject.lower() in ['user', 'i'] and 
                target_object in fact.object.lower()):
                
                # Try to extract the property value
                if property_type == 'color':
                    # Look for color words before the object
                    obj_words = fact.object.lower().split()
                    for i, word in enumerate(obj_words):
                        if target_object in word and i > 0:
                            # Previous word might be the color
                            potential_color = obj_words[i-1]
                            matching_facts.append({
                                'fact': fact,
                                'property_value': potential_color
                            })
                            break
        
        if not matching_facts:
            return {
                "type": "fact_retrieval",
                "response": f"I don't have information about what {property_type} {target_object} you like or dislike.",
                "facts": [],
                "success": True
            }
        
        # Format response
        response_parts = []
        facts = []
        
        for match in matching_facts:
            fact = match['fact']
            value = match['property_value']
            
            if fact.predicate.lower() in ['like', 'love']:
                response_parts.append(f"You like {value} {target_object}")
            else:
                response_parts.append(f"You dislike {value} {target_object}")
            
            facts.append(fact)
        
        response = ". ".join(response_parts) + "."
        
        return {
            "type": "fact_retrieval",
            "response": response,
            "facts": facts[:3],
            "success": True,
            "personality_applied": personality
        }
    
    def _handle_favorite_query(self, query: str, user_profile_id: str, personality: str) -> Dict[str, Any]:
        """Handle favorite queries like 'what is my favorite color'"""
        
        # Extract what they're asking about
        query_clean = query.lower().replace('?', '').strip()
        words = query_clean.split()
        
        if 'favorite' in words:
            fav_index = words.index('favorite')
            if fav_index + 1 < len(words):
                target_property = ' '.join(words[fav_index + 1:])  # e.g., 'color', 'car make'
        elif 'fav' in words:
            fav_index = words.index('fav')
            if fav_index + 1 < len(words):
                target_property = ' '.join(words[fav_index + 1:])
        else:
            return {"type": "fact_retrieval", "response": "What's your favorite what? Please be more specific.", "facts": [], "success": False}
        
        # Search for facts with matching predicate or object
        all_facts = self.memory_log.get_all_facts()
        favorite_facts = []
        
        for f in all_facts:
            # Check various storage patterns for favorite facts
            # Pattern 1: "user favorite color" / "is" / "blue"
            if f.subject.lower() == f"user favorite {target_property}" and f.predicate.lower() == "is":
                favorite_facts.append(f)
            # Pattern 2: "user" / "favorite" / "color is blue"  
            elif f.subject.lower() in ['user', 'my', 'i'] and f.predicate.lower() == "favorite" and target_property in f.object.lower():
                favorite_facts.append(f)
            # Pattern 3: Standard pattern with favorite in predicate
            elif f.subject.lower() in ['user', 'my', 'i'] and 'favorite' in f.predicate.lower() and target_property in f.predicate.lower():
                favorite_facts.append(f)
        
        if not favorite_facts:
            return {
                "type": "fact_retrieval",
                "response": f"I don't know your favorite {target_property}. What is it?",
                "facts": [],
                "success": True
            }
        
        best_fact = max(favorite_facts, key=lambda f: getattr(f, 'confidence', 1.0))
        confidence = getattr(best_fact, 'confidence', 1.0)
        
        if personality == "skeptical":
            confidence /= 1.5
        
        return {
            "type": "fact_retrieval",
            "response": f"Your favorite {target_property} is {best_fact.object} (confidence: {confidence:.2f})",
            "facts": [best_fact],
            "success": True,
            "personality_applied": personality
        }
    
    def _semantic_search_facts(self, query: str, user_profile_id: str, 
                             personality: str, max_facts: int, session_id: str = None) -> Dict[str, Any]:
        """Perform semantic search on facts."""
        
        try:
            # Use existing semantic search
            # Note: session_id is not used in semantic search currently
            results = self.memory_log.semantic_search(
                query, 
                topk=max_facts, 
                user_profile_id=user_profile_id
            )
            
            if not results:
                # Try a simpler text-based search as fallback
                all_facts = self.memory_log.get_all_facts()
                user_facts = [f for f in all_facts if getattr(f, 'user_profile_id', None) == user_profile_id or (hasattr(f, 'subject') and f.subject.lower() == 'user')]
                
                # Simple text matching
                query_words = set(query.lower().split())
                matching_facts = []
                for fact in user_facts:
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                    if any(word in fact_text for word in query_words if len(word) > 2):
                        matching_facts.append({
                            'subject': fact.subject,
                            'predicate': fact.predicate, 
                            'object': fact.object,
                            'confidence': getattr(fact, 'confidence', 1.0),
                            'id': fact.id
                        })
                
                results = matching_facts[:max_facts]
            
            if not results:
                return {
                    "type": "fact_retrieval",
                    "response": generate_uncertainty_response(query, "no_memory"),
                    "facts": [],
                    "success": True
                }
            
            # Apply personality adjustments using the personality operations module
            from cortex.personality_ops import apply_personality
            results = apply_personality(results, personality)
            
            # Format response
            if len(results) == 1:
                result = results[0]
                response = f"Found 1 fact: You {result['predicate']} {result['object']} (confidence: {result.get('confidence', 1.0):.2f})"
            else:
                items = []
                for result in results[:3]:
                    items.append(f"You {result['predicate']} {result['object']} (confidence: {result.get('confidence', 1.0):.2f})")
                response = f"Found {len(results)} facts: " + "; ".join(items)
            
            return {
                "type": "fact_retrieval",
                "response": response,
                "facts": results,
                "success": True,
                "personality_applied": personality
            }
            
        except Exception as e:
            logging.error(f"Error in semantic search: {e}")
            # Fallback to simple text search
            try:
                all_facts = self.memory_log.get_all_facts()
                user_facts = [f for f in all_facts if getattr(f, 'user_profile_id', None) == user_profile_id or (hasattr(f, 'subject') and f.subject.lower() == 'user')]
                
                query_words = set(query.lower().split())
                matching_facts = []
                for fact in user_facts:
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                    if any(word in fact_text for word in query_words if len(word) > 2):
                        matching_facts.append({
                            'subject': fact.subject,
                            'predicate': fact.predicate, 
                            'object': fact.object,
                            'confidence': getattr(fact, 'confidence', 1.0),
                            'id': fact.id
                        })
                
                if matching_facts:
                    if len(matching_facts) == 1:
                        result = matching_facts[0]
                        response = f"Found 1 fact: You {result['predicate']} {result['object']} (confidence: {result.get('confidence', 1.0):.2f})"
                    else:
                        items = []
                        for result in matching_facts[:3]:
                            items.append(f"You {result['predicate']} {result['object']} (confidence: {result.get('confidence', 1.0):.2f})")
                        response = f"Found {len(matching_facts)} facts from fallback: " + "; ".join(items)
                    return {
                        "type": "fact_retrieval",
                        "response": response,
                        "facts": matching_facts[:max_facts],
                        "success": True,
                        "fallback_used": True
                    }
            except Exception as fallback_error:
                logging.error(f"Fallback search also failed: {fallback_error}")
            
            return {
                "type": "fact_retrieval",
                "response": "I had trouble searching my memory. Could you rephrase your question?",
                "facts": [],
                "success": False,
                "error": str(e)
            }
    
    def _handle_command(self, query: str) -> Dict[str, Any]:
        """Handle command-type queries."""
        
        return {
            "type": "command",
            "response": f"Command detected: {query}",
            "command": query.lower(),
            "success": True
        }
    
    def _handle_general_query(self, query: str, user_profile_id: str, 
                            personality: str, max_facts: int, session_id: str = None) -> Dict[str, Any]:
        """Handle general queries that don't fit other categories."""
        
        # Use semantic search as fallback
        return self._semantic_search_facts(query, user_profile_id, personality, max_facts, session_id)
    
    def _handle_comparison_query(self, query: str, user_profile_id: str, personality: str) -> Dict[str, Any]:
        """Handle comparison queries like 'do i like the color blue or do i love it'"""
        
        # Extract the object being asked about
        query_lower = query.lower()
        
        # Try to extract what's being compared
        # Pattern: "do i [predicate1] [object] or do i [predicate2] it"
        import re
        pattern = r"do i (\w+) (.+?) or do i (\w+)"
        match = re.search(pattern, query_lower)
        
        if not match:
            return self._semantic_search_facts(query, user_profile_id, personality, 10)
        
        pred1 = match.group(1)
        object_text = match.group(2)
        pred2 = match.group(3)
        
        # Look for facts about this object
        all_facts = self.memory_log.get_all_facts()
        matching_facts = []
        
        for fact in all_facts:
            if (fact.subject.lower() in ['user', 'i'] and 
                object_text in fact.object.lower()):
                matching_facts.append(fact)
        
        if not matching_facts:
            return {
                "type": "fact_retrieval",
                "response": f"I don't have any information about your feelings toward {object_text}.",
                "facts": [],
                "success": True
            }
        
        # Check what predicate is actually stored
        response_parts = []
        for fact in matching_facts:
            pred = fact.predicate.lower()
            conf = getattr(fact, 'confidence', 1.0)
            if pred in [pred1, pred2]:
                response_parts.append(f"You {pred} {fact.object} (confidence: {conf:.2f})")
            else:
                response_parts.append(f"Actually, you {pred} {fact.object} (confidence: {conf:.2f})")
        
        response = ". ".join(response_parts) if response_parts else f"I found information about {object_text} but with different feelings."
        
        return {
            "type": "fact_retrieval",
            "response": response,
            "facts": matching_facts[:3],
            "success": True,
            "personality_applied": personality
        } 