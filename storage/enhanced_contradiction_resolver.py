"""
Contradiction Resolution and Volatility Management for MeRNSTA
Handles detection, resolution, and tracking of belief changes over time
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import spacy
from nltk.corpus import wordnet as wn

from storage.enhanced_memory_model import EnhancedTripletFact, ContradictionRecord


class ContradictionResolver:
    """
    Manages contradictions and volatility in the memory system.
    Tracks belief changes over time and identifies volatile topics.
    """
    
    def __init__(self, volatility_threshold: float = 0.4):
        self.volatility_threshold = volatility_threshold
        self.contradiction_history: List[ContradictionRecord] = []
        try:
            from config.settings import get_config
            config = get_config()
            spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logging.warning("spaCy model not found, using basic fallback for contradiction detection")
            self.nlp = None
        
    def are_antonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are antonyms using WordNet and semantic rules."""
        # First check common antonym pairs that we know about
        common_antonyms = {
            ('like', 'dislike'), ('like', 'hate'), ('love', 'hate'), 
            ('enjoy', 'dislike'), ('prefer', 'avoid'), ('want', 'avoid'),
            ('good', 'bad'), ('happy', 'sad'), ('hot', 'cold')
        }
        
        pair = (word1.lower(), word2.lower())
        reverse_pair = (word2.lower(), word1.lower())
        
        if pair in common_antonyms or reverse_pair in common_antonyms:
            return True
        
        # Check for negation patterns
        if (word1.startswith('not_') and word1[4:] == word2) or (word2.startswith('not_') and word2[4:] == word1):
            return True
            
        # Check WordNet antonyms
        try:
            antonyms = set()
            for syn in wn.synsets(word1):
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name())
            return word2 in antonyms
        except:
            return False

    def _objects_similar(self, obj1: str, obj2: str, threshold: float = 0.85) -> bool:
        """Check if two objects are semantically similar using spaCy."""
        # Handle exact matches and substring matches first
        obj1_norm = obj1.lower().strip()
        obj2_norm = obj2.lower().strip()
        
        if obj1_norm == obj2_norm:
            return True
            
        if obj1_norm in obj2_norm or obj2_norm in obj1_norm:
            return True
            
        # Use spaCy similarity if available
        if self.nlp:
            try:
                doc1 = self.nlp(obj1_norm)
                doc2 = self.nlp(obj2_norm)
                sim = doc1.similarity(doc2)
                print(f"[ContradictionCheck] Object similarity: '{obj1}' vs '{obj2}' = {sim}")
                return sim > threshold
            except Exception as e:
                logging.warning(f"spaCy similarity failed: {e}")
        
        # Fallback to word overlap
        words1 = set(obj1_norm.split())
        words2 = set(obj2_norm.split())
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = overlap / union if union > 0 else 0.0
        return similarity > 0.7

    def _predicates_conflict(self, pred1: str, neg1: bool, pred2: str, neg2: bool) -> bool:
        """Check if two predicates conflict, considering negation."""
        pred1_norm = pred1.lower().strip()
        pred2_norm = pred2.lower().strip()
        
        # Direct negation check
        if pred1_norm == pred2_norm and neg1 != neg2:
            return True
            
        # Check for antonyms
        if self.are_antonyms(pred1_norm, pred2_norm):
            return True
            
        # Handle special cases for like/dislike
        positive_predicates = {'like', 'love', 'enjoy', 'prefer', 'want'}
        negative_predicates = {'dislike', 'hate', 'despise', 'avoid', 'detest'}
        
        # If one is positive and other is negative, they conflict
        if ((pred1_norm in positive_predicates and pred2_norm in negative_predicates) or
            (pred1_norm in negative_predicates and pred2_norm in positive_predicates)):
            return True
            
        return False

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for consistent comparison."""
        if not text:
            return ""
        return text.lower().strip()

    def check_for_contradictions(self, new_fact: EnhancedTripletFact, existing_facts: List[EnhancedTripletFact]) -> List[ContradictionRecord]:
        """Enhanced contradiction detection using semantic similarity, antonym detection, and token-based analysis."""
        contradictions = []
        print(f"[ContradictionCheck] New fact: ({new_fact.subject}, {new_fact.predicate}, {new_fact.object})")
        
        for existing_fact in existing_facts:
            # Skip if different user/session
            if (existing_fact.user_profile_id != new_fact.user_profile_id or
                existing_fact.session_id != new_fact.session_id):
                continue
            
            # Normalize for comparison
            subj1 = self._normalize_for_comparison(existing_fact.subject)
            subj2 = self._normalize_for_comparison(new_fact.subject)
            
            # Only compare facts about the same subject
            if subj1 != subj2:
                continue
                
            obj1 = self._normalize_for_comparison(existing_fact.object)
            obj2 = self._normalize_for_comparison(new_fact.object)
            pred1 = self._normalize_for_comparison(existing_fact.predicate)
            pred2 = self._normalize_for_comparison(new_fact.predicate)
            
            neg1 = getattr(existing_fact, 'negation', False)
            neg2 = getattr(new_fact, 'negation', False)
            
            print(f"[ContradictionCheck] Compare: ({subj1}, {pred1}, {obj1}, neg={neg1}) vs ({subj2}, {pred2}, {obj2}, neg={neg2})")
            
            # TOKEN-BASED CONTRADICTION DETECTION
            token_contradiction_score = self._check_token_based_contradiction(existing_fact, new_fact)
            if token_contradiction_score > 0.8:
                print(f"[ContradictionCheck] TOKEN-BASED CONTRADICTION: score={token_contradiction_score:.3f}")
                contradiction = ContradictionRecord(
                    id=None,
                    fact_a_id=existing_fact.id,
                    fact_b_id=new_fact.id,
                    confidence=token_contradiction_score
                )
                contradictions.append(contradiction)
                self._mark_facts_contradictory(existing_fact, new_fact)
                continue
            
            # Check if objects are semantically similar (e.g., "pineapple on pizza" vs "pineapple pizza")
            if self._objects_similar(obj1, obj2):
                # Check if predicates conflict
                if self._predicates_conflict(pred1, neg1, pred2, neg2):
                    print(f"[ContradictionCheck] CONTRADICTION: predicate conflict for same (subject, object)")
                    contradiction = ContradictionRecord(
                        id=None,
                        fact_a_id=existing_fact.id,
                        fact_b_id=new_fact.id,
                        confidence=self._calculate_contradiction_confidence(existing_fact, new_fact)
                    )
                    contradictions.append(contradiction)
                    self._mark_facts_contradictory(existing_fact, new_fact)
            
            # ENHANCED: Check for preference conflicts (e.g., "prefer tea" vs "prefer coffee")
            elif self._preference_conflict(pred1, obj1, pred2, obj2):
                print(f"[ContradictionCheck] PREFERENCE CONTRADICTION: {pred1} {obj1} vs {pred2} {obj2}")
                contradiction = ContradictionRecord(
                    id=None,
                    fact_a_id=existing_fact.id,
                    fact_b_id=new_fact.id,
                    confidence=0.8  # High confidence for preference conflicts
                )
                contradictions.append(contradiction)
                self._mark_facts_contradictory(existing_fact, new_fact)
        
        print(f"[ContradictionCheck] Contradictions found: {len(contradictions)}")
        return contradictions
    
    def _check_token_based_contradiction(self, fact_a: EnhancedTripletFact, fact_b: EnhancedTripletFact) -> float:
        """
        Check for contradictions based on token analysis.
        
        Args:
            fact_a: First fact to compare
            fact_b: Second fact to compare
            
        Returns:
            Contradiction score (0.0 to 1.0, higher = more likely contradiction)
        """
        if not hasattr(fact_a, 'token_ids') or not hasattr(fact_b, 'token_ids'):
            return 0.0
        
        if not fact_a.token_ids or not fact_b.token_ids:
            return 0.0
        
        # Calculate token similarities
        token_jaccard = fact_a.get_token_jaccard_similarity(fact_b)
        token_overlap = fact_a.get_token_overlap_ratio(fact_b)
        
        print(f"[TokenContradiction] Jaccard: {token_jaccard:.3f}, Overlap: {token_overlap:.3f}")
        
        # High similarity + conflicting predicates = likely contradiction
        if token_jaccard > 0.7 and token_overlap > 0.6:
            # Check if predicates conflict
            pred1 = fact_a.predicate.lower().strip()
            pred2 = fact_b.predicate.lower().strip()
            
            if self._predicates_conflict(pred1, fact_a.negation, pred2, fact_b.negation):
                # High token similarity + predicate conflict = strong contradiction
                contradiction_score = min(0.9, token_jaccard * 1.2)
                print(f"[TokenContradiction] High similarity + predicate conflict: {contradiction_score:.3f}")
                return contradiction_score
        
        # Zero overlap + conflicting predicates = possible contradiction
        elif token_jaccard < 0.1 and token_overlap < 0.1:
            pred1 = fact_a.predicate.lower().strip()
            pred2 = fact_b.predicate.lower().strip()
            
            if self._predicates_conflict(pred1, fact_a.negation, pred2, fact_b.negation):
                # Low token similarity + predicate conflict = moderate contradiction
                contradiction_score = 0.6
                print(f"[TokenContradiction] Low similarity + predicate conflict: {contradiction_score:.3f}")
                return contradiction_score
        
        # Moderate similarity + same subject = potential paraphrase (not contradiction)
        elif token_jaccard > 0.3 and token_overlap > 0.4:
            print(f"[TokenContradiction] Moderate similarity - likely paraphrase, not contradiction")
            return 0.0
        
        return 0.0
    
    def _mark_facts_contradictory(self, fact_a: EnhancedTripletFact, fact_b: EnhancedTripletFact):
        """Mark two facts as contradictory."""
        fact_a.contradiction = True
        fact_b.contradiction = True
        
        if hasattr(fact_a, 'contradicts_with'):
            fact_a.contradicts_with.append(fact_b.id)
        else:
            fact_a.contradicts_with = [fact_b.id]
            
        if hasattr(fact_b, 'contradicts_with'):
            fact_b.contradicts_with.append(fact_a.id)
        else:
            fact_b.contradicts_with = [fact_a.id]

    def _preference_conflict(self, pred1: str, obj1: str, pred2: str, obj2: str) -> bool:
        """
        Dynamic preference conflict detection using semantic similarity and memory patterns.
        No hardcoded categories - uses embeddings and semantic understanding.
        """
        pred1_norm = pred1.lower().strip()
        pred2_norm = pred2.lower().strip()
        
        # Both must be preference predicates
        preference_predicates = {'prefer', 'like', 'love', 'enjoy', 'want', 'choose'}
        if pred1_norm not in preference_predicates or pred2_norm not in preference_predicates:
            return False
        
        obj1_norm = obj1.lower().strip()
        obj2_norm = obj2.lower().strip()
        
        if obj1_norm == obj2_norm:
            return False  # Same object, not conflicting
        
        # DYNAMIC APPROACH 1: Use semantic similarity to detect related objects
        if self.nlp:
            try:
                doc1 = self.nlp(obj1_norm)
                doc2 = self.nlp(obj2_norm)
                
                # If objects are semantically similar but different, they might be conflicting preferences
                similarity = doc1.similarity(doc2)
                if 0.3 < similarity < 0.9:  # Related but not identical
                    print(f"[PreferenceConflict] Semantic similarity between '{obj1_norm}' and '{obj2_norm}': {similarity:.3f}")
                    return True
            except Exception as e:
                print(f"[PreferenceConflict] spaCy similarity failed: {e}")
        
        # DYNAMIC APPROACH 2: Check if they're both simple, different objects with same predicate
        if pred1_norm == pred2_norm == "prefer":
            # For "prefer X" vs "prefer Y", if both are simple objects, likely conflicting
            obj1_words = obj1_norm.split()
            obj2_words = obj2_norm.split()
            
            if (len(obj1_words) <= 2 and len(obj2_words) <= 2 and 
                obj1_norm != obj2_norm):
                print(f"[PreferenceConflict] Simple preference conflict: '{pred1_norm} {obj1_norm}' vs '{pred2_norm} {obj2_norm}'")
                return True
        
        # DYNAMIC APPROACH 3: Learn from memory patterns
        # TODO: Could analyze past contradictions to learn new conflict patterns
        # This would make the system truly self-adaptive over time
        
        return False

    def _calculate_contradiction_confidence(self, fact1: EnhancedTripletFact, fact2: EnhancedTripletFact) -> float:
        """Calculate confidence score for contradiction detection."""
        base_confidence = 0.8
        
        # Higher confidence if objects are exact matches
        if fact1.object.lower() == fact2.object.lower():
            base_confidence = 0.95
            
        # Higher confidence if predicates are known antonyms
        if self.are_antonyms(fact1.predicate, fact2.predicate):
            base_confidence = 0.9
            
        return base_confidence

    def get_volatile_topics(self, facts: List[EnhancedTripletFact]) -> List[Tuple[str, str, float]]:
        """Identify volatile topics where user has contradictory facts."""
        volatile_topics = []
        
        # Group facts by (subject, object) pairs using semantic similarity
        topic_groups = defaultdict(list)
        
        for fact in facts:
            # Find or create a group for this fact
            group_key = None
            for existing_key in topic_groups.keys():
                existing_subj, existing_obj = existing_key
                if (self._normalize_for_comparison(fact.subject) == self._normalize_for_comparison(existing_subj) and
                    self._objects_similar(fact.object, existing_obj)):
                    group_key = existing_key
                    break
            
            if group_key is None:
                group_key = (fact.subject, fact.object)
            
            topic_groups[group_key].append(fact)
        
        # Check each group for volatility (2+ contradictory facts)
        for (subject, obj), group_facts in topic_groups.items():
            contradictory_facts = [f for f in group_facts if getattr(f, 'contradiction', False)]
            
            if len(contradictory_facts) >= 2:
                # Calculate volatility score based on contradiction count and time spread
                volatility_score = min(1.0, len(contradictory_facts) / 5.0)  # Max out at 5 contradictions
                
                # Mark all facts in this topic as volatile
                for fact in group_facts:
                    fact.volatile = True
                    fact.volatility_score = volatility_score
                
                volatile_topics.append((subject, 'opinions about', volatility_score))
        
        return volatile_topics

    def suggest_clarification_questions(self, volatile_topics: List[Tuple[str, str, float]]) -> List[str]:
        """Generate clarification questions for volatile topics."""
        questions = []
        
        for subject, predicate, volatility in volatile_topics:
            if volatility > 0.7:
                questions.append(f"You've changed your mind about {subject} several times. What's your current feeling?")
            else:
                questions.append(f"I notice you have conflicting opinions about {subject}. Could you clarify?")
        
        return questions 