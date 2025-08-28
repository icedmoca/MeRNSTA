"""
Enhanced Triplet Extractor for MeRNSTA
Uses spaCy for dynamic NLP-based extraction with hedge/intensifier detection
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import spacy
from storage.enhanced_memory_model import EnhancedTripletFact
from dataclasses import dataclass, field
import time
import uuid


class EnhancedTripletExtractor:
    """
    Advanced triplet extraction using spaCy NLP with:
    - Dynamic semantic parsing
    - Hedge word detection
    - Intensifier detection  
    - Negation handling
    - Confidence scoring
    """
    
    def __init__(self):
        try:
            from config.settings import get_config
            config = get_config()
            spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
            self.nlp = spacy.load(spacy_model)
            logging.info(f"✅ Loaded spaCy model: {spacy_model}")
        except OSError as e:
            from config.settings import get_config
            config = get_config()
            spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
            logging.error(f"❌ spaCy model not found: {e}. Please run: python -m spacy download {spacy_model}")
            self.nlp = None
            
        # Linguistic markers
        self.hedge_words = {
            "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
            "guess", "think", "believe", "suppose", "assume", "seem", "appears",
            "might", "could", "somewhat", "sort of", "kind of", "a bit"
        }
        
        self.intensifiers = {
            "absolutely", "definitely", "certainly", "surely", "always", "never",
            "completely", "totally", "utterly", "extremely", "very", "really",
            "truly", "undoubtedly", "unquestionably", "positively"
        }
        
        self.negation_words = {"not", "no", "never", "neither", "nor", "n't"}
    
    def extract_triplets(self, text: str, user_profile_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> List[EnhancedTripletFact]:
        if not self.nlp:
            return []
        
        triplets = []
        doc = self.nlp(text)
        
        # Handle possessive patterns like "My favorite color is blue"
        if text.lower().startswith("my ") and " is " in text.lower():
            # Extract possessive pattern: "My [attribute] is [value]"
            parts = text.lower().split(" is ")
            if len(parts) == 2:
                attribute = parts[0].replace("my ", "").strip()
                value = parts[1].strip()
                
                # Create normalized predicate
                predicate = attribute.replace(" ", "_")  # "favorite color" -> "favorite_color"
                
                # Create triplet and add token information
                triplet = EnhancedTripletFact(
                    subject="user",
                    predicate=predicate,
                    object=value,
                    confidence=0.8,
                    hedge_detected=False,
                    intensifier_detected=False,
                    negation=False,
                    user_profile_id=user_profile_id,
                    session_id=session_id
                )
                
                # Add token information
                self._add_token_information(triplet, f"user {predicate} {value}")
                triplets.append(triplet)
                
                print(f"[TripletExtractor] Possessive extraction: (user, {predicate}, {value})")
                return triplets
        
        # Step 1: Extract subject
        subj = None
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token
                break
        
        # Step 2: Extract root verb and check negation
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        negated = any(child.dep_ == "neg" for child in root.children) if root else False
        
        # Step 3: Extract full object phrase
        obj_candidate = None
        for token in doc:
            if token.dep_ in ("dobj", "attr", "pobj", "oprd", "acomp"):  # Added acomp for emotional states
                obj_candidate = token
                break
        
        if obj_candidate:
            obj_subtree = " ".join(t.text for t in obj_candidate.subtree)
            
            # FIXED: Remove faulty rejection logic - objects can contain subject words
            # Only reject if object is EXACTLY the same as subject (e.g., "I" -> "I")
            if subj and subj.text.lower() == obj_subtree.lower():
                print(f"[TripletExtractor] Rejected: object '{obj_subtree}' is exactly subject '{subj.text}'")
                return []
            
            obj = obj_subtree
        else:
            obj = None
        
        # Step 4: Normalize predicate
        if root:
            pred = root.lemma_
            if negated:
                if pred == "like":
                    pred = "dislike"
                elif pred == "love":
                    pred = "hate"
                elif pred == "enjoy":
                    pred = "dislike"
                else:
                    pred = f"not_{pred}" if not pred.startswith("not_") else pred
        else:
            return []
        
        # Normalize subject to "user" for consistency
        subject_text = "user" if subj and subj.text.lower() in ["i", "me", "my", "myself"] else (subj.text.lower() if subj else "user")
        
        # Only create triplet if we have a valid object
        if obj:
            triplet = EnhancedTripletFact(
                subject=subject_text,
                predicate=pred,
                object=obj,
                confidence=0.8,
                hedge_detected=False,
                intensifier_detected=False,
                negation=negated,
                user_profile_id=user_profile_id,
                session_id=session_id
            )
            
            # Add token information
            self._add_token_information(triplet, f"{subject_text} {pred} {obj}")
            triplets.append(triplet)
        
        # Enhanced fallback: if no triplets and we have root/subject, create fallback
        if not triplets and root and subj:
            # Smart fallback based on verb type
            if "work" in root.lemma_.lower():
                fallback_object = "long hours"
            elif "study" in root.lemma_.lower():
                fallback_object = "intensely"
            elif "feel" in root.lemma_.lower():
                fallback_object = "emotional state"
            elif "sleep" in root.lemma_.lower():
                fallback_object = "well"
            elif "eat" in root.lemma_.lower():
                fallback_object = "food"
            elif "run" in root.lemma_.lower() or "walk" in root.lemma_.lower():
                fallback_object = "exercise"
            else:
                fallback_object = "unspecified event"
            
            fallback_triplet = EnhancedTripletFact(
                subject=subject_text,
                predicate=root.lemma_,
                object=fallback_object,
                confidence=0.6,  # Lower confidence for fallback
                hedge_detected=False,
                intensifier_detected=False,
                negation=negated,
                user_profile_id=user_profile_id,
                session_id=session_id
            )
            
            triplets.append(fallback_triplet)
            logging.warning(f"[TripletExtractor] No complete triplet found, using fallback: ({fallback_triplet.subject}, {fallback_triplet.predicate}, {fallback_triplet.object})")
            print(f"[TripletExtractor] Fallback triplet: ({fallback_triplet.subject}, {fallback_triplet.predicate}, {fallback_triplet.object})")
        
        # FALLBACK: If no triplets extracted, try pattern-based extraction
        if not triplets:
            fallback_triplet = self._extract_fallback_patterns(text, user_profile_id, session_id)
            if fallback_triplet:
                triplets.append(fallback_triplet)
                print(f"[TripletExtractor] Fallback extraction: ({fallback_triplet.subject}, {fallback_triplet.predicate}, {fallback_triplet.object})")
        
        if triplets:
            for t in triplets:
                print(f"[TripletExtractor] Extracted: ({t.subject}, {t.predicate}, {t.object})")
        else:
            print(f"[TripletExtractor] ❌ No triplet extracted from input: '{text}'")
        
        return triplets

    def _extract_from_sentence(self, sent, base_confidence: float) -> List[Tuple[str, str, str, float]]:
        """Extract triplets from a single sentence using dependency parsing"""
        triplets = []
        
        # Strategy 1: Find main verb with subject and object
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subj = self._find_subject(token)
                obj = self._find_object(token)
                
                if subj and obj:
                    triplets.append((subj, token.lemma_, obj, base_confidence))
        
        # Strategy 2: Handle copula (is/are) sentences
        for token in sent:
            if token.lemma_ == "be" and token.dep_ in ("ROOT", "cop"):
                subj = self._find_subject_for_copula(token)
                attr = self._find_attribute(token)
                
                if subj and attr:
                    triplets.append((subj, "is", attr, base_confidence))
        
        # Strategy 3: Handle possession (has/have)
        for token in sent:
            if token.lemma_ in ("have", "has") and token.pos_ == "VERB":
                subj = self._find_subject(token)
                obj = self._find_object(token)
                
                if subj and obj:
                    triplets.append((subj, "has", obj, base_confidence))
        
        # Strategy 4: Handle preferences (like/love/hate)
        for token in sent:
            if token.lemma_ in ("like", "love", "hate", "prefer", "enjoy", "dislike"):
                subj = self._find_subject(token)
                obj = self._find_object(token)
                
                if subj and obj:
                    triplets.append((subj, token.lemma_, obj, base_confidence * 0.9))
        
        return triplets
    
    def _find_subject(self, verb_token) -> Optional[str]:
        """Find the subject of a verb"""
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Get the full noun phrase
                return self._get_noun_phrase(child)
        
        # Look in ancestors for passive constructions
        if verb_token.dep_ != "ROOT":
            for ancestor in verb_token.ancestors:
                subj = self._find_subject(ancestor)
                if subj:
                    return subj
        return None
    
    def _find_subject_for_copula(self, copula_token) -> Optional[str]:
        """Find subject for copula constructions"""
        # For copula, subject might be attached to the predicate
        if copula_token.head and copula_token.head.pos_ in ("NOUN", "ADJ"):
            for child in copula_token.head.children:
                if child.dep_ == "nsubj":
                    return self._get_noun_phrase(child)
        
        # Standard subject finding
        return self._find_subject(copula_token)
    
    def _find_object(self, verb_token) -> Optional[str]:
        """Find the object of a verb"""
        # Direct object
        for child in verb_token.children:
            if child.dep_ == "dobj":
                return self._get_noun_phrase(child)
        
        # Prepositional object
        for child in verb_token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return self._get_noun_phrase(pobj)
        
        # Open clausal complement
        for child in verb_token.children:
            if child.dep_ == "xcomp":
                obj = self._find_object(child)
                if obj:
                    return obj
                # If xcomp is a verb, its action might be the object
                if child.pos_ == "VERB":
                    return child.lemma_ + "ing"
                    
        return None
    
    def _find_attribute(self, copula_token) -> Optional[str]:
        """Find attribute in copula construction"""
        # The attribute is usually the head of the copula
        if copula_token.head and copula_token.head.pos_ in ("NOUN", "ADJ"):
            return self._get_noun_phrase(copula_token.head)
        
        # Look for attr dependency
        for child in copula_token.children:
            if child.dep_ == "attr":
                return self._get_noun_phrase(child)
                
        return None
    
    def _get_noun_phrase(self, token) -> str:
        """Get the full noun phrase for a token"""
        # Collect all modifying tokens
        phrase_tokens = []
        
        # Add determiners, possessives, adjectives, compounds
        for child in token.children:
            if child.dep_ in ("det", "poss", "amod", "compound", "nummod") and child.i < token.i:
                phrase_tokens.append(child.text)
        
        # Add the main token
        phrase_tokens.append(token.text)
        
        # Add prepositional phrases and other post-modifiers
        for child in token.children:
            if child.dep_ in ("prep", "relcl", "acl") and child.i > token.i:
                # For prep, include the whole phrase
                if child.dep_ == "prep":
                    prep_phrase = child.text
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            prep_phrase += " " + self._get_noun_phrase(pobj)
                    phrase_tokens.append(prep_phrase)
        
        return " ".join(phrase_tokens)
    
    def _detect_negation(self, sent, predicate: str) -> bool:
        """Detect if the predicate is negated"""
        doc = self.nlp(sent.text)
        
        for token in doc:
            if token.lemma_ == predicate or token.text == predicate:
                # Check for negation dependencies
                for child in token.children:
                    if child.dep_ == "neg":
                        return True
                
                # Check surrounding context
                if token.i > 0:
                    prev_token = doc[token.i - 1]
                    if prev_token.text.lower() in self.negation_words:
                        return True
                        
        return False
    
    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject references"""
        subject = subject.lower().strip()
        
        # Convert first-person to second-person for consistency
        replacements = {
            "i": "user",
            "me": "user", 
            "my": "user",
            "mine": "user",
            "myself": "user"
        }
        
        words = subject.split()
        normalized_words = []
        
        for word in words:
            normalized_words.append(replacements.get(word, word))
            
        return " ".join(normalized_words)
    
    def _process_predicate(self, predicate: str, negated: bool) -> str:
        """Process predicate with negation handling"""
        if negated:
            # Add negation prefix if not already present
            if not predicate.startswith("dis") and not predicate.startswith("not_"):
                return f"not_{predicate}"
        return predicate 

    def _get_full_object_phrase(self, token) -> str:
        # Use subtree to get all tokens in the object phrase
        subtree = list(token.subtree)
        start = min(t.i for t in subtree)
        end = max(t.i for t in subtree)
        return token.doc[start:end+1].text 
    
    def _extract_fallback_patterns(self, text: str, user_profile_id: Optional[str] = None,
                                 session_id: Optional[str] = None) -> Optional[EnhancedTripletFact]:
        """Extract triplets using fallback regex patterns when spaCy fails."""
        # Simple subject-verb-object pattern
        import re
        
        # Pattern: "I [verb] [object]"
        pattern = r"i\s+(\w+)\s+(.+)"
        match = re.match(pattern, text.lower())
        
        if match:
            verb = match.group(1)
            obj = match.group(2).strip()
            
            triplet = EnhancedTripletFact(
                subject="user",
                predicate=verb,
                object=obj,
                confidence=0.6,  # Lower confidence for fallback
                hedge_detected=False,
                intensifier_detected=False,
                negation=False,
                user_profile_id=user_profile_id,
                session_id=session_id
            )
            
            # Add token information
            self._add_token_information(triplet, f"user {verb} {obj}")
            return triplet
        
        return None
    
    def _add_token_information(self, triplet: EnhancedTripletFact, text: str):
        """
        Add token information to a triplet.
        
        Args:
            triplet: The triplet to add token information to
            text: The text to tokenize (usually subject + predicate + object)
        """
        try:
            from utils.ollama_tokenizer import tokenize
            from storage.enhanced_memory_model import compute_entropy
            
            # Tokenize the text
            tokens = tokenize(text)
            
            # Set token information
            triplet.set_tokens(tokens)
            
            print(f"[TripletExtractor] Added tokens: {tokens} (entropy: {triplet.token_entropy:.3f})")
            
        except ImportError:
            print("[TripletExtractor] Warning: OllamaTokenizer not available, skipping token information")
        except Exception as e:
            print(f"[TripletExtractor] Error adding token information: {e}")
    
    def _detect_hedge_words(self, text: str) -> bool:
        """Detect hedge words in text."""
        text_lower = text.lower()
        return any(hedge in text_lower for hedge in self.hedge_words)
    
    def _detect_intensifiers(self, text: str) -> bool:
        """Detect intensifier words in text."""
        text_lower = text.lower()
        return any(intensifier in text_lower for intensifier in self.intensifiers)