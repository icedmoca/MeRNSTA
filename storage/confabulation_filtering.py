#!/usr/bin/env python3
"""
🛡️ Confabulation Filtering System for MeRNSTA

Use contradiction history and confidence to filter fabricated answers — i.e., suppress 
hallucinations in live chat. Analyzes response reliability based on memory consistency
and prevents the system from confidently stating unverified information.
"""

import logging
import time
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

try:
    import spacy
    from config.settings import get_config
    config = get_config()
    spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
    nlp = spacy.load(spacy_model)
except (ImportError, OSError):
    nlp = None
    logging.warning("spaCy not available for confabulation filtering")

from .enhanced_triplet_extractor import EnhancedTripletFact


@dataclass
class ResponseAssessment:
    """Assessment of a response's reliability and potential for confabulation."""
    response_text: str
    confidence_score: float  # Overall confidence in response accuracy
    confabulation_risk: float  # Risk of containing fabricated information
    supporting_facts: List[EnhancedTripletFact]
    contradicting_facts: List[EnhancedTripletFact]
    uncertainty_indicators: List[str]
    reliability_factors: Dict[str, float]
    filtered_response: str  # Response after filtering
    action_taken: str  # "approved", "hedged", "rejected", "clarified"


class ConfabulationFilteringSystem:
    """
    Analyzes response reliability and filters out potential hallucinations.
    Uses memory consistency, confidence scores, and contradiction patterns.
    """
    
    def __init__(self):
        from config.settings import DEFAULT_VALUES, CONFIDENCE_THRESHOLDS
        
        self.confidence_threshold = CONFIDENCE_THRESHOLDS.get('medium', 0.6)  # Minimum confidence for direct answers
        self.confabulation_threshold = DEFAULT_VALUES.get('confabulation_threshold', 0.7)  # Maximum acceptable confabulation risk
        self.contradiction_penalty = 0.3  # How much contradictions reduce confidence
        self.uncertainty_phrases = [
            "I think", "I believe", "It seems", "Perhaps", "Possibly", "Maybe",
            "I'm not sure", "It appears", "Likely", "Probably", "It might be"
        ]
        
    def assess_response_reliability(self, response: str, query: str, 
                                  available_facts: List[EnhancedTripletFact],
                                  contradiction_history: Dict = None) -> ResponseAssessment:
        """
        Assess the reliability of a response and identify confabulation risks.
        """
        print(f"[ConfabulationFilter] Assessing response: '{response[:100]}...'")
        
        # Extract claims from response
        response_claims = self._extract_response_claims(response)
        
        # Find supporting and contradicting facts
        supporting_facts, contradicting_facts = self._analyze_factual_support(
            response_claims, available_facts
        )
        
        # Calculate confidence and confabulation risk
        reliability_factors = self._calculate_reliability_factors(
            response, response_claims, supporting_facts, contradicting_facts, contradiction_history
        )
        
        confidence_score = self._calculate_confidence_score(reliability_factors)
        confabulation_risk = self._calculate_confabulation_risk(reliability_factors)
        
        # Identify uncertainty indicators
        uncertainty_indicators = self._identify_uncertainty_indicators(response)
        
        # Determine action and create filtered response
        action_taken, filtered_response = self._determine_filtering_action(
            response, confidence_score, confabulation_risk, supporting_facts, contradicting_facts
        )
        
        assessment = ResponseAssessment(
            response_text=response,
            confidence_score=confidence_score,
            confabulation_risk=confabulation_risk,
            supporting_facts=supporting_facts,
            contradicting_facts=contradicting_facts,
            uncertainty_indicators=uncertainty_indicators,
            reliability_factors=reliability_factors,
            filtered_response=filtered_response,
            action_taken=action_taken
        )
        
        print(f"[ConfabulationFilter] Assessment: confidence={confidence_score:.2f}, "
              f"confabulation_risk={confabulation_risk:.2f}, action={action_taken}")
        
        return assessment
    
    def _extract_response_claims(self, response: str) -> List[str]:
        """Extract factual claims from a response."""
        claims = []
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
                
            # Filter out questions, commands, and meta-statements
            if any(sentence.lower().startswith(prefix) for prefix in [
                'what', 'why', 'how', 'when', 'where', 'who',
                'do you', 'can you', 'would you', 'should you',
                'let me', 'i need to', 'i should', 'i will'
            ]):
                continue
            
            # Filter out uncertainty expressions
            if any(uncertain in sentence.lower() for uncertain in [
                "i don't know", "i'm not sure", "unclear", "uncertain"
            ]):
                continue
            
            claims.append(sentence)
        
        return claims
    
    def _analyze_factual_support(self, claims: List[str], facts: List[EnhancedTripletFact]) -> Tuple[List[EnhancedTripletFact], List[EnhancedTripletFact]]:
        """Analyze which facts support or contradict the response claims."""
        supporting_facts = []
        contradicting_facts = []
        
        for claim in claims:
            claim_lower = claim.lower()
            
            for fact in facts:
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                
                # Check for semantic overlap
                if nlp:
                    try:
                        claim_doc = nlp(claim_lower)
                        fact_doc = nlp(fact_text)
                        
                        similarity = claim_doc.similarity(fact_doc)
                        
                        if similarity > 0.6:  # High similarity suggests support
                            if not any(f.id == fact.id for f in supporting_facts):
                                supporting_facts.append(fact)
                        elif similarity > 0.4:  # Medium similarity - check for contradiction
                            if self._claim_contradicts_fact(claim_lower, fact):
                                if not any(f.id == fact.id for f in contradicting_facts):
                                    contradicting_facts.append(fact)
                    except Exception as e:
                        print(f"[ConfabulationFilter] spaCy error: {e}")
                
                # Fallback: simple keyword matching
                claim_words = set(claim_lower.split())
                fact_words = set(fact_text.split())
                
                overlap = len(claim_words.intersection(fact_words))
                
                if overlap >= 2:  # Significant word overlap
                    # Check if this is supporting or contradicting
                    if self._claim_contradicts_fact(claim_lower, fact):
                        if not any(f.id == fact.id for f in contradicting_facts):
                            contradicting_facts.append(fact)
                    else:
                        if not any(f.id == fact.id for f in supporting_facts):
                            supporting_facts.append(fact)
        
        return supporting_facts, contradicting_facts
    
    def _claim_contradicts_fact(self, claim: str, fact: EnhancedTripletFact) -> bool:
        """Check if a claim contradicts a known fact."""
        claim_lower = claim.lower()
        predicate = fact.predicate.lower()
        obj = fact.object.lower()
        
        # Look for direct contradictions
        contradictory_patterns = [
            # Negation patterns
            (f"not {predicate}", predicate),
            (f"don't {predicate}", predicate),
            (f"doesn't {predicate}", predicate),
            (f"never {predicate}", predicate),
            
            # Antonym patterns
            ("hate", "love"), ("hate", "like"), ("hate", "enjoy"),
            ("dislike", "love"), ("dislike", "like"), ("dislike", "enjoy"),
            ("avoid", "prefer"), ("reject", "choose")
        ]
        
        for neg_pattern, pos_pattern in contradictory_patterns:
            if neg_pattern in claim_lower and pos_pattern == predicate:
                return True
            if pos_pattern in claim_lower and neg_pattern == predicate:
                return True
        
        # Check for object contradictions in preferences
        if predicate in ['prefer', 'choose', 'like', 'love'] and obj in claim_lower:
            # Look for mentions of different objects with same predicate
            preference_indicators = ['prefer', 'choose', 'like', 'love']
            for indicator in preference_indicators:
                if indicator in claim_lower and obj not in claim_lower:
                    return True
        
        return False
    
    def _calculate_reliability_factors(self, response: str, claims: List[str],
                                     supporting_facts: List[EnhancedTripletFact],
                                     contradicting_facts: List[EnhancedTripletFact],
                                     contradiction_history: Dict = None) -> Dict[str, float]:
        """Calculate various factors that affect response reliability."""
        factors = {}
        
        # 1. Factual support ratio
        total_relevant_facts = len(supporting_facts) + len(contradicting_facts)
        if total_relevant_facts > 0:
            factors['factual_support'] = len(supporting_facts) / total_relevant_facts
        else:
            factors['factual_support'] = 0.0  # No factual basis
        
        # 2. Average fact confidence
        if supporting_facts:
            avg_confidence = sum(getattr(f, 'confidence', 0.5) for f in supporting_facts) / len(supporting_facts)
            factors['fact_confidence'] = avg_confidence
        else:
            factors['fact_confidence'] = 0.0
        
        # 3. Contradiction severity
        if contradicting_facts:
            contradiction_severity = sum(getattr(f, 'confidence', 0.5) for f in contradicting_facts) / len(contradicting_facts)
            factors['contradiction_severity'] = contradiction_severity
        else:
            factors['contradiction_severity'] = 0.0
        
        # 4. Response specificity (specific responses are riskier if unsupported)
        specificity = self._calculate_response_specificity(response)
        factors['specificity'] = specificity
        
        # 5. Uncertainty language (hedging reduces confabulation risk)
        uncertainty_score = self._calculate_uncertainty_score(response)
        factors['uncertainty_hedging'] = uncertainty_score
        
        # 6. Historical contradiction rate
        if contradiction_history:
            factors['historical_reliability'] = contradiction_history.get('reliability_score', 0.5)
        else:
            factors['historical_reliability'] = 0.5
        
        # 7. Response length and complexity (longer responses more likely to contain errors)
        word_count = len(response.split())
        factors['length_complexity'] = max(0.0, 1.0 - (word_count / 100.0))  # Penalty for very long responses
        
        return factors
    
    def _calculate_response_specificity(self, response: str) -> float:
        """Calculate how specific/detailed the response is."""
        specificity_indicators = [
            r'\d+',  # Numbers
            r'\b(exactly|precisely|specifically|definitely)\b',  # Certainty words
            r'\b(at|on|in)\s+\d+',  # Dates/times
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Proper nouns
        ]
        
        specificity_score = 0.0
        response_lower = response.lower()
        
        for pattern in specificity_indicators:
            matches = len(re.findall(pattern, response_lower, re.IGNORECASE))
            specificity_score += matches * 0.2
        
        return min(1.0, specificity_score)
    
    def _calculate_uncertainty_score(self, response: str) -> float:
        """Calculate how much uncertainty language is used."""
        uncertainty_count = 0
        response_lower = response.lower()
        
        for phrase in self.uncertainty_phrases:
            uncertainty_count += response_lower.count(phrase.lower())
        
        # Normalize by response length
        word_count = len(response.split())
        return min(1.0, uncertainty_count / max(1, word_count / 10))
    
    def _calculate_confidence_score(self, factors: Dict[str, float]) -> float:
        """Calculate overall confidence score based on reliability factors."""
        weights = {
            'factual_support': 0.3,
            'fact_confidence': 0.2,
            'contradiction_severity': -0.25,  # Negative weight
            'specificity': -0.1,  # High specificity without support is risky
            'uncertainty_hedging': 0.15,
            'historical_reliability': 0.15,
            'length_complexity': 0.05
        }
        
        confidence = 0.5  # Base confidence
        
        for factor, value in factors.items():
            if factor in weights:
                confidence += weights[factor] * value
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_confabulation_risk(self, factors: Dict[str, float]) -> float:
        """Calculate risk of confabulation (inverse of confidence with different weighting)."""
        risk_factors = {
            'factual_support': -0.4,  # Strong factual support reduces risk
            'fact_confidence': -0.2,
            'contradiction_severity': 0.3,  # Contradictions increase risk
            'specificity': 0.2,  # High specificity without support increases risk
            'uncertainty_hedging': -0.15,  # Hedging reduces risk
            'historical_reliability': -0.1,
            'length_complexity': 0.05
        }
        
        base_risk = 0.3  # Base confabulation risk
        
        for factor, value in factors.items():
            if factor in risk_factors:
                base_risk += risk_factors[factor] * value
        
        return max(0.0, min(1.0, base_risk))
    
    def _identify_uncertainty_indicators(self, response: str) -> List[str]:
        """Identify uncertainty indicators in the response."""
        indicators = []
        response_lower = response.lower()
        
        for phrase in self.uncertainty_phrases:
            if phrase.lower() in response_lower:
                indicators.append(phrase)
        
        return indicators
    
    def _determine_filtering_action(self, response: str, confidence: float, 
                                  confabulation_risk: float,
                                  supporting_facts: List[EnhancedTripletFact],
                                  contradicting_facts: List[EnhancedTripletFact]) -> Tuple[str, str]:
        """Determine what action to take based on assessment."""
        
        # High confidence, low risk: approve as-is
        if confidence >= self.confidence_threshold and confabulation_risk <= 0.3:
            return "approved", response
        
        # Very high risk: reject completely
        if confabulation_risk >= self.confabulation_threshold:
            if contradicting_facts:
                return "rejected", "I have conflicting information about that. Let me clarify what I know for certain."
            else:
                return "rejected", "I don't have reliable information about that in my memory."
        
        # Medium confidence: add hedging
        if confidence >= 0.4:
            hedged_response = self._add_hedging_language(response, confidence)
            return "hedged", hedged_response
        
        # Low confidence: request clarification
        clarification = self._generate_clarification_response(supporting_facts, contradicting_facts)
        return "clarified", clarification
    
    def _add_hedging_language(self, response: str, confidence: float) -> str:
        """Add appropriate hedging language based on confidence level."""
        if confidence >= 0.6:
            hedge = "Based on what I remember, "
        elif confidence >= 0.5:
            hedge = "I believe "
        else:
            hedge = "I think "
        
        # Don't add hedge if already present
        if any(uncertain in response.lower() for uncertain in self.uncertainty_phrases):
            return response
        
        return hedge + response.lower()
    
    def _generate_clarification_response(self, supporting_facts: List[EnhancedTripletFact],
                                       contradicting_facts: List[EnhancedTripletFact]) -> str:
        """Generate a clarification response when confidence is low."""
        if contradicting_facts:
            return ("I have some conflicting information about that. "
                   "Could you help clarify what you're specifically asking about?")
        elif supporting_facts:
            return ("I have some related information, but I'm not completely certain. "
                   "Could you be more specific about what you'd like to know?")
        else:
            return ("I don't have clear information about that in my memory. "
                   "Could you provide more context or ask about something else?")
    
    def update_response_history(self, assessment: ResponseAssessment, 
                              user_feedback: str = None) -> Dict:
        """Update historical reliability tracking based on response assessment and feedback."""
        # This would typically update a persistent reliability history
        # For now, return the current assessment metrics
        return {
            'timestamp': time.time(),
            'confidence_score': assessment.confidence_score,
            'confabulation_risk': assessment.confabulation_risk,
            'action_taken': assessment.action_taken,
            'user_feedback': user_feedback,
            'supporting_fact_count': len(assessment.supporting_facts),
            'contradicting_fact_count': len(assessment.contradicting_facts)
        }
    
    def get_filtering_summary(self, assessments: List[ResponseAssessment]) -> Dict:
        """Get a summary of filtering performance."""
        if not assessments:
            return {'message': 'No assessments to analyze'}
        
        action_counts = Counter(a.action_taken for a in assessments)
        avg_confidence = sum(a.confidence_score for a in assessments) / len(assessments)
        avg_confabulation_risk = sum(a.confabulation_risk for a in assessments) / len(assessments)
        
        return {
            'total_assessments': len(assessments),
            'action_distribution': dict(action_counts),
            'average_confidence': avg_confidence,
            'average_confabulation_risk': avg_confabulation_risk,
            'filtering_rate': (len(assessments) - action_counts.get('approved', 0)) / len(assessments),
            'reliability_indicators': {
                'high_confidence_responses': sum(1 for a in assessments if a.confidence_score >= 0.8),
                'high_risk_responses': sum(1 for a in assessments if a.confabulation_risk >= 0.7),
                'hedged_responses': action_counts.get('hedged', 0),
                'rejected_responses': action_counts.get('rejected', 0)
            }
        } 