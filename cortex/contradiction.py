# cortex/contradiction.py
import numpy as np
from typing import List, Tuple, Dict
import math

class ContradictionDetector:
    """
    Contradiction detection engine implementing the mathematical core.
    Contradict(w) = max_i[I_rule + γ(1-cos θ_w,i)]
    """
    
    def __init__(self, gamma: float = 0.15):
        self.gamma = gamma
        self.rules = []
        
    def add_rule(self, good: str, bad: str):
        """Add a contradiction rule"""
        self.rules.append({"good": good, "bad": bad})
        
    def set_rules(self, rules: List[Dict[str, str]]):
        """Set multiple contradiction rules"""
        self.rules = rules
        
    def contradict(self, token: str, memory_tokens: List[Tuple[str, float]], 
                  embeddings: Dict[str, np.ndarray]) -> float:
        """
        Calculate contradiction score for a token against memory.
        
        Args:
            token: Current token to check
            memory_tokens: List of (token, similarity_score) from memory
            embeddings: Dict mapping tokens to their embeddings
            
        Returns:
            Contradiction score (higher = more contradictory)
        """
        max_contradiction = 0.0
        
        for rule in self.rules:
            good_token = rule["good"]
            bad_token = rule["bad"]
            
            # Check if current token contains the "bad" part of a rule
            if bad_token.lower() in token.lower():
                # Find if we have the "good" token in memory
                for mem_token, sim_score in memory_tokens:
                    if mem_token == good_token:
                        # Rule-based contradiction indicator
                        rule_indicator = 1.0
                        
                        # Semantic distance component
                        if token in embeddings and good_token in embeddings:
                            cos_sim = self._cosine_similarity(
                                embeddings[token], embeddings[good_token]
                            )
                            semantic_distance = 1.0 - cos_sim
                        else:
                            semantic_distance = 0.5  # Default distance
                            
                        # Combined contradiction score
                        contradiction_score = rule_indicator + self.gamma * semantic_distance
                        max_contradiction = max(max_contradiction, contradiction_score)
                        
        return max_contradiction
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def should_veto(self, token: str, memory_tokens: List[Tuple[str, float]], 
                   embeddings: Dict[str, np.ndarray], threshold: float = 0.5) -> bool:
        """Determine if a token should be vetoed based on contradiction score"""
        contradiction_score = self.contradict(token, memory_tokens, embeddings)
        return contradiction_score > threshold
        
    def get_logit_penalty(self, token: str, memory_tokens: List[Tuple[str, float]], 
                         embeddings: Dict[str, np.ndarray], beta: float = 5.0) -> float:
        """
        Calculate logit penalty: l'_w = l_w - β * Contradict(w, M_hi)
        """
        contradiction_score = self.contradict(token, memory_tokens, embeddings)
        return -beta * contradiction_score 