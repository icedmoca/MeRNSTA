# cortex/entropy.py
import numpy as np
from typing import List, Dict, Tuple
import math

class EntropyCalculator:
    """
    Calculate conditional entropy over token logits to detect confusion/uncertainty.
    H(W|C_t) = -∑_{v∈V} P(v|C_t) log P(v|C_t)
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def conditional_entropy(self, logits: np.ndarray, context: List[str] = None) -> float:
        """
        Calculate conditional entropy H(W|C_t) from token logits.
        
        Args:
            logits: Raw logits from language model
            context: Optional context tokens for conditioning
            
        Returns:
            Entropy value (higher = more uncertainty)
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Convert to probabilities via softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Calculate entropy: H = -∑ p_i * log(p_i)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
                
        return entropy
        
    def perplexity(self, logits: np.ndarray) -> float:
        """
        Calculate perplexity as exp(entropy).
        Perplexity measures how "surprised" the model is.
        """
        entropy = self.conditional_entropy(logits)
        return math.exp(entropy)
        
    def confidence_score(self, logits: np.ndarray) -> float:
        """
        Calculate confidence as 1 - normalized_entropy.
        Higher confidence = lower entropy.
        """
        entropy = self.conditional_entropy(logits)
        max_entropy = math.log(len(logits))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return 1.0 - normalized_entropy
        
    def detect_confusion(self, logits: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Detect if model is confused (high entropy).
        
        Args:
            logits: Token logits
            threshold: Entropy threshold for confusion detection
            
        Returns:
            True if model appears confused
        """
        entropy = self.conditional_entropy(logits)
        return entropy > threshold
        
    def top_k_entropy(self, logits: np.ndarray, k: int = 5) -> float:
        """
        Calculate entropy over top-k tokens only.
        Useful for focused uncertainty measurement.
        """
        # Get top-k indices
        top_k_indices = np.argsort(logits)[-k:]
        top_k_logits = logits[top_k_indices]
        
        return self.conditional_entropy(top_k_logits)
        
    def entropy_gradient(self, logits: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Calculate gradient of entropy with respect to logits.
        Useful for optimization and analysis.
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Gradient: ∂H/∂logits = -probs * (log(probs) + H)
        entropy = self.conditional_entropy(logits)
        gradient = -probs * (np.log(probs + epsilon) + entropy)
        
        return gradient
        
    def uncertainty_metrics(self, logits: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive uncertainty metrics.
        
        Returns:
            Dictionary with entropy, perplexity, confidence, and confusion flag
        """
        return {
            'entropy': self.conditional_entropy(logits),
            'perplexity': self.perplexity(logits),
            'confidence': self.confidence_score(logits),
            'confused': self.detect_confusion(logits),
            'top_5_entropy': self.top_k_entropy(logits, k=5)
        } 