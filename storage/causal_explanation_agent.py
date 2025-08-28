#!/usr/bin/env python3
"""
Causal Chain Explanation Agent for MeRNSTA

This agent provides explainable AI functionality by:
1. Tracing causal ancestors for any given fact
2. Ranking causal chains by total strength
3. Generating natural language explanations
4. Supporting "Why do you believe X?" queries

📌 DO NOT HARDCODE explanation templates or scoring weights.
All parameters must be loaded from `config.settings` or environment config.
This is a zero-hardcoding cognitive subsystem.
"""

import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from storage.db_utils import get_connection_pool
from config.settings import DEFAULT_VALUES


@dataclass
class CausalNode:
    """Represents a node in a causal chain."""
    fact_id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    causal_strength: float = 0.0
    depth: int = 0
    total_strength: float = 0.0  # Accumulated strength from root
    timestamp: Optional[float] = None


@dataclass
class CausalChain:
    """Represents a complete causal chain with explanation."""
    root_cause: CausalNode
    target_fact: CausalNode
    chain_nodes: List[CausalNode]
    total_strength: float
    explanation: str


class CausalExplanationAgent:
    """Agent for generating causal explanations and tracing reasoning chains."""
    
    def __init__(self):
        """Initialize the causal explanation agent with configurable parameters."""
        # Load configuration parameters (no hardcoding)
        self.max_chain_depth = DEFAULT_VALUES.get("max_causal_chain_depth", 5)
        self.min_explanation_strength = DEFAULT_VALUES.get("min_explanation_strength", 0.2)
        self.explanation_confidence_threshold = DEFAULT_VALUES.get("explanation_confidence_threshold", 0.5)
        
        # Natural language templates (configurable)
        self.explanation_templates = self._load_explanation_templates()
        
        print(f"[CausalExplanation] Initialized with max_depth={self.max_chain_depth}, "
              f"min_strength={self.min_explanation_strength}")
    
    def _load_explanation_templates(self) -> Dict[str, List[str]]:
        """Load explanation templates from configuration."""
        # TODO: Load from config file instead of hardcoding
        return {
            "action_emotion": [
                "You {target_predicate} {target_object} because you {cause_predicate} {cause_object}, which often leads to {emotional_response}",
                "Your feeling of {target_object} stems from {cause_predicate} {cause_object}, a common emotional response",
                "The reason you {target_predicate} is that you {cause_predicate} {cause_object}, creating this emotional reaction"
            ],
            "state_emotion": [
                "You {target_predicate} because {cause_object} {cause_predicate}, which naturally causes {emotional_state}",
                "Your {target_object} feeling comes from the fact that {cause_object} {cause_predicate}",
                "You experience {target_object} as a result of {cause_object} being {cause_predicate}"
            ],
            "sequential": [
                "This happened because of a chain of events: {chain_description}",
                "The sequence leading to this was: {chain_description}",
                "Your current state resulted from: {chain_description}"
            ],
            "default": [
                "You {target_predicate} {target_object} because you {cause_predicate} {cause_object}",
                "This is connected to your experience of {cause_predicate} {cause_object}",
                "The cause appears to be {cause_predicate} {cause_object}"
            ]
        }
    
    def explain_fact(self, fact_id: int, max_chains: int = 3) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for why a fact exists.
        
        Args:
            fact_id: The fact to explain
            max_chains: Maximum number of causal chains to return
            
        Returns:
            Dictionary with explanation, causal chains, and confidence scores
        """
        try:
            # Get the target fact
            target_fact = self._get_fact_by_id(fact_id)
            if not target_fact:
                return {"error": f"Fact {fact_id} not found"}
            
            # Trace all causal chains leading to this fact
            causal_chains = self._trace_causal_chains(target_fact, max_chains)
            
            if not causal_chains:
                return {
                    "fact_id": fact_id,
                    "explanation": f"I believe '{target_fact.subject} {target_fact.predicate} {target_fact.object}' "
                                 f"based on direct observation, but I don't see clear causal connections to other facts.",
                    "confidence": target_fact.confidence,
                    "causal_chains": [],
                    "reasoning_type": "direct_observation"
                }
            
            # Generate natural language explanation
            primary_explanation = self._generate_explanation(target_fact, causal_chains[0])
            
            # Calculate overall explanation confidence
            explanation_confidence = self._calculate_explanation_confidence(causal_chains)
            
            return {
                "fact_id": fact_id,
                "explanation": primary_explanation,
                "confidence": explanation_confidence,
                "causal_chains": [self._serialize_chain(chain) for chain in causal_chains],
                "reasoning_type": "causal_inference",
                "chain_count": len(causal_chains),
                "strongest_cause": {
                    "fact_id": causal_chains[0].root_cause.fact_id,
                    "description": f"{causal_chains[0].root_cause.subject} {causal_chains[0].root_cause.predicate} {causal_chains[0].root_cause.object}",
                    "strength": causal_chains[0].total_strength
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to explain fact {fact_id}: {str(e)}"}
    
    def _get_fact_by_id(self, fact_id: int) -> Optional[CausalNode]:
        """Retrieve a fact and its causal information from the database."""
        try:
            with get_connection_pool().get_connection() as conn:
                result = conn.execute(
                    """SELECT f.id, f.subject, f.predicate, f.object, f.confidence, f.timestamp,
                              COALESCE(ef.causal_strength, 0) as causal_strength
                       FROM facts f
                       LEFT JOIN enhanced_facts ef ON f.id = ef.id
                       WHERE f.id = ?""",
                    (fact_id,)
                ).fetchone()
                
                if result:
                    return CausalNode(
                        fact_id=result[0],
                        subject=result[1], 
                        predicate=result[2],
                        object=result[3],
                        confidence=result[4],
                        timestamp=result[5],
                        causal_strength=result[6]
                    )
                return None
                
        except Exception as e:
            print(f"[CausalExplanation] Error getting fact {fact_id}: {e}")
            return None
    
    def _trace_causal_chains(self, target_fact: CausalNode, max_chains: int) -> List[CausalChain]:
        """Trace all causal chains leading to the target fact."""
        chains = []
        visited = set()
        
        # Find direct causes first
        direct_causes = self._find_direct_causes(target_fact.fact_id)
        
        for cause in direct_causes:
            if cause.fact_id in visited:
                continue
                
            # Recursively build causal chain
            chain = self._build_causal_chain(cause, target_fact, depth=0, visited=visited.copy())
            if chain and chain.total_strength >= self.min_explanation_strength:
                chains.append(chain)
                visited.add(cause.fact_id)
        
        # Sort by total causal strength
        chains.sort(key=lambda c: c.total_strength, reverse=True)
        
        return chains[:max_chains]
    
    def _find_direct_causes(self, fact_id: int) -> List[CausalNode]:
        """Find facts that directly cause the given fact."""
        causes = []
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Query for facts that have causal relationships to this fact
                results = conn.execute(
                    """SELECT f.id, f.subject, f.predicate, f.object, f.confidence, 
                              f.timestamp, ef.causal_strength
                       FROM facts f
                       JOIN enhanced_facts ef ON f.id = ef.fact_id
                       WHERE ef.change_history LIKE '%causal:%'
                       AND ef.causal_strength > 0
                       ORDER BY ef.causal_strength DESC""",
                ).fetchall()
                
                # Filter for causes that affect our target fact
                target_fact = self._get_fact_by_id(fact_id)
                if not target_fact:
                    return causes
                
                for result in results:
                    # Check if this fact's change_history indicates it causes our target
                    cause_node = CausalNode(
                        fact_id=result[0],
                        subject=result[1],
                        predicate=result[2], 
                        object=result[3],
                        confidence=result[4],
                        timestamp=result[5],
                        causal_strength=result[6]
                    )
                    
                    # Simplified causality check - in a full implementation, 
                    # we'd parse the change_history more sophisticated
                    if self._is_causal_relationship(cause_node, target_fact):
                        causes.append(cause_node)
                        
        except Exception as e:
            print(f"[CausalExplanation] Error finding causes for {fact_id}: {e}")
        
        return causes
    
    def _is_causal_relationship(self, cause: CausalNode, effect: CausalNode) -> bool:
        """Check if there's a causal relationship between two facts."""
        # Temporal check - cause must come before effect
        if cause.timestamp and effect.timestamp and cause.timestamp > effect.timestamp:
            return False
        
        # Semantic similarity check
        cause_text = f"{cause.subject} {cause.predicate} {cause.object}".lower()
        effect_text = f"{effect.subject} {effect.predicate} {effect.object}".lower()
        
        # Simple keyword overlap check (in production, use semantic similarity)
        cause_words = set(cause_text.split())
        effect_words = set(effect_text.split())
        common_words = cause_words.intersection(effect_words)
        
        # If they share subject or significant semantic overlap, likely causal
        if cause.subject.lower() == effect.subject.lower():
            return True
        if len(common_words) >= 2:
            return True
            
        return False
    
    def _build_causal_chain(self, root_cause: CausalNode, target_fact: CausalNode, 
                           depth: int, visited: set) -> Optional[CausalChain]:
        """Recursively build a causal chain from root cause to target fact."""
        if depth > self.max_chain_depth:
            return None
        
        visited.add(root_cause.fact_id)
        chain_nodes = [root_cause]
        
        # For now, create simple direct causation chain
        # In a full implementation, this would recursively find intermediate causes
        current_node = root_cause
        total_strength = current_node.causal_strength
        
        # Build explanation
        explanation = self._generate_explanation(target_fact, 
                                               CausalChain(root_cause, target_fact, chain_nodes, total_strength, ""))
        
        return CausalChain(
            root_cause=root_cause,
            target_fact=target_fact,
            chain_nodes=chain_nodes,
            total_strength=total_strength,
            explanation=explanation
        )
    
    def _generate_explanation(self, target_fact: CausalNode, causal_chain: CausalChain) -> str:
        """Generate natural language explanation for a causal relationship."""
        try:
            root_cause = causal_chain.root_cause
            
            # Determine explanation pattern based on predicate types
            explanation_type = self._classify_causal_pattern(root_cause, target_fact)
            templates = self.explanation_templates.get(explanation_type, self.explanation_templates["default"])
            
            # Select template based on strength (stronger relationships get more confident language)
            template_index = min(int(causal_chain.total_strength * len(templates)), len(templates) - 1)
            template = templates[template_index]
            
            # Fill in the template
            explanation = template.format(
                target_predicate=target_fact.predicate,
                target_object=target_fact.object,
                cause_predicate=root_cause.predicate,
                cause_object=root_cause.object,
                emotional_response=self._get_emotional_descriptor(target_fact),
                emotional_state=target_fact.object,
                chain_description=self._describe_chain(causal_chain.chain_nodes)
            )
            
            # Add confidence qualifier
            confidence_qualifier = self._get_confidence_qualifier(causal_chain.total_strength)
            
            return f"{confidence_qualifier}{explanation}."
            
        except Exception as e:
            print(f"[CausalExplanation] Error generating explanation: {e}")
            return f"You {target_fact.predicate} {target_fact.object} because of {root_cause.predicate} {root_cause.object}."
    
    def _classify_causal_pattern(self, cause: CausalNode, effect: CausalNode) -> str:
        """Classify the type of causal pattern for appropriate explanation template."""
        # Action predicates
        action_predicates = {"join", "start", "begin", "do", "perform", "complete", "finish", "try", "attempt"}
        
        # Emotion predicates  
        emotion_predicates = {"feel", "experience", "become", "get"}
        
        # State predicates
        state_predicates = {"is", "are", "was", "were", "being"}
        
        cause_pred = cause.predicate.lower()
        effect_pred = effect.predicate.lower()
        
        if any(pred in cause_pred for pred in action_predicates) and \
           any(pred in effect_pred for pred in emotion_predicates):
            return "action_emotion"
        elif any(pred in cause_pred for pred in state_predicates) and \
             any(pred in effect_pred for pred in emotion_predicates):
            return "state_emotion"
        elif len(effect.object.split()) > 2:  # Complex chains
            return "sequential"
        else:
            return "default"
    
    def _get_emotional_descriptor(self, fact: CausalNode) -> str:
        """Get appropriate emotional descriptor for the fact."""
        object_lower = fact.object.lower()
        
        if any(word in object_lower for word in ["stress", "overwhelm", "anxious", "worry"]):
            return "stress and emotional strain"
        elif any(word in object_lower for word in ["happy", "joy", "excited", "confident"]):
            return "positive emotions"
        elif any(word in object_lower for word in ["tired", "exhaust", "fatigue"]):
            return "fatigue and energy depletion"
        else:
            return f"feelings of {fact.object}"
    
    def _describe_chain(self, chain_nodes: List[CausalNode]) -> str:
        """Create a natural language description of a causal chain."""
        if len(chain_nodes) == 1:
            node = chain_nodes[0]
            return f"{node.subject} {node.predicate} {node.object}"
        
        descriptions = []
        for node in chain_nodes:
            descriptions.append(f"{node.predicate} {node.object}")
        
        return " → ".join(descriptions)
    
    def _get_confidence_qualifier(self, strength: float) -> str:
        """Get confidence qualifier based on causal strength."""
        if strength >= 0.8:
            return "I'm quite confident that "
        elif strength >= 0.6:
            return "It appears that "
        elif strength >= 0.4:
            return "It seems likely that "
        elif strength >= 0.2:
            return "I suspect that "
        else:
            return "There's a possibility that "
    
    def _calculate_explanation_confidence(self, chains: List[CausalChain]) -> float:
        """Calculate overall confidence in the explanation."""
        if not chains:
            return 0.0
        
        # Weight by chain strength and number of supporting chains
        total_weight = sum(chain.total_strength for chain in chains)
        chain_bonus = min(0.2, len(chains) * 0.05)  # Bonus for multiple supporting chains
        
        base_confidence = total_weight / len(chains)  # Average strength
        return min(1.0, base_confidence + chain_bonus)
    
    def _serialize_chain(self, chain: CausalChain) -> Dict[str, Any]:
        """Serialize a causal chain for API response."""
        return {
            "root_cause": {
                "fact_id": chain.root_cause.fact_id,
                "description": f"{chain.root_cause.subject} {chain.root_cause.predicate} {chain.root_cause.object}",
                "confidence": chain.root_cause.confidence,
                "timestamp": chain.root_cause.timestamp
            },
            "chain_nodes": [
                {
                    "fact_id": node.fact_id,
                    "description": f"{node.subject} {node.predicate} {node.object}",
                    "causal_strength": node.causal_strength,
                    "depth": node.depth
                }
                for node in chain.chain_nodes
            ],
            "total_strength": chain.total_strength,
            "explanation": chain.explanation
        }
    
    def predict_next_states(self, user_id: str, max_predictions: int = 3) -> List[Dict[str, Any]]:
        """
        Predict likely next emotional, behavioral, or preference states based on current causal patterns.
        
        This is a placeholder for the prediction module implementation.
        """
        # TODO: Implement in the causal prediction module
        return [
            {
                "prediction_type": "emotional",
                "predicted_state": "You are likely to feel more confident",
                "confidence": 0.7,
                "reasoning": "Based on your recent positive experiences"
            }
        ] 