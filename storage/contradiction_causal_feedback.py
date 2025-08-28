#!/usr/bin/env python3
"""
Contradiction-Causal Feedback Loop for MeRNSTA

This system creates a feedback loop between contradiction detection and causal reasoning:
1. When contradictions arise, check if they invalidate existing causal chains
2. Flag upstream facts as suspicious if their causal chains are broken
3. Add volatility_boost to downstream beliefs affected by invalidated chains
4. Adjust confidence scores for facts in broken causal chains

📌 DO NOT HARDCODE volatility boosts or confidence penalties.
All parameters must be loaded from `config.settings` or environment config.
This is a zero-hardcoding cognitive subsystem.
"""

import time
import math
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from storage.db_utils import get_connection_pool
from config.settings import DEFAULT_VALUES


@dataclass
class CausalChainInvalidation:
    """Represents an invalidated causal chain due to contradiction."""
    chain_id: str
    invalidating_fact_id: int
    affected_fact_ids: List[int]
    contradiction_type: str
    invalidation_confidence: float
    volatility_adjustments: Dict[int, float]  # fact_id -> volatility_boost


@dataclass
class ContradictionImpact:
    """Represents the impact of a contradiction on the causal graph."""
    contradiction_id: int
    fact_a_id: int
    fact_b_id: int
    invalidated_chains: List[CausalChainInvalidation]
    upstream_suspicions: List[int]  # fact_ids marked as suspicious
    volatility_propagation: Dict[int, float]  # fact_id -> new_volatility


class ContradictionCausalFeedback:
    """System for managing feedback between contradictions and causal reasoning."""
    
    def __init__(self):
        """Initialize the contradiction-causal feedback system."""
        # Load configuration parameters (no hardcoding)
        self.volatility_boost_factor = DEFAULT_VALUES.get("contradiction_volatility_boost", 0.3)
        self.confidence_penalty_factor = DEFAULT_VALUES.get("causal_confidence_penalty", 0.2)
        self.propagation_decay_rate = DEFAULT_VALUES.get("volatility_propagation_decay", 0.8)
        self.suspicion_threshold = DEFAULT_VALUES.get("upstream_suspicion_threshold", 0.7)
        self.max_propagation_depth = DEFAULT_VALUES.get("max_volatility_propagation_depth", 3)
        self.confidence_decay_rate = DEFAULT_VALUES.get("belief_confidence_decay_rate", 0.85)
        
        print(f"[ContradictionCausal] Initialized with volatility_boost={self.volatility_boost_factor}, "
              f"confidence_penalty={self.confidence_penalty_factor}")
    
    def process_contradiction(self, fact_a_id: int, fact_b_id: int, 
                            contradiction_strength: float) -> ContradictionImpact:
        """
        Process a newly detected contradiction and its impact on causal chains.
        
        Args:
            fact_a_id: First contradicting fact
            fact_b_id: Second contradicting fact
            contradiction_strength: Strength of the contradiction (0-1)
            
        Returns:
            ContradictionImpact object describing all effects
        """
        try:
            print(f"[ContradictionCausal] Processing contradiction between facts {fact_a_id} and {fact_b_id}")
            
            # Find causal chains involving these facts
            affected_chains = self._find_causal_chains_involving_facts([fact_a_id, fact_b_id])
            
            # Analyze which chains are invalidated
            invalidated_chains = []
            for chain in affected_chains:
                invalidation = self._analyze_chain_invalidation(
                    chain, fact_a_id, fact_b_id, contradiction_strength
                )
                if invalidation:
                    invalidated_chains.append(invalidation)
            
            # Identify upstream facts to mark as suspicious
            upstream_suspicions = self._identify_upstream_suspicions(invalidated_chains)
            
            # Calculate volatility propagation
            volatility_propagation = self._calculate_volatility_propagation(
                invalidated_chains, contradiction_strength
            )
            
            # Calculate confidence decay propagation for downstream beliefs
            confidence_decay_propagation = self._calculate_confidence_decay_propagation(
                invalidated_chains, contradiction_strength
            )
            
            # Apply the changes to the database
            self._apply_contradiction_effects(
                fact_a_id, fact_b_id, invalidated_chains, 
                upstream_suspicions, volatility_propagation, confidence_decay_propagation
            )
            
            return ContradictionImpact(
                contradiction_id=int(time.time() * 1000),  # Simple ID
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                invalidated_chains=invalidated_chains,
                upstream_suspicions=upstream_suspicions,
                volatility_propagation=volatility_propagation,
                confidence_decay_propagation=confidence_decay_propagation
            )
            
        except Exception as e:
            print(f"[ContradictionCausal] Error processing contradiction: {e}")
            return ContradictionImpact(0, fact_a_id, fact_b_id, [], [], {})
    
    def _find_causal_chains_involving_facts(self, fact_ids: List[int]) -> List[Dict[str, Any]]:
        """Find all causal chains that involve the specified facts."""
        chains = []
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Find facts with causal relationships involving our target facts
                for fact_id in fact_ids:
                    # Facts caused by this fact
                    results = conn.execute("""
                        SELECT f.id, f.subject, f.predicate, f.object,
                               ef.causal_strength, ef.change_history
                        FROM facts f
                        JOIN enhanced_facts ef ON f.id = ef.id
                        WHERE ef.change_history LIKE ?
                        AND ef.causal_strength > 0
                    """, (f"%{fact_id}%",)).fetchall()
                    
                    for result in results:
                        chains.append({
                            "fact_id": result[0],
                            "description": f"{result[1]} {result[2]} {result[3]}",
                            "causal_strength": result[4],
                            "change_history": result[5],
                            "involves_fact": fact_id
                        })
                
                # Also find chains where our facts are effects
                for fact_id in fact_ids:
                    result = conn.execute("""
                        SELECT f.id, f.subject, f.predicate, f.object,
                               ef.causal_strength, ef.change_history
                        FROM facts f  
                        JOIN enhanced_facts ef ON f.id = ef.id
                        WHERE f.id = ?
                        AND ef.causal_strength > 0
                    """, (fact_id,)).fetchone()
                    
                    if result:
                        chains.append({
                            "fact_id": result[0],
                            "description": f"{result[1]} {result[2]} {result[3]}",
                            "causal_strength": result[4],
                            "change_history": result[5],
                            "involves_fact": fact_id
                        })
                        
        except Exception as e:
            print(f"[ContradictionCausal] Error finding causal chains: {e}")
        
        return chains
    
    def _analyze_chain_invalidation(self, chain: Dict[str, Any], 
                                  fact_a_id: int, fact_b_id: int,
                                  contradiction_strength: float) -> Optional[CausalChainInvalidation]:
        """Analyze whether a contradiction invalidates a causal chain."""
        try:
            # Determine invalidation confidence based on several factors
            invalidation_confidence = 0.0
            
            # Direct involvement in contradiction
            chain_fact_id = chain["fact_id"]
            if chain_fact_id in [fact_a_id, fact_b_id]:
                invalidation_confidence += 0.5
            
            # Causal strength of the chain (weaker chains more easily invalidated)
            causal_strength = chain["causal_strength"]
            strength_penalty = (1.0 - causal_strength) * 0.3
            invalidation_confidence += strength_penalty
            
            # Contradiction strength
            invalidation_confidence += contradiction_strength * 0.4
            
            # Check logical inconsistency
            if self._check_logical_inconsistency(chain, fact_a_id, fact_b_id):
                invalidation_confidence += 0.3
            
            # Only invalidate if confidence is above threshold
            if invalidation_confidence < 0.6:
                return None
            
            # Calculate volatility adjustments for affected facts
            volatility_adjustments = {}
            affected_facts = self._extract_affected_facts_from_chain(chain)
            
            for affected_fact_id in affected_facts:
                # Boost volatility based on invalidation confidence
                volatility_boost = self.volatility_boost_factor * invalidation_confidence
                volatility_adjustments[affected_fact_id] = volatility_boost
            
            return CausalChainInvalidation(
                chain_id=f"chain_{chain_fact_id}_{int(time.time())}",
                invalidating_fact_id=fact_a_id if chain_fact_id == fact_b_id else fact_b_id,
                affected_fact_ids=affected_facts,
                contradiction_type="logical_inconsistency",
                invalidation_confidence=invalidation_confidence,
                volatility_adjustments=volatility_adjustments
            )
            
        except Exception as e:
            print(f"[ContradictionCausal] Error analyzing chain invalidation: {e}")
            return None
    
    def _check_logical_inconsistency(self, chain: Dict[str, Any], 
                                   fact_a_id: int, fact_b_id: int) -> bool:
        """Check if the contradiction creates logical inconsistency in the causal chain."""
        # Simplified logic - in production would be more sophisticated
        
        # If the chain involves both contradicting facts, there's logical inconsistency
        change_history = chain.get("change_history", "")
        
        # Check if both facts are referenced in the causal history
        has_fact_a = str(fact_a_id) in change_history
        has_fact_b = str(fact_b_id) in change_history
        
        return has_fact_a and has_fact_b
    
    def _extract_affected_facts_from_chain(self, chain: Dict[str, Any]) -> List[int]:
        """Extract all fact IDs that are part of the causal chain."""
        affected_facts = [chain["fact_id"]]
        
        # Parse change_history for related fact IDs
        change_history = chain.get("change_history", "")
        if change_history:
            # Simple parsing - in production would be more sophisticated
            import re
            fact_id_pattern = r'\b\d+\b'
            found_ids = re.findall(fact_id_pattern, change_history)
            
            for fact_id_str in found_ids:
                try:
                    fact_id = int(fact_id_str)
                    if fact_id not in affected_facts:
                        affected_facts.append(fact_id)
                except ValueError:
                    continue
        
        return affected_facts
    
    def _identify_upstream_suspicions(self, invalidated_chains: List[CausalChainInvalidation]) -> List[int]:
        """Identify upstream facts that should be marked as suspicious."""
        suspicious_facts = []
        
        for invalidation in invalidated_chains:
            if invalidation.invalidation_confidence >= self.suspicion_threshold:
                # Mark the invalidating fact and its causal ancestors as suspicious
                suspicious_facts.append(invalidation.invalidating_fact_id)
                
                # Find causal ancestors
                ancestors = self._find_causal_ancestors(invalidation.invalidating_fact_id)
                suspicious_facts.extend(ancestors)
        
        return list(set(suspicious_facts))  # Remove duplicates
    
    def _find_causal_ancestors(self, fact_id: int, depth: int = 0) -> List[int]:
        """Find causal ancestors of a fact (facts that caused it)."""
        if depth >= self.max_propagation_depth:
            return []
        
        ancestors = []
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Find facts that caused this one
                result = conn.execute("""
                    SELECT change_history FROM enhanced_facts WHERE fact_id = ?
                """, (fact_id,)).fetchone()
                
                if result and result[0]:
                    change_history = result[0]
                    
                    # Extract causal fact IDs from change_history
                    import re
                    if "causal:" in change_history:
                        # Simple parsing - in production would be more sophisticated
                        causal_lines = [line for line in change_history.split('\n') if "causal:" in line]
                        
                        for line in causal_lines:
                            # Extract fact IDs
                            fact_id_pattern = r'\b\d+\b'
                            found_ids = re.findall(fact_id_pattern, line)
                            
                            for ancestor_id_str in found_ids:
                                try:
                                    ancestor_id = int(ancestor_id_str)
                                    if ancestor_id != fact_id:
                                        ancestors.append(ancestor_id)
                                        # Recursively find ancestors
                                        deeper_ancestors = self._find_causal_ancestors(ancestor_id, depth + 1)
                                        ancestors.extend(deeper_ancestors)
                                except ValueError:
                                    continue
                        
        except Exception as e:
            print(f"[ContradictionCausal] Error finding ancestors: {e}")
        
        return list(set(ancestors))  # Remove duplicates
    
    def _calculate_volatility_propagation(self, invalidated_chains: List[CausalChainInvalidation],
                                        contradiction_strength: float) -> Dict[int, float]:
        """Calculate how volatility should propagate through the causal graph."""
        volatility_propagation = {}
        
        for invalidation in invalidated_chains:
            # Apply volatility adjustments from the invalidation
            for fact_id, volatility_boost in invalidation.volatility_adjustments.items():
                if fact_id in volatility_propagation:
                    # Combine multiple volatility boosts
                    volatility_propagation[fact_id] = min(1.0, 
                        volatility_propagation[fact_id] + volatility_boost)
                else:
                    volatility_propagation[fact_id] = volatility_boost
            
            # Propagate volatility to causally connected facts
            for fact_id in invalidation.affected_fact_ids:
                self._propagate_volatility_to_connected_facts(
                    fact_id, volatility_boost, volatility_propagation, depth=0
                )
        
        return volatility_propagation
    
    def _calculate_confidence_decay_propagation(self, invalidated_chains: List[CausalChainInvalidation],
                                              contradiction_strength: float) -> Dict[int, float]:
        """Calculate confidence decay that should propagate through causal chains."""
        confidence_decay_map = {}
        
        for invalidation in invalidated_chains:
            # Start from the invalidated chain and propagate downstream
            for fact_id in invalidation.affected_fact_ids:
                # Calculate base decay based on invalidation confidence and contradiction strength
                base_decay = (invalidation.invalidation_confidence * contradiction_strength * 
                            (1.0 - self.confidence_decay_rate))
                
                if fact_id in confidence_decay_map:
                    # Combine multiple decay sources
                    confidence_decay_map[fact_id] = max(confidence_decay_map[fact_id], base_decay)
                else:
                    confidence_decay_map[fact_id] = base_decay
                
                # Propagate to causally connected downstream facts
                self._propagate_confidence_decay_to_downstream(
                    fact_id, base_decay, confidence_decay_map, depth=0
                )
        
        return confidence_decay_map
    
    def _propagate_confidence_decay_to_downstream(self, start_fact_id: int, base_decay: float,
                                                confidence_decay_map: Dict[int, float], depth: int):
        """Propagate confidence decay to downstream facts that depend on this fact."""
        if depth >= self.max_propagation_depth:
            return
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Find facts that are causally dependent on this fact
                results = conn.execute("""
                    SELECT f.id FROM facts f
                    JOIN enhanced_facts ef ON f.id = ef.fact_id
                    WHERE ef.change_history LIKE ?
                    AND ef.causal_strength > 0
                """, (f"%{start_fact_id}%",)).fetchall()
                
                for result in results:
                    downstream_fact_id = result[0]
                    if downstream_fact_id != start_fact_id:
                        # Apply decayed confidence reduction
                        decayed_reduction = base_decay * (self.confidence_decay_rate ** (depth + 1))
                        
                        if downstream_fact_id in confidence_decay_map:
                            confidence_decay_map[downstream_fact_id] = max(
                                confidence_decay_map[downstream_fact_id], decayed_reduction)
                        else:
                            confidence_decay_map[downstream_fact_id] = decayed_reduction
                        
                        # Continue propagation
                        self._propagate_confidence_decay_to_downstream(
                            downstream_fact_id, decayed_reduction, confidence_decay_map, depth + 1
                        )
                        
        except Exception as e:
            print(f"[ContradictionCausal] Error propagating confidence decay: {e}")
    
    def _propagate_volatility_to_connected_facts(self, start_fact_id: int, base_volatility: float,
                                               volatility_map: Dict[int, float], depth: int):
        """Propagate volatility to causally connected facts with decay."""
        if depth >= self.max_propagation_depth:
            return
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Find facts causally connected to this one
                results = conn.execute("""
                    SELECT f.id FROM facts f
                    JOIN enhanced_facts ef ON f.id = ef.fact_id
                    WHERE ef.change_history LIKE ?
                    AND ef.causal_strength > 0
                """, (f"%{start_fact_id}%",)).fetchall()
                
                for result in results:
                    connected_fact_id = result[0]
                    if connected_fact_id != start_fact_id:
                        # Apply decayed volatility
                        decayed_volatility = base_volatility * (self.propagation_decay_rate ** (depth + 1))
                        
                        if connected_fact_id in volatility_map:
                            volatility_map[connected_fact_id] = min(1.0,
                                volatility_map[connected_fact_id] + decayed_volatility)
                        else:
                            volatility_map[connected_fact_id] = decayed_volatility
                        
                        # Continue propagation
                        self._propagate_volatility_to_connected_facts(
                            connected_fact_id, decayed_volatility, volatility_map, depth + 1
                        )
                        
        except Exception as e:
            print(f"[ContradictionCausal] Error propagating volatility: {e}")
    
    def _apply_contradiction_effects(self, fact_a_id: int, fact_b_id: int,
                                   invalidated_chains: List[CausalChainInvalidation],
                                   upstream_suspicions: List[int],
                                   volatility_propagation: Dict[int, float],
                                   confidence_decay_propagation: Dict[int, float]):
        """Apply the calculated effects of contradiction to the database."""
        try:
            with get_connection_pool().get_connection() as conn:
                # Update volatility scores for affected facts
                for fact_id, volatility_boost in volatility_propagation.items():
                    # Get current volatility
                    current_volatility = conn.execute(
                        "SELECT volatility_score FROM facts WHERE id = ?",
                        (fact_id,)
                    ).fetchone()
                    
                    if current_volatility:
                        new_volatility = min(1.0, (current_volatility[0] or 0) + volatility_boost)
                        
                        conn.execute(
                            "UPDATE facts SET volatility_score = ? WHERE id = ?",
                            (new_volatility, fact_id)
                        )
                        
                        print(f"[ContradictionCausal] Boosted volatility for fact {fact_id}: "
                              f"{current_volatility[0]:.3f} → {new_volatility:.3f}")
                
                # Apply confidence penalties to facts in invalidated chains
                for invalidation in invalidated_chains:
                    for fact_id in invalidation.affected_fact_ids:
                        current_confidence = conn.execute(
                            "SELECT confidence FROM facts WHERE id = ?",
                            (fact_id,)
                        ).fetchone()
                        
                        if current_confidence:
                            penalty = self.confidence_penalty_factor * invalidation.invalidation_confidence
                            new_confidence = max(0.0, (current_confidence[0] or 1.0) - penalty)
                            
                            conn.execute(
                                "UPDATE facts SET confidence = ? WHERE id = ?",
                                (new_confidence, fact_id)
                            )
                            
                            print(f"[ContradictionCausal] Applied confidence penalty to fact {fact_id}: "
                                  f"{current_confidence[0]:.3f} → {new_confidence:.3f}")
                
                # Mark upstream facts as suspicious (add to enhanced_facts metadata)
                for suspicious_fact_id in upstream_suspicions:
                    # Update the enhanced_facts table with suspicion flag
                    conn.execute("""
                        UPDATE enhanced_facts 
                        SET change_history = change_history || ?
                        WHERE fact_id = ?
                    """, (f"\nsuspicious: marked due to contradiction invalidation", suspicious_fact_id))
                    
                    print(f"[ContradictionCausal] Marked fact {suspicious_fact_id} as suspicious")
                
                # Apply confidence decay to downstream beliefs
                for fact_id, confidence_reduction in confidence_decay_propagation.items():
                    current_confidence = conn.execute(
                        "SELECT confidence FROM facts WHERE id = ?",
                        (fact_id,)
                    ).fetchone()
                    
                    if current_confidence:
                        new_confidence = max(0.0, (current_confidence[0] or 1.0) - confidence_reduction)
                        
                        conn.execute(
                            "UPDATE facts SET confidence = ? WHERE id = ?",
                            (new_confidence, fact_id)
                        )
                        
                        print(f"[ContradictionCausal] Applied confidence decay to fact {fact_id}: "
                              f"{current_confidence[0]:.3f} → {new_confidence:.3f} (decay: {confidence_reduction:.3f})")
                
                conn.commit()
                
        except Exception as e:
            print(f"[ContradictionCausal] Error applying contradiction effects: {e}")
    
    def get_contradiction_impact_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get a summary of recent contradiction impacts on causal reasoning."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (hours_back * 3600)
            
            with get_connection_pool().get_connection() as conn:
                # Count facts with recent volatility increases
                volatile_facts = conn.execute("""
                    SELECT COUNT(*) FROM facts 
                    WHERE volatility_score > 0.5 
                    AND timestamp >= ?
                """, (cutoff_time,)).fetchone()[0]
                
                # Count suspicious facts
                suspicious_facts = conn.execute("""
                    SELECT COUNT(*) FROM enhanced_facts 
                    WHERE change_history LIKE '%suspicious:%'
                """).fetchone()[0]
                
                # Count facts with reduced confidence
                low_confidence_facts = conn.execute("""
                    SELECT COUNT(*) FROM facts 
                    WHERE confidence < 0.7 
                    AND timestamp >= ?
                """, (cutoff_time,)).fetchone()[0]
                
                return {
                    "analysis_period_hours": hours_back,
                    "volatile_facts": volatile_facts,
                    "suspicious_facts": suspicious_facts,
                    "low_confidence_facts": low_confidence_facts,
                    "system_stability": "stable" if volatile_facts < 5 else "unstable",
                    "generated_at": current_time
                }
                
        except Exception as e:
            print(f"[ContradictionCausal] Error generating impact summary: {e}")
            return {"error": str(e)} 