#!/usr/bin/env python3
"""
Causal Graph Auto-Compression System for MeRNSTA

This system periodically compresses linear causal chains into higher-order summaries:
1. Identifies linear causal chains (A → B → C → D)
2. Compresses them into meaningful summaries ("Work transition led to increased stress")
3. Stores compressed summaries while preserving original granular data
4. Provides both detailed and summary views of causal relationships

📌 DO NOT HARDCODE compression patterns or summary templates.
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
class CausalChain:
    """Represents a linear causal chain."""
    chain_id: str
    fact_ids: List[int]
    chain_description: str
    total_strength: float
    start_time: float
    end_time: float
    compression_summary: str = ""
    compression_confidence: float = 0.0


@dataclass
class CompressionPattern:
    """Represents a pattern for compressing causal chains."""
    pattern_type: str  # e.g., "work_stress", "learning_process", "relationship_change"
    keywords: List[str]
    template: str
    confidence_threshold: float
    min_chain_length: int


class CausalGraphCompression:
    """System for auto-compressing causal graphs into meaningful summaries."""
    
    def __init__(self):
        """Initialize the causal graph compression system."""
        # Load configuration parameters (no hardcoding)
        self.min_chain_length = DEFAULT_VALUES.get("min_compression_chain_length", 3)
        self.compression_threshold = DEFAULT_VALUES.get("compression_confidence_threshold", 0.6)
        self.max_time_gap_hours = DEFAULT_VALUES.get("max_compression_time_gap_hours", 72)
        self.compression_interval_hours = DEFAULT_VALUES.get("compression_interval_hours", 24)
        
        # Load compression patterns
        self.compression_patterns = self._load_compression_patterns()
        
        # Track last compression run
        self.last_compression_time = 0
        
        print(f"[CausalCompression] Initialized with min_length={self.min_chain_length}, "
              f"threshold={self.compression_threshold}")
    
    def _load_compression_patterns(self) -> List[CompressionPattern]:
        """Load compression patterns from configuration."""
        # TODO: Load from config file instead of hardcoding
        return [
            CompressionPattern(
                pattern_type="work_stress",
                keywords=["job", "work", "startup", "project", "deadline", "boss", "overtime"],
                template="Work-related changes led to {emotional_outcome}",
                confidence_threshold=0.7,
                min_chain_length=2
            ),
            CompressionPattern(
                pattern_type="learning_process", 
                keywords=["learn", "study", "practice", "skill", "course", "training"],
                template="Learning journey resulted in {outcome_state}",
                confidence_threshold=0.6,
                min_chain_length=3
            ),
            CompressionPattern(
                pattern_type="relationship_change",
                keywords=["friend", "partner", "family", "relationship", "social", "meet"],
                template="Social dynamics led to {emotional_outcome}",
                confidence_threshold=0.7,
                min_chain_length=2
            ),
            CompressionPattern(
                pattern_type="lifestyle_change",
                keywords=["exercise", "diet", "sleep", "health", "routine", "habit"],
                template="Lifestyle adjustments caused {behavioral_outcome}",
                confidence_threshold=0.6,
                min_chain_length=3
            ),
            CompressionPattern(
                pattern_type="general_sequence",
                keywords=[],  # Fallback pattern
                template="A sequence of events led to {final_state}",
                confidence_threshold=0.4,
                min_chain_length=3
            )
        ]
    
    def run_compression_cycle(self, user_id: str = None) -> Dict[str, Any]:
        """
        Run a compression cycle to identify and compress causal chains.
        
        Args:
            user_id: Optional specific user to compress for, or None for all users
            
        Returns:
            Dictionary with compression results and statistics
        """
        current_time = time.time()
        
        # Check if it's time to run compression
        if current_time - self.last_compression_time < (self.compression_interval_hours * 3600):
            return {"message": "Compression not needed yet", "next_run_in_hours": 
                   (self.compression_interval_hours * 3600 - (current_time - self.last_compression_time)) / 3600}
        
        try:
            print(f"[CausalCompression] Starting compression cycle...")
            
            # Find linear causal chains
            chains = self._identify_linear_chains(user_id)
            
            # Compress eligible chains
            compression_results = []
            for chain in chains:
                if len(chain.fact_ids) >= self.min_chain_length:
                    compression = self._compress_chain(chain)
                    if compression["success"]:
                        compression_results.append(compression)
            
            # Store compressed summaries
            stored_count = 0
            for result in compression_results:
                if self._store_compression(result):
                    stored_count += 1
            
            self.last_compression_time = current_time
            
            return {
                "compression_cycle_completed": True,
                "chains_found": len(chains),
                "chains_compressed": len(compression_results),
                "summaries_stored": stored_count,
                "next_compression_in_hours": self.compression_interval_hours
            }
            
        except Exception as e:
            print(f"[CausalCompression] Error in compression cycle: {e}")
            return {"error": f"Compression cycle failed: {str(e)}"}
    
    def _identify_linear_chains(self, user_id: str = None) -> List[CausalChain]:
        """Identify linear causal chains in the graph."""
        chains = []
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Query for facts with causal relationships
                query = """SELECT f.id, f.subject, f.predicate, f.object, f.timestamp,
                                 ef.causal_strength, ef.change_history
                          FROM facts f
                          JOIN enhanced_facts ef ON f.id = ef.id
                          WHERE ef.causal_strength > 0
                          ORDER BY f.timestamp ASC"""
                
                results = conn.execute(query).fetchall()
                
                # Build adjacency graph
                fact_graph = {}
                fact_details = {}
                
                for result in results:
                    fact_id = result[0]
                    fact_details[fact_id] = {
                        "id": fact_id,
                        "subject": result[1],
                        "predicate": result[2],
                        "object": result[3],
                        "timestamp": result[4],
                        "causal_strength": result[5],
                        "change_history": result[6] or ""
                    }
                    
                    # Parse causal relationships from change_history
                    # This is simplified - in production, would parse more sophisticated
                    if "causal:" in fact_details[fact_id]["change_history"]:
                        # Find what this fact is caused by
                        causal_info = fact_details[fact_id]["change_history"]
                        # Simple parsing - would be more sophisticated in production
                        causes = self._extract_causes_from_history(causal_info, fact_details)
                        
                        if causes:
                            fact_graph[fact_id] = causes
                
                # Find linear chains using DFS
                visited = set()
                for fact_id in fact_details:
                    if fact_id not in visited:
                        chain = self._find_chain_from_fact(fact_id, fact_graph, fact_details, visited)
                        if chain and len(chain.fact_ids) >= 2:
                            chains.append(chain)
                
        except Exception as e:
            print(f"[CausalCompression] Error identifying chains: {e}")
        
        return chains
    
    def _extract_causes_from_history(self, change_history: str, fact_details: Dict) -> List[int]:
        """Extract causal fact IDs from change history."""
        causes = []
        
        # Simple extraction - in production would be more sophisticated
        if "causal:" in change_history:
            # This is a simplified approach
            # In a full implementation, we'd parse structured causal information
            lines = change_history.split('\n')
            for line in lines:
                if "causal:" in line:
                    # Extract fact description and find matching fact ID
                    causal_desc = line.split("causal:")[1].split("(")[0].strip()
                    
                    # Find matching fact
                    for fact_id, details in fact_details.items():
                        fact_desc = f"{details['subject']} {details['predicate']} {details['object']}"
                        if causal_desc in fact_desc:
                            causes.append(fact_id)
                            break
        
        return causes
    
    def _find_chain_from_fact(self, start_fact_id: int, fact_graph: Dict, 
                             fact_details: Dict, visited: Set) -> Optional[CausalChain]:
        """Find a linear causal chain starting from a specific fact."""
        if start_fact_id in visited:
            return None
        
        # Trace backward to find the root of the chain
        chain_facts = []
        current_fact = start_fact_id
        
        # Build chain by following causal links
        chain_stack = [current_fact]
        processed = set()
        
        while chain_stack:
            fact_id = chain_stack.pop()
            if fact_id in processed:
                continue
                
            processed.add(fact_id)
            chain_facts.append(fact_id)
            
            # Add facts that this fact causes
            for next_fact, causes in fact_graph.items():
                if fact_id in causes and next_fact not in processed:
                    chain_stack.append(next_fact)
        
        if len(chain_facts) < 2:
            return None
        
        # Sort by timestamp to get proper order
        chain_facts.sort(key=lambda f: fact_details[f]["timestamp"] or 0)
        
        # Mark as visited
        visited.update(chain_facts)
        
        # Calculate chain properties
        total_strength = sum(fact_details[f]["causal_strength"] for f in chain_facts) / len(chain_facts)
        start_time = fact_details[chain_facts[0]]["timestamp"] or 0
        end_time = fact_details[chain_facts[-1]]["timestamp"] or 0
        
        # Create chain description
        descriptions = []
        for fact_id in chain_facts:
            fact = fact_details[fact_id]
            descriptions.append(f"{fact['predicate']} {fact['object']}")
        
        chain_description = " → ".join(descriptions)
        
        return CausalChain(
            chain_id=f"chain_{start_fact_id}_{int(time.time())}",
            fact_ids=chain_facts,
            chain_description=chain_description,
            total_strength=total_strength,
            start_time=start_time,
            end_time=end_time
        )
    
    def _compress_chain(self, chain: CausalChain) -> Dict[str, Any]:
        """Compress a causal chain into a higher-order summary."""
        try:
            # Find the best matching compression pattern
            best_pattern = None
            best_match_score = 0
            
            for pattern in self.compression_patterns:
                match_score = self._calculate_pattern_match(chain, pattern)
                if match_score > best_match_score and match_score >= pattern.confidence_threshold:
                    best_match_score = match_score
                    best_pattern = pattern
            
            if not best_pattern:
                # Use fallback general pattern
                best_pattern = self.compression_patterns[-1]  # general_sequence
                best_match_score = 0.5
            
            # Generate compression summary
            summary = self._generate_compression_summary(chain, best_pattern, best_match_score)
            
            return {
                "success": True,
                "chain_id": chain.chain_id,
                "original_description": chain.chain_description,
                "compression_summary": summary,
                "pattern_type": best_pattern.pattern_type,
                "compression_confidence": best_match_score,
                "fact_ids": chain.fact_ids,
                "total_strength": chain.total_strength
            }
            
        except Exception as e:
            print(f"[CausalCompression] Error compressing chain {chain.chain_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_pattern_match(self, chain: CausalChain, pattern: CompressionPattern) -> float:
        """Calculate how well a chain matches a compression pattern."""
        if len(chain.fact_ids) < pattern.min_chain_length:
            return 0.0
        
        # Check keyword overlap
        chain_text = chain.chain_description.lower()
        keyword_matches = sum(1 for keyword in pattern.keywords if keyword in chain_text)
        
        if not pattern.keywords:  # General pattern
            return 0.4  # Base score for general pattern
        
        keyword_score = keyword_matches / len(pattern.keywords)
        
        # Bonus for chain strength and length
        strength_bonus = min(0.3, chain.total_strength)
        length_bonus = min(0.2, (len(chain.fact_ids) - pattern.min_chain_length) * 0.05)
        
        return keyword_score + strength_bonus + length_bonus
    
    def _generate_compression_summary(self, chain: CausalChain, 
                                    pattern: CompressionPattern, confidence: float) -> str:
        """Generate a natural language summary for the compressed chain."""
        try:
            # Extract key elements from the chain
            chain_parts = chain.chain_description.split(" → ")
            
            if len(chain_parts) >= 2:
                first_event = chain_parts[0]
                last_event = chain_parts[-1]
                
                # Determine outcome type based on the final event
                outcome_state = self._classify_outcome(last_event)
                
                # Fill in the template
                if pattern.pattern_type == "work_stress":
                    summary = pattern.template.format(emotional_outcome=outcome_state)
                elif pattern.pattern_type == "learning_process":
                    summary = pattern.template.format(outcome_state=outcome_state)
                elif pattern.pattern_type == "relationship_change":
                    summary = pattern.template.format(emotional_outcome=outcome_state)
                elif pattern.pattern_type == "lifestyle_change":
                    summary = pattern.template.format(behavioral_outcome=outcome_state)
                else:  # general_sequence
                    summary = pattern.template.format(final_state=outcome_state)
                
                # Add confidence qualifier
                if confidence >= 0.8:
                    confidence_qualifier = ""
                elif confidence >= 0.6:
                    confidence_qualifier = "likely "
                else:
                    confidence_qualifier = "possibly "
                
                return f"{confidence_qualifier}{summary}"
            
            else:
                return f"Causal sequence: {chain.chain_description}"
                
        except Exception as e:
            print(f"[CausalCompression] Error generating summary: {e}")
            return f"Compressed causal chain: {chain.chain_description}"
    
    def _classify_outcome(self, final_event: str) -> str:
        """Classify the type of outcome from the final event in a chain."""
        event_lower = final_event.lower()
        
        # Emotional outcomes
        if any(emotion in event_lower 
               for emotion in ["stress", "anxious", "overwhelm", "worry", "frustrated"]):
            return "increased stress and emotional strain"
        elif any(emotion in event_lower
                 for emotion in ["happy", "confident", "satisfied", "excited", "joy"]):
            return "positive emotional outcomes"
        elif any(emotion in event_lower
                 for emotion in ["tired", "exhausted", "fatigue", "drained"]):
            return "fatigue and energy depletion"
        
        # Behavioral outcomes
        elif any(behavior in event_lower
                 for behavior in ["perform", "complete", "achieve", "accomplish"]):
            return "improved performance and achievement"
        elif any(behavior in event_lower
                 for behavior in ["struggle", "difficult", "challenge", "problem"]):
            return "behavioral challenges and difficulties"
        
        # Default
        else:
            return f"changes in {final_event}"
    
    def _store_compression(self, compression_result: Dict[str, Any]) -> bool:
        """Store a compression result in the database."""
        try:
            with get_connection_pool().get_connection() as conn:
                # Create compression summaries table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS causal_compressions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chain_id TEXT UNIQUE,
                        fact_ids TEXT,
                        original_description TEXT,
                        compression_summary TEXT,
                        pattern_type TEXT,
                        compression_confidence REAL,
                        total_strength REAL,
                        created_at REAL
                    )
                """)
                
                # Insert compression summary
                conn.execute("""
                    INSERT OR REPLACE INTO causal_compressions 
                    (chain_id, fact_ids, original_description, compression_summary, 
                     pattern_type, compression_confidence, total_strength, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    compression_result["chain_id"],
                    ",".join(map(str, compression_result["fact_ids"])),
                    compression_result["original_description"],
                    compression_result["compression_summary"],
                    compression_result["pattern_type"],
                    compression_result["compression_confidence"],
                    compression_result["total_strength"],
                    time.time()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"[CausalCompression] Error storing compression: {e}")
            return False
    
    def get_compressed_summaries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve stored compression summaries."""
        summaries = []
        
        try:
            with get_connection_pool().get_connection() as conn:
                results = conn.execute("""
                    SELECT chain_id, fact_ids, original_description, compression_summary,
                           pattern_type, compression_confidence, total_strength, created_at
                    FROM causal_compressions
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()
                
                for result in results:
                    summaries.append({
                        "chain_id": result[0],
                        "fact_ids": [int(x) for x in result[1].split(",") if x],
                        "original_description": result[2],
                        "compression_summary": result[3],
                        "pattern_type": result[4],
                        "compression_confidence": result[5],
                        "total_strength": result[6],
                        "created_at": result[7]
                    })
                    
        except Exception as e:
            print(f"[CausalCompression] Error retrieving summaries: {e}")
        
        return summaries 