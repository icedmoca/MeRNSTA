#!/usr/bin/env python3
"""
DriftAnalysisAgent - Monitors token drift and semantic shifts

Tracks how tokens change meaning over time and provides insights into
belief evolution and semantic drift patterns.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from storage.token_graph import TokenPropagationGraph, TokenDriftEvent, TokenCluster
from storage.enhanced_memory_model import EnhancedTripletFact


@dataclass
class DriftAnalysis:
    """Analysis of token drift patterns."""
    token_id: int
    drift_score: float
    old_cluster: str
    new_cluster: str
    affected_facts: List[str]
    drift_type: str
    confidence: float
    timestamp: float
    semantic_shift_description: str


class DriftAnalysisAgent:
    """
    Autonomous agent that monitors and analyzes token drift patterns.
    
    Features:
    - Continuous drift monitoring
    - Semantic shift detection
    - Belief evolution tracking
    - Drift pattern analysis
    - Predictive drift modeling
    """
    
    def __init__(self, token_graph: Optional[TokenPropagationGraph] = None):
        self.token_graph = token_graph
        self.drift_history: List[DriftAnalysis] = []
        self.drift_patterns: Dict[str, List[float]] = defaultdict(list)
        self.semantic_evolution: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        
        # Configuration
        self.drift_threshold = 0.3
        self.analysis_interval = 3600  # 1 hour
        self.last_analysis = 0
        
        print(f"[DriftAnalysisAgent] Initialized drift analysis agent")
    
    def analyze_token_drift(self, token_id: int) -> Optional[DriftAnalysis]:
        """
        Analyze drift for a specific token.
        
        Args:
            token_id: Token to analyze
            
        Returns:
            DriftAnalysis if drift detected, None otherwise
        """
        if not self.token_graph:
            return None
        
        # Get recent drift events for this token
        drift_events = self.token_graph.get_token_drift_events(token_id, limit=5)
        
        if not drift_events:
            return None
        
        # Analyze the most recent drift event
        latest_event = drift_events[-1]
        
        # Calculate confidence based on cluster coherence
        old_cluster = self.token_graph.get_token_cluster(token_id)
        confidence = old_cluster.coherence_score if old_cluster else 0.5
        
        # Generate semantic shift description
        semantic_shift = self._describe_semantic_shift(latest_event)
        
        analysis = DriftAnalysis(
            token_id=token_id,
            drift_score=latest_event.drift_score,
            old_cluster=latest_event.old_semantic_cluster,
            new_cluster=latest_event.new_semantic_cluster,
            affected_facts=latest_event.affected_facts,
            drift_type=latest_event.drift_type,
            confidence=confidence,
            timestamp=latest_event.timestamp,
            semantic_shift_description=semantic_shift
        )
        
        # Store in history
        self.drift_history.append(analysis)
        self.semantic_evolution[token_id].append((latest_event.new_semantic_cluster, latest_event.timestamp))
        
        return analysis
    
    def analyze_all_drift(self) -> List[DriftAnalysis]:
        """
        Analyze drift for all tokens with recent drift events.
        
        Returns:
            List of drift analyses
        """
        if not self.token_graph:
            return []
        
        # Get recent drift events
        recent_events = self.token_graph.get_token_drift_events(limit=50)
        
        analyses = []
        analyzed_tokens = set()
        
        for event in recent_events:
            if event.token_id not in analyzed_tokens:
                analysis = self.analyze_token_drift(event.token_id)
                if analysis:
                    analyses.append(analysis)
                    analyzed_tokens.add(event.token_id)
        
        return analyses
    
    def detect_drift_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in token drift over time.
        
        Returns:
            Dictionary of drift patterns
        """
        patterns = {
            "frequent_drifters": [],
            "stable_tokens": [],
            "drift_clusters": [],
            "temporal_patterns": {}
        }
        
        # Analyze drift frequency
        token_drift_counts = defaultdict(int)
        for analysis in self.drift_history:
            token_drift_counts[analysis.token_id] += 1
        
        # Find frequent drifters
        frequent_drifters = [(tid, count) for tid, count in token_drift_counts.items() if count > 2]
        patterns["frequent_drifters"] = sorted(frequent_drifters, key=lambda x: x[1], reverse=True)
        
        # Find stable tokens (no drift)
        all_tokens = set(self.token_graph.graph.keys()) if self.token_graph else set()
        drifted_tokens = set(token_drift_counts.keys())
        stable_tokens = list(all_tokens - drifted_tokens)
        patterns["stable_tokens"] = stable_tokens[:10]  # Top 10 stable tokens
        
        # Analyze temporal patterns
        for analysis in self.drift_history:
            hour = time.strftime("%H", time.localtime(analysis.timestamp))
            if hour not in patterns["temporal_patterns"]:
                patterns["temporal_patterns"][hour] = 0
            patterns["temporal_patterns"][hour] += 1
        
        return patterns
    
    def predict_drift_risk(self, token_id: int) -> float:
        """
        Predict the risk of future drift for a token.
        
        Args:
            token_id: Token to analyze
            
        Returns:
            Drift risk score (0.0 to 1.0)
        """
        if not self.token_graph or token_id not in self.token_graph.graph:
            return 0.0
        
        node = self.token_graph.graph[token_id]
        
        # Factors that increase drift risk:
        # 1. High usage count (more exposure to different contexts)
        # 2. High entropy (diverse usage patterns)
        # 3. Recent drift events
        # 4. Low cluster coherence
        
        usage_factor = min(1.0, node.usage_count / 50.0)  # Normalize usage
        entropy_factor = node.entropy_score / 5.0  # Normalize entropy
        
        # Recent drift factor
        recent_drift = len([a for a in self.drift_history if a.token_id == token_id and 
                          time.time() - a.timestamp < 86400])  # Last 24 hours
        recent_drift_factor = min(1.0, recent_drift / 3.0)
        
        # Cluster coherence factor (inverse)
        cluster_coherence = 0.0
        if node.semantic_cluster and node.semantic_cluster in self.token_graph.clusters:
            cluster_coherence = self.token_graph.clusters[node.semantic_cluster].coherence_score
        coherence_factor = 1.0 - cluster_coherence
        
        # Combined risk score
        risk_score = (
            usage_factor * 0.3 +
            entropy_factor * 0.3 +
            recent_drift_factor * 0.3 +
            coherence_factor * 0.1
        )
        
        return min(1.0, risk_score)
    
    def generate_drift_report(self) -> str:
        """
        Generate a comprehensive drift analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.token_graph:
            return "Token graph not available for drift analysis."
        
        # Get recent analyses
        recent_analyses = self.analyze_all_drift()
        
        # Get patterns
        patterns = self.detect_drift_patterns()
        
        lines = ["🔄 Token Drift Analysis Report"]
        lines.append("=" * 50)
        
        # Summary statistics
        lines.append(f"Recent drift events: {len(recent_analyses)}")
        lines.append(f"Total tokens analyzed: {len(self.token_graph.graph)}")
        lines.append(f"Frequent drifters: {len(patterns['frequent_drifters'])}")
        lines.append(f"Stable tokens: {len(patterns['stable_tokens'])}")
        
        # Recent drift events
        if recent_analyses:
            lines.append("\n📊 Recent Drift Events:")
            for i, analysis in enumerate(recent_analyses[:5], 1):
                lines.append(f"{i}. Token {analysis.token_id}:")
                lines.append(f"   {analysis.old_cluster} → {analysis.new_cluster}")
                lines.append(f"   Drift score: {analysis.drift_score:.3f}")
                lines.append(f"   Confidence: {analysis.confidence:.3f}")
                lines.append(f"   {analysis.semantic_shift_description}")
        
        # Frequent drifters
        if patterns['frequent_drifters']:
            lines.append("\n🔄 Frequent Drifters:")
            for token_id, count in patterns['frequent_drifters'][:5]:
                risk = self.predict_drift_risk(token_id)
                lines.append(f"  - Token {token_id}: {count} drifts, risk: {risk:.3f}")
        
        # Temporal patterns
        if patterns['temporal_patterns']:
            lines.append("\n⏰ Temporal Drift Patterns:")
            sorted_hours = sorted(patterns['temporal_patterns'].items(), key=lambda x: x[1], reverse=True)
            for hour, count in sorted_hours[:3]:
                lines.append(f"  - Hour {hour}: {count} drift events")
        
        return "\n".join(lines)
    
    def _describe_semantic_shift(self, event: TokenDriftEvent) -> str:
        """
        Generate a human-readable description of semantic shift.
        
        Args:
            event: Drift event to describe
            
        Returns:
            Description string
        """
        old_cluster = self.token_graph.clusters.get(event.old_semantic_cluster)
        new_cluster = self.token_graph.clusters.get(event.new_semantic_cluster)
        
        if not old_cluster or not new_cluster:
            return "Semantic shift detected"
        
        old_size = len(old_cluster.token_ids)
        new_size = len(new_cluster.token_ids)
        old_facts = len(old_cluster.fact_ids)
        new_facts = len(new_cluster.fact_ids)
        
        if event.drift_score > 0.7:
            intensity = "major"
        elif event.drift_score > 0.4:
            intensity = "moderate"
        else:
            intensity = "minor"
        
        if new_size > old_size:
            context_change = "expanded semantic context"
        elif new_size < old_size:
            context_change = "narrowed semantic context"
        else:
            context_change = "shifted semantic context"
        
        return f"{intensity} {context_change} (affects {len(event.affected_facts)} facts)"
    
    def run_continuous_analysis(self):
        """Run continuous drift analysis in background."""
        while True:
            try:
                current_time = time.time()
                
                if current_time - self.last_analysis > self.analysis_interval:
                    # Run analysis
                    analyses = self.analyze_all_drift()
                    
                    if analyses:
                        print(f"[DriftAnalysisAgent] Detected {len(analyses)} new drift events")
                        
                        # Generate report for significant drifts
                        significant_drifts = [a for a in analyses if a.drift_score > 0.5]
                        if significant_drifts:
                            report = self.generate_drift_report()
                            print(f"[DriftAnalysisAgent] Significant drift detected:\n{report}")
                    
                    self.last_analysis = current_time
                
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print("[DriftAnalysisAgent] Continuous analysis stopped")
                break
            except Exception as e:
                print(f"[DriftAnalysisAgent] Error in continuous analysis: {e}")
                time.sleep(60)


# Global drift analysis agent instance
_drift_agent_instance = None


def get_drift_agent() -> DriftAnalysisAgent:
    """Get or create the global drift analysis agent instance."""
    global _drift_agent_instance
    
    if _drift_agent_instance is None:
        from storage.token_graph import get_token_graph
        token_graph = get_token_graph()
        _drift_agent_instance = DriftAnalysisAgent(token_graph)
    
    return _drift_agent_instance


def analyze_token_drift(token_id: int) -> Optional[DriftAnalysis]:
    """Convenience function to analyze token drift."""
    return get_drift_agent().analyze_token_drift(token_id)


def generate_drift_report() -> str:
    """Convenience function to generate drift report."""
    return get_drift_agent().generate_drift_report() 