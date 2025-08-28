# vector_memory/memory_source_logger.py

"""
Memory Source Logger for MeRNSTA

This module provides functionality to log memory recall sources in LLM prompts,
enabling transparency and traceability in hybrid memory intelligence.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

class MemorySourceLogger:
    """
    Logger for memory source attribution in LLM prompts.
    
    Tracks which memory backends contributed to retrieved information
    and formats this for inclusion in LLM context.
    """
    
    def __init__(self):
        self.source_history = []
        self.session_stats = {
            'total_queries': 0,
            'backend_usage': {},
            'fusion_events': 0
        }
    
    def log_memory_recall(self, query: str, results: List[Dict[str, Any]], 
                         strategy: str = "unknown") -> str:
        """
        Log memory recall event and return formatted source information.
        
        Args:
            query: The memory query that was executed
            results: List of memory results with source information
            strategy: The hybrid strategy used
            
        Returns:
            Formatted source attribution string for LLM context
        """
        timestamp = datetime.now().isoformat()
        
        # Extract source information
        sources = self._extract_source_info(results)
        
        # Log the event
        log_entry = {
            'timestamp': timestamp,
            'query': query,
            'strategy': strategy,
            'sources': sources,
            'result_count': len(results)
        }
        
        self.source_history.append(log_entry)
        self._update_session_stats(sources, strategy)
        
        # Format for LLM context
        source_context = self._format_source_context(sources, strategy)
        
        logging.info(f"🧠 Memory recall: {len(results)} results from {len(sources)} backends")
        
        return source_context
    
    def _extract_source_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and aggregate source information from results."""
        sources = {
            'backends': {},
            'fusion_info': {},
            'confidence_range': [1.0, 0.0],
            'total_results': len(results)
        }
        
        for result in results:
            memory_source = result.get('memory_source', {})
            backend = memory_source.get('backend', 'unknown')
            confidence = memory_source.get('confidence', 0.0)
            
            # Track backend usage
            if backend not in sources['backends']:
                sources['backends'][backend] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            
            sources['backends'][backend]['count'] += 1
            sources['backends'][backend]['confidences'].append(confidence)
            
            # Update confidence range
            sources['confidence_range'][0] = min(sources['confidence_range'][0], confidence)
            sources['confidence_range'][1] = max(sources['confidence_range'][1], confidence)
            
            # Track fusion information
            fusion_info = memory_source.get('fusion_info', {})
            if fusion_info:
                sources['fusion_info'] = fusion_info
        
        # Calculate average confidences
        for backend_info in sources['backends'].values():
            confidences = backend_info['confidences']
            backend_info['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
        
        return sources
    
    def _format_source_context(self, sources: Dict[str, Any], strategy: str) -> str:
        """Format source information for LLM context."""
        lines = [
            f"[MEMORY SOURCES - Strategy: {strategy}]"
        ]
        
        # Backend breakdown
        backend_lines = []
        for backend, info in sources['backends'].items():
            count = info['count']
            avg_conf = info['avg_confidence']
            
            # Map backend names to descriptions
            backend_desc = {
                'default': 'Semantic (Ollama)',
                'hrrformer': 'Symbolic (HRR)',
                'vecsymr': 'Analogical (VSA)',
                'hybrid_2': 'Hybrid (2 backends)',
                'hybrid_3': 'Hybrid (3 backends)'
            }.get(backend, backend)
            
            backend_lines.append(f"{backend_desc}: {count} results (conf: {avg_conf:.2f})")
        
        lines.extend(backend_lines)
        
        # Confidence range
        conf_min, conf_max = sources['confidence_range']
        lines.append(f"Confidence range: {conf_min:.2f} - {conf_max:.2f}")
        
        # Fusion information
        fusion_info = sources.get('fusion_info', {})
        if fusion_info:
            participating_backends = fusion_info.get('backends', [])
            if len(participating_backends) > 1:
                lines.append(f"Fusion: {len(participating_backends)} backends combined")
        
        lines.append(f"Total: {sources['total_results']} memory items retrieved")
        
        return "\n".join(lines)
    
    def _update_session_stats(self, sources: Dict[str, Any], strategy: str):
        """Update session-level statistics."""
        self.session_stats['total_queries'] += 1
        
        # Track backend usage
        for backend, info in sources['backends'].items():
            if backend not in self.session_stats['backend_usage']:
                self.session_stats['backend_usage'][backend] = 0
            self.session_stats['backend_usage'][backend] += info['count']
        
        # Track fusion events
        if sources.get('fusion_info'):
            self.session_stats['fusion_events'] += 1
    
    def get_session_summary(self) -> str:
        """Get summary of memory usage for the current session."""
        stats = self.session_stats
        
        lines = [
            "[MEMORY SESSION SUMMARY]",
            f"Total queries: {stats['total_queries']}",
            f"Fusion events: {stats['fusion_events']}"
        ]
        
        if stats['backend_usage']:
            lines.append("Backend usage:")
            total_retrievals = sum(stats['backend_usage'].values())
            
            for backend, count in stats['backend_usage'].items():
                percentage = (count / total_retrievals) * 100 if total_retrievals > 0 else 0
                lines.append(f"  {backend}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def get_recent_sources(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memory source events."""
        return self.source_history[-limit:] if self.source_history else []
    
    def create_llm_context_prompt(self, base_prompt: str, memory_sources: str) -> str:
        """
        Create an enhanced LLM prompt with memory source context.
        
        Args:
            base_prompt: The original prompt
            memory_sources: Formatted source information
            
        Returns:
            Enhanced prompt with memory source context
        """
        enhanced_prompt = f"""
{base_prompt}

{memory_sources}

Note: The above information was retrieved using MeRNSTA's hybrid memory intelligence, 
combining semantic search, symbolic reasoning (HRR), and analogical mapping (VSA) 
for comprehensive knowledge recall.
"""
        
        return enhanced_prompt.strip()
    
    def reset_session(self):
        """Reset session statistics."""
        self.session_stats = {
            'total_queries': 0,
            'backend_usage': {},
            'fusion_events': 0
        }
        # Keep history but clear session stats
        logging.info("🔄 Memory source logger session reset")

# Global instance for easy access
memory_source_logger = MemorySourceLogger()