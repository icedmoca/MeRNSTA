#!/usr/bin/env python3
"""
Self-Summarizing Cortex Report
The cortex introspects itself and summarizes recent memory, contradictions, and conflicts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from typing import Dict, List, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from storage.memory_log import MemoryLog
from storage.memory_utils import TripletFact
from config.settings import PERSONALITY_PROFILES, DEFAULT_PERSONALITY

class CortexReport:
    """
    Generates comprehensive reports about the memory system's state
    """
    
    def __init__(self, memory_log: MemoryLog):
        self.memory_log = memory_log
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cortex report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory_stats": self._get_memory_stats(),
            "recent_high_confidence_facts": self._get_recent_high_confidence_facts(),
            "unresolved_contradictions": self._get_unresolved_contradictions(),
            "personality_state": self._get_personality_state(),
            "dominant_subject_topics": self._get_dominant_subject_topics(),
            "emotional_tone_summary": self._get_emotional_tone_summary(),
            "memory_health": self._get_memory_health_metrics(),
            "compression_stats": self._get_compression_stats(),
            "trust_scores": self._get_subject_trust_scores()
        }
        return report
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get basic memory statistics"""
        stats = self.memory_log.get_memory_stats()
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        return {
            "total_messages": stats["total_messages"],
            "total_facts": len(facts),
            "high_confidence_facts": len([f for f in facts if getattr(f, 'decayed_confidence', 1.0) > 0.8]),
            "contradictory_facts": len([f for f in facts if f.contradiction_score > 0.3]),
            "volatile_facts": len([f for f in facts if f.volatility_score > 0.5]),
            "recent_facts_24h": len([f for f in facts if self._is_recent(f.timestamp, hours=24)]),
            "recent_facts_7d": len([f for f in facts if self._is_recent(f.timestamp, days=7)])
        }
    
    def _get_recent_high_confidence_facts(self, limit: int = 10) -> List[Dict]:
        """Get recent high-confidence facts"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        # Filter for high confidence and recent facts
        high_conf_facts = []
        for fact in facts:
            confidence = getattr(fact, 'decayed_confidence', 1.0)
            if confidence > 0.7 and self._is_recent(fact.timestamp, days=7):
                high_conf_facts.append({
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "confidence": confidence,
                    "timestamp": fact.timestamp,
                    "frequency": fact.frequency
                })
        
        # Sort by confidence and recency
        high_conf_facts.sort(key=lambda x: (x["confidence"], x["timestamp"]), reverse=True)
        return high_conf_facts[:limit]
    
    def _get_unresolved_contradictions(self) -> List[Dict]:
        """Get unresolved contradictions"""
        contradictions = self.memory_log.get_contradictions(resolved=False)
        
        unresolved = []
        for contradiction in contradictions:
            unresolved.append({
                "id": contradiction["id"],
                "fact_a": contradiction["fact_a_text"],
                "fact_b": contradiction["fact_b_text"],
                "confidence": contradiction["confidence"],
                "timestamp": contradiction["timestamp"]
            })
        
        return unresolved
    
    def _get_personality_state(self) -> Dict[str, Any]:
        """Get current personality state and characteristics"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        # Analyze personality indicators
        personality_indicators = {
            "loyalty_signals": 0,
            "skepticism_signals": 0,
            "emotional_signals": 0,
            "analytical_signals": 0
        }
        
        for fact in facts:
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            emotion_score = getattr(fact, 'emotion_score', 0.0)
            
            # Loyalty indicators
            if any(word in fact_text for word in ["friend", "trust", "loyal", "support"]):
                personality_indicators["loyalty_signals"] += 1
            
            # Skepticism indicators
            if any(word in fact_text for word in ["doubt", "question", "suspicious", "verify"]):
                personality_indicators["skepticism_signals"] += 1
            
            # Emotional indicators
            if emotion_score > 0.5:
                personality_indicators["emotional_signals"] += 1
            
            # Analytical indicators
            if any(word in fact_text for word in ["analyze", "data", "evidence", "logical"]):
                personality_indicators["analytical_signals"] += 1
        
        # Determine dominant personality
        max_signals = max(personality_indicators.values())
        dominant_personality = "neutral"
        
        if max_signals > 0:
            for personality, signals in personality_indicators.items():
                if signals == max_signals:
                    dominant_personality = personality.replace("_signals", "")
                    break
        
        return {
            "dominant_personality": dominant_personality,
            "personality_indicators": personality_indicators,
            "emotional_facts_ratio": personality_indicators["emotional_signals"] / len(facts) if facts else 0,
            "trust_level": personality_indicators["loyalty_signals"] / len(facts) if facts else 0
        }
    
    def _get_dominant_subject_topics(self, limit: int = 10) -> List[Dict]:
        """Get dominant subject topics"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        # Count facts by subject
        subject_counts = Counter()
        subject_confidence = defaultdict(list)
        subject_emotion = defaultdict(list)
        
        for fact in facts:
            subject_counts[fact.subject] += 1
            confidence = getattr(fact, 'decayed_confidence', 1.0)
            emotion = getattr(fact, 'emotion_score', 0.0)
            subject_confidence[fact.subject].append(confidence)
            subject_emotion[fact.subject].append(emotion)
        
        # Create topic summary
        topics = []
        for subject, count in subject_counts.most_common(limit):
            avg_confidence = sum(subject_confidence[subject]) / len(subject_confidence[subject])
            avg_emotion = sum(subject_emotion[subject]) / len(subject_emotion[subject])
            
            topics.append({
                "subject": subject,
                "fact_count": count,
                "average_confidence": avg_confidence,
                "average_emotion": avg_emotion,
                "recent_activity": len([f for f in facts if f.subject == subject and self._is_recent(f.timestamp, days=7)])
            })
        
        return topics
    
    def _get_emotional_tone_summary(self) -> Dict[str, Any]:
        """Get emotional tone summary"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        if not facts:
            return {"average_emotion": 0.0, "emotional_distribution": {}, "emotional_trend": "neutral"}
        
        emotion_scores = [getattr(f, 'emotion_score', 0.0) for f in facts]
        avg_emotion = sum(emotion_scores) / len(emotion_scores)
        
        # Categorize emotions
        emotional_distribution = {
            "very_positive": len([e for e in emotion_scores if e > 0.7]),
            "positive": len([e for e in emotion_scores if 0.3 < e <= 0.7]),
            "neutral": len([e for e in emotion_scores if -0.3 <= e <= 0.3]),
            "negative": len([e for e in emotion_scores if -0.7 <= e < -0.3]),
            "very_negative": len([e for e in emotion_scores if e < -0.7])
        }
        
        # Determine emotional trend
        recent_facts = [f for f in facts if self._is_recent(f.timestamp, days=3)]
        if recent_facts:
            recent_emotions = [getattr(f, 'emotion_score', 0.0) for f in recent_facts]
            recent_avg = sum(recent_emotions) / len(recent_emotions)
            
            if recent_avg > avg_emotion + 0.2:
                emotional_trend = "increasingly_positive"
            elif recent_avg < avg_emotion - 0.2:
                emotional_trend = "increasingly_negative"
            else:
                emotional_trend = "stable"
        else:
            emotional_trend = "insufficient_recent_data"
        
        return {
            "average_emotion": avg_emotion,
            "emotional_distribution": emotional_distribution,
            "emotional_trend": emotional_trend,
            "total_emotional_facts": len([e for e in emotion_scores if abs(e) > 0.3])
        }
    
    def _get_memory_health_metrics(self) -> Dict[str, Any]:
        """Get memory health metrics"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        if not facts:
            return {"health_score": 0.0, "issues": [], "recommendations": []}
        
        # Calculate health metrics
        contradiction_ratio = len([f for f in facts if f.contradiction_score > 0.3]) / len(facts)
        volatility_ratio = len([f for f in facts if f.volatility_score > 0.5]) / len(facts)
        low_confidence_ratio = len([f for f in facts if getattr(f, 'decayed_confidence', 1.0) < 0.5]) / len(facts)
        
        # Health score (0-1, higher is better)
        health_score = 1.0 - (contradiction_ratio * 0.4 + volatility_ratio * 0.3 + low_confidence_ratio * 0.3)
        
        # Identify issues
        issues = []
        if contradiction_ratio > 0.2:
            issues.append("High contradiction rate detected")
        if volatility_ratio > 0.3:
            issues.append("Many volatile facts detected")
        if low_confidence_ratio > 0.4:
            issues.append("Many low-confidence facts detected")
        
        # Generate recommendations
        recommendations = []
        if contradiction_ratio > 0.1:
            recommendations.append("Run contradiction resolution")
        if volatility_ratio > 0.2:
            recommendations.append("Review volatile facts for accuracy")
        if len(facts) > 1000:
            recommendations.append("Consider memory compression")
        
        return {
            "health_score": health_score,
            "contradiction_ratio": contradiction_ratio,
            "volatility_ratio": volatility_ratio,
            "low_confidence_ratio": low_confidence_ratio,
            "issues": issues,
            "recommendations": recommendations
        }
    
    def _get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        try:
            from storage.memory_compression import MemoryCompressionEngine
            compression_engine = MemoryCompressionEngine(self.memory_log)
            compression_engine.create_compression_log_table()
            return compression_engine.get_compression_stats()
        except ImportError:
            return {"compression_events": 0, "total_facts_compressed": 0, "current_facts": 0, "compression_ratio": 0.0}
    
    def _get_subject_trust_scores(self, limit: int = 10) -> List[Dict]:
        """Get subject-level trust scores"""
        facts = self.memory_log.get_all_facts(prune_contradictions=False)
        
        # Group facts by subject
        subject_facts = defaultdict(list)
        for fact in facts:
            subject_facts[fact.subject].append(fact)
        
        trust_scores = []
        for subject, subject_fact_list in subject_facts.items():
            if len(subject_fact_list) < 2:
                continue
            
            # Calculate trust metrics
            contradiction_freq = len([f for f in subject_fact_list if f.contradiction_score > 0.3]) / len(subject_fact_list)
            avg_volatility = sum(f.volatility_score for f in subject_fact_list) / len(subject_fact_list)
            avg_confidence = sum(getattr(f, 'decayed_confidence', 1.0) for f in subject_fact_list) / len(subject_fact_list)
            
            # Trust score (0-1, higher is better)
            trust_score = avg_confidence * (1 - contradiction_freq) * (1 - avg_volatility)
            
            trust_scores.append({
                "subject": subject,
                "trust_score": trust_score,
                "fact_count": len(subject_fact_list),
                "contradiction_frequency": contradiction_freq,
                "average_volatility": avg_volatility,
                "average_confidence": avg_confidence
            })
        
        # Sort by trust score (highest first)
        trust_scores.sort(key=lambda x: x["trust_score"], reverse=True)
        return trust_scores[:limit]
    
    def _is_recent(self, timestamp: str, days: int = None, hours: int = None) -> bool:
        """Check if a timestamp is recent"""
        try:
            fact_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            if days:
                return (current_time - fact_time).days <= days
            elif hours:
                return (current_time - fact_time).total_seconds() <= hours * 3600
            else:
                return False
        except Exception:
            return False
    
    def format_report_for_cli(self, report: Dict[str, Any]) -> str:
        """Format the report for CLI display"""
        lines = []
        lines.append("🧠 CORTEX SELF-REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report['timestamp']}")
        lines.append("")
        
        # Memory Stats
        lines.append("📊 MEMORY STATISTICS")
        lines.append("-" * 40)
        stats = report["memory_stats"]
        lines.append(f"Total Messages: {stats['total_messages']}")
        lines.append(f"Total Facts: {stats['total_facts']}")
        lines.append(f"High Confidence: {stats['high_confidence_facts']}")
        lines.append(f"Contradictory: {stats['contradictory_facts']}")
        lines.append(f"Volatile: {stats['volatile_facts']}")
        lines.append(f"Recent (24h): {stats['recent_facts_24h']}")
        lines.append(f"Recent (7d): {stats['recent_facts_7d']}")
        lines.append("")
        
        # Memory Health
        lines.append("🏥 MEMORY HEALTH")
        lines.append("-" * 40)
        health = report["memory_health"]
        lines.append(f"Health Score: {health['health_score']:.2f}/1.0")
        if health["issues"]:
            lines.append("Issues:")
            for issue in health["issues"]:
                lines.append(f"  ⚠️ {issue}")
        if health["recommendations"]:
            lines.append("Recommendations:")
            for rec in health["recommendations"]:
                lines.append(f"  💡 {rec}")
        lines.append("")
        
        # Personality State
        lines.append("🎭 PERSONALITY STATE")
        lines.append("-" * 40)
        personality = report["personality_state"]
        lines.append(f"Dominant: {personality['dominant_personality'].title()}")
        lines.append(f"Emotional Facts: {personality['emotional_facts_ratio']:.1%}")
        lines.append(f"Trust Level: {personality['trust_level']:.1%}")
        lines.append("")
        
        # Top Subjects
        lines.append("🏷️ TOP SUBJECTS")
        lines.append("-" * 40)
        for i, topic in enumerate(report["dominant_subject_topics"][:5], 1):
            lines.append(f"{i}. {topic['subject']} ({topic['fact_count']} facts, {topic['average_confidence']:.2f} conf)")
        lines.append("")
        
        # Emotional Tone
        lines.append("😊 EMOTIONAL TONE")
        lines.append("-" * 40)
        emotion = report["emotional_tone_summary"]
        lines.append(f"Average Emotion: {emotion['average_emotion']:.2f}")
        lines.append(f"Trend: {emotion['emotional_trend'].replace('_', ' ').title()}")
        lines.append("")
        
        # Unresolved Contradictions
        if report["unresolved_contradictions"]:
            lines.append("⚠️ UNRESOLVED CONTRADICTIONS")
            lines.append("-" * 40)
            for contradiction in report["unresolved_contradictions"][:3]:
                lines.append(f"• {contradiction['fact_a']} vs {contradiction['fact_b']}")
            lines.append("")
        
        # Trust Scores
        lines.append("🔒 TOP TRUSTED SUBJECTS")
        lines.append("-" * 40)
        for i, trust in enumerate(report["trust_scores"][:5], 1):
            lines.append(f"{i}. {trust['subject']} ({trust['trust_score']:.2f} trust)")
        
        return "\n".join(lines)
    
    def save_report_to_file(self, report: Dict[str, Any], filename: str = None):
        """Save report to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"cortex_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to {filename}")

def generate_cortex_report(memory_log: MemoryLog, output_format: str = "cli") -> str:
    """Generate and return a cortex report"""
    reporter = CortexReport(memory_log)
    report = reporter.generate_full_report()
    
    if output_format == "cli":
        return reporter.format_report_for_cli(report)
    elif output_format == "json":
        return json.dumps(report, indent=2, default=str)
    else:
        return report

if __name__ == "__main__":
    # Test the report generation
    memory_log = MemoryLog("memory.db")
    reporter = CortexReport(memory_log)
    
    print("Generating cortex report...")
    report = reporter.generate_full_report()
    
    # Display CLI format
    print(reporter.format_report_for_cli(report))
    
    # Save to file
    reporter.save_report_to_file(report) 