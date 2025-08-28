"""
Summarization and Reflection Engine for MeRNSTA
Generates insights about user beliefs, volatility, and patterns
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter

from storage.enhanced_memory_model import EnhancedTripletFact
from storage.enhanced_contradiction_resolver import ContradictionResolver


class MemorySummarizer:
    """
    Generates intelligent summaries of memory facts,
    highlighting patterns, contradictions, and volatile beliefs.
    """
    
    def __init__(self):
        self.contradiction_resolver = ContradictionResolver()
    
    def summarize_user_facts(self, facts: List[EnhancedTripletFact], user_profile_id: Optional[str] = None) -> str:
        if not facts:
            return "I don't have any stored facts about you yet."
        # Group facts by (subject, object) using spaCy similarity
        from collections import defaultdict
        import spacy
        try:
            from config.settings import get_config
            config = get_config()
            spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
            nlp = spacy.load(spacy_model)
        except OSError:
            nlp = None
        clusters = defaultdict(list)
        for fact in facts:
            key = (fact.subject, fact.object)
            found = False
            for k in clusters:
                doc1 = nlp(fact.object)
                doc2 = nlp(k[1])
                if fact.subject == k[0] and doc1.similarity(doc2) > 0.85:
                    clusters[k].append(fact)
                    found = True
                    break
            if not found:
                clusters[key].append(fact)
        summary_parts = []
        for (subject, obj), group in clusters.items():
            if any(f.volatile for f in group):
                summary_parts.append(f"You’ve changed your mind multiple times about {obj}. Marked as volatile.")
            elif any(f.contradiction for f in group):
                summary_parts.append(f"You have conflicting beliefs about {obj}.")
            else:
                # Most recent fact
                latest = sorted(group, key=lambda f: f.timestamp)[-1]
                summary_parts.append(f"You {latest.predicate} {obj}.")
        return "\n".join(summary_parts)

    def generate_meta_goals(self, facts: List[EnhancedTripletFact]) -> list:
        import spacy
        try:
            from config.settings import get_config
            config = get_config()
            spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
            nlp = spacy.load(spacy_model)
        except OSError:
            nlp = None
        goals = []
        
        # Group by (subject, object) cluster
        clusters = defaultdict(list)
        for fact in facts:
            key = (fact.subject, fact.object)
            found = False
            for k in clusters:
                doc1 = nlp(fact.object)
                doc2 = nlp(k[1])
                if fact.subject == k[0] and doc1.similarity(doc2) > 0.85:
                    clusters[k].append(fact)
                    found = True
                    break
            if not found:
                clusters[key].append(fact)
        
        for (subject, obj), group in clusters.items():
            if any(f.volatile for f in group) or any(f.contradiction for f in group):
                goals.append(f"Clarify whether you currently like or dislike {obj}.")
        
        # Add drift-triggered goals
        try:
            from agents.cognitive_repair_agent import detect_drift_triggered_goals
            
            drift_goals = detect_drift_triggered_goals(limit=20)
            
            for goal in drift_goals:
                if goal.priority > 0.5:  # Only include high-priority goals
                    goals.append(goal.goal)
                    
        except ImportError:
            pass  # Cognitive repair agent not available
        
        return goals
    
    def _summarize_single_fact(self, fact: EnhancedTripletFact) -> str:
        """Summarize a single fact with confidence indicators"""
        sentence = self._fact_to_natural_language(fact)
        
        # Add confidence/hedge indicators
        if fact.hedge_detected:
            sentence = f"You seem uncertain, but {sentence}"
        elif fact.intensifier_detected:
            sentence = f"You strongly believe that {sentence}"
        elif fact.confidence < 0.5:
            sentence = f"You might think that {sentence}"
            
        return sentence
    
    def _summarize_fact_group(self, subject: str, predicate: str, 
                             facts: List[EnhancedTripletFact]) -> str:
        """Summarize a group of facts about the same subject-predicate"""
        # Sort by timestamp
        facts.sort(key=lambda f: f.timestamp)
        
        # Check for volatility
        volatility_score = self.contradiction_resolver._compute_volatility(facts)
        
        if volatility_score > 0.4:
            # High volatility - describe the changes
            objects = [f.object for f in facts]
            unique_objects = list(dict.fromkeys(objects))  # Preserve order
            
            if predicate in ["like", "love", "enjoy", "prefer"]:
                return (f"You've expressed mixed feelings about {subject}: "
                       f"you've said you {predicate} {', then '.join(unique_objects)}. "
                       f"Your opinion seems to change frequently.")
            elif predicate == "is":
                return (f"You've described {subject} differently over time: "
                       f"as {', then as '.join(unique_objects)}.")
            else:
                return (f"Your view on '{subject} {predicate}' has varied: "
                       f"{', then '.join(unique_objects)}.")
        else:
            # Low volatility - take the most recent
            latest_fact = facts[-1]
            sentence = self._fact_to_natural_language(latest_fact)
            
            if len(set(f.object for f in facts)) > 1:
                # Had changes but now stable
                old_object = facts[0].object
                if old_object != latest_fact.object:
                    sentence += f" (previously: {old_object})"
                    
            return sentence
    
    def _fact_to_natural_language(self, fact: EnhancedTripletFact) -> str:
        """Convert fact to natural language"""
        if fact.predicate == "is":
            return f"{fact.subject} is {fact.object}"
        elif fact.predicate in ["has", "have"]:
            return f"{fact.subject} has {fact.object}"
        elif fact.predicate.startswith("not_"):
            base_pred = fact.predicate[4:]
            return f"{fact.subject} doesn't {base_pred} {fact.object}"
        else:
            return f"{fact.subject} {fact.predicate} {fact.object}"
    
    def _generate_insights(self, facts: List[EnhancedTripletFact]) -> str:
        """Generate meta-level insights about the user's beliefs"""
        insights = []
        
        # Count volatile topics
        volatile_topics = self.contradiction_resolver.get_volatile_topics(facts)
        if volatile_topics:
            insights.append(f"📊 I notice you frequently change your mind about: "
                          f"{', '.join(f'{s} {p}' for s, p, _ in volatile_topics[:3])}")
        
        # Confidence patterns
        low_confidence_facts = [f for f in facts if f.confidence < 0.5]
        if len(low_confidence_facts) > len(facts) * 0.3:
            insights.append("🤔 You seem uncertain about many things - "
                          "lots of 'maybe' and 'probably' in your statements.")
        
        high_confidence_facts = [f for f in facts if f.confidence > 0.9]
        if len(high_confidence_facts) > len(facts) * 0.5:
            insights.append("💪 You express strong convictions about most things.")
        
        # Contradiction patterns
        contradictory_facts = [f for f in facts if f.contradiction]
        if contradictory_facts:
            insights.append(f"⚠️ You have {len(contradictory_facts)} contradictory statements "
                          f"that might need clarification.")
        
        return "\n".join(insights) if insights else ""
    
    def generate_reflection_report(self, facts: List[EnhancedTripletFact],
                                 time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate a detailed reflection report on memory patterns.
        
        Args:
            facts: List of facts to analyze
            time_window: Optional time window to focus on
            
        Returns:
            Dictionary with various analytics
        """
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            facts = [f for f in facts if f.timestamp > cutoff]
        
        report = {
            "total_facts": len(facts),
            "time_range": {
                "start": min(f.timestamp for f in facts) if facts else None,
                "end": max(f.timestamp for f in facts) if facts else None
            },
            "confidence_analysis": self._analyze_confidence(facts),
            "volatility_analysis": self._analyze_volatility(facts),
            "subject_frequency": self._analyze_subjects(facts),
            "predicate_patterns": self._analyze_predicates(facts),
            "contradiction_summary": self._analyze_contradictions(facts),
            "temporal_patterns": self._analyze_temporal_patterns(facts)
        }
        
        return report
    
    def _analyze_confidence(self, facts: List[EnhancedTripletFact]) -> Dict[str, Any]:
        """Analyze confidence patterns"""
        if not facts:
            return {"average": 0, "distribution": {}}
            
        confidences = [f.confidence for f in facts]
        
        return {
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "distribution": {
                "high": len([c for c in confidences if c > 0.8]),
                "medium": len([c for c in confidences if 0.5 <= c <= 0.8]),
                "low": len([c for c in confidences if c < 0.5])
            },
            "hedge_detected": len([f for f in facts if f.hedge_detected]),
            "intensifier_detected": len([f for f in facts if f.intensifier_detected])
        }
    
    def _analyze_volatility(self, facts: List[EnhancedTripletFact]) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        volatile_facts = [f for f in facts if f.volatile]
        volatile_topics = self.contradiction_resolver.get_volatile_topics(facts)
        
        return {
            "volatile_fact_count": len(volatile_facts),
            "volatile_topic_count": len(volatile_topics),
            "top_volatile_topics": [(s, p, score) for s, p, score in volatile_topics[:5]],
            "average_volatility": sum(f.volatility_score for f in facts) / len(facts) if facts else 0
        }
    
    def _analyze_subjects(self, facts: List[EnhancedTripletFact]) -> Dict[str, int]:
        """Analyze subject frequency"""
        subject_counter = Counter(f.subject for f in facts)
        return dict(subject_counter.most_common(10))
    
    def _analyze_predicates(self, facts: List[EnhancedTripletFact]) -> Dict[str, Any]:
        """Analyze predicate patterns"""
        predicate_counter = Counter(f.predicate for f in facts)
        
        # Categorize predicates
        preferences = ["like", "love", "hate", "prefer", "enjoy", "dislike"]
        states = ["is", "are", "was", "were"]
        possessions = ["has", "have", "had"]
        
        return {
            "frequency": dict(predicate_counter.most_common(10)),
            "categories": {
                "preferences": len([f for f in facts if f.predicate in preferences]),
                "states": len([f for f in facts if f.predicate in states]),
                "possessions": len([f for f in facts if f.predicate in possessions]),
                "negations": len([f for f in facts if f.predicate.startswith("not_")])
            }
        }
    
    def _analyze_contradictions(self, facts: List[EnhancedTripletFact]) -> Dict[str, Any]:
        """Analyze contradiction patterns"""
        contradictory_facts = [f for f in facts if f.contradiction]
        
        # Group by subject-predicate
        contradiction_groups = defaultdict(list)
        for fact in contradictory_facts:
            key = (fact.subject, fact.predicate)
            contradiction_groups[key].append(fact)
        
        return {
            "total_contradictions": len(contradictory_facts),
            "contradiction_groups": len(contradiction_groups),
            "top_contradictory_topics": [
                {"subject": s, "predicate": p, "count": len(facts)}
                for (s, p), facts in sorted(
                    contradiction_groups.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:5]
            ]
        }
    
    def _analyze_temporal_patterns(self, facts: List[EnhancedTripletFact]) -> Dict[str, Any]:
        """Analyze temporal patterns in fact creation"""
        if not facts:
            return {"facts_per_day": {}}
            
        # Group by date
        facts_by_date = defaultdict(int)
        for fact in facts:
            date = fact.timestamp.date()
            facts_by_date[date] += 1
        
        # Calculate averages
        total_days = len(facts_by_date)
        avg_facts_per_day = sum(facts_by_date.values()) / total_days if total_days > 0 else 0
        
        return {
            "facts_per_day": {str(date): count for date, count in facts_by_date.items()},
            "average_facts_per_day": avg_facts_per_day,
            "most_active_day": max(facts_by_date.items(), key=lambda x: x[1]) if facts_by_date else None
        } 