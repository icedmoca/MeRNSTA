#!/usr/bin/env python3
"""
Background Auto-Reconciliation System
Runs every 30s or after each message to detect and resolve contradictions
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from storage.memory_log import MemoryLog
from storage.memory_utils import (TripletFact, calculate_contradiction_score,
                                  detect_contradictions, get_sentiment_score)

# Use background logger to avoid interfering with user input
background_logger = logging.getLogger("background")


@dataclass
class ContradictionRanking:
    """Represents a contradiction with ranking factors"""

    fact_a: TripletFact
    fact_b: TripletFact
    contradiction_score: float
    recency_score: float  # Based on timestamp
    emotion_score: float  # Average emotion score
    confidence_score: float  # Average confidence
    overall_rank: float  # Combined ranking score


class AutoReconciliationEngine:
    """
    Background engine that automatically detects and resolves contradictions
    """

    def __init__(self, memory_log: MemoryLog, check_interval: int = 30):
        self.memory_log = memory_log
        self.check_interval = check_interval  # seconds
        self.running = False
        self.thread = None
        self.last_check_time = 0

        # Configuration
        self.auto_resolve_threshold = (
            0.7  # Contradiction score threshold for auto-resolution
        )
        self.llm_escalation_threshold = 0.8  # Threshold for LLM escalation
        self.low_confidence_threshold = 0.3  # Threshold for low confidence facts
        self.old_fact_threshold_days = 7  # Days to consider a fact "old"

    def start_background_loop(self):
        """Start the background reconciliation loop (now handled by Celery)"""
        # Background tasks are now handled by Celery task queue
        # This method is kept for backward compatibility
        background_logger.info(
            "🔄 Auto-reconciliation now handled by Celery task queue"
        )

    def stop_background_loop(self):
        """Stop the background reconciliation loop (now handled by Celery)"""
        # Background tasks are now handled by Celery task queue
        # This method is kept for backward compatibility
        background_logger.info(
            "🛑 Auto-reconciliation now handled by Celery task queue"
        )

    def _background_loop(self):
        """Main background loop for contradiction detection and resolution (deprecated)"""
        # This method is deprecated - use Celery tasks instead
        background_logger.warning("_background_loop is deprecated - use Celery tasks")
        pass

    def trigger_check(self):
        """Manually trigger a contradiction check (called after each message)"""
        self._check_and_resolve_contradictions()

    def _check_and_resolve_contradictions(self):
        """Check for contradictions and attempt resolution"""
        background_logger.info("🔍 Running background contradiction check...")

        # Get all facts
        facts = self.memory_log.get_all_facts(prune_contradictions=False)

        if len(facts) < 2:
            background_logger.info("Not enough facts for contradiction check")
            return

        background_logger.info(f"Checking {len(facts)} facts for contradictions")

        # Find contradictions
        contradictions = self._find_all_contradictions(facts)

        background_logger.info(f"Found {len(contradictions)} contradictions")

        if not contradictions:
            background_logger.info("No contradictions found")
            return

        # Log the contradictions found
        for i, (fact_a, fact_b, score) in enumerate(contradictions):
            background_logger.info(
                f"Contradiction {i+1}: {fact_a.subject} {fact_a.predicate} {fact_a.object} vs {fact_b.subject} {fact_b.predicate} {fact_b.object} (score: {score:.2f})"
            )

        # Rank contradictions by severity
        ranked_contradictions = self._rank_contradictions(contradictions)

        # Process top contradictions
        for ranked_contradiction in ranked_contradictions[:3]:  # Process top 3
            # Prefer higher frequency or more recent facts
            a, b, score = (
                ranked_contradiction.fact_a,
                ranked_contradiction.fact_b,
                ranked_contradiction.contradiction_score,
            )
            conf_a = getattr(a, "confidence", getattr(a, "decayed_confidence", 1.0))
            conf_b = getattr(b, "confidence", getattr(b, "decayed_confidence", 1.0))
            freq_a = getattr(a, "frequency", 1)
            freq_b = getattr(b, "frequency", 1)
            ts_a = a.timestamp
            ts_b = b.timestamp

            background_logger.info(
                f"Processing contradiction: A({conf_a:.2f}) vs B({conf_b:.2f}), score: {score:.2f}"
            )

            # If strong contradiction and low confidence, delete/downgrade
            if score > 0.8:
                if conf_a < 0.6 and conf_b >= 0.6:
                    self._delete_fact(a)
                    background_logger.info(
                        f"🗑️ Auto-deleted low-confidence fact: {a.subject} {a.predicate} {a.object} (conf: {conf_a:.2f})"
                    )
                    continue
                elif conf_b < 0.6 and conf_a >= 0.6:
                    self._delete_fact(b)
                    background_logger.info(
                        f"🗑️ Auto-deleted low-confidence fact: {b.subject} {b.predicate} {b.object} (conf: {conf_b:.2f})"
                    )
                    continue
                # If both low confidence, delete the older one
                elif conf_a < 0.6 and conf_b < 0.6:
                    if ts_a < ts_b:
                        self._delete_fact(a)
                        background_logger.info(
                            f"🗑️ Auto-deleted older low-confidence fact: {a.subject} {a.predicate} {a.object} (conf: {conf_a:.2f})"
                        )
                    else:
                        self._delete_fact(b)
                        background_logger.info(
                            f"🗑️ Auto-deleted older low-confidence fact: {b.subject} {b.predicate} {b.object} (conf: {conf_b:.2f})"
                        )
                    continue

            # Otherwise, prefer higher frequency or more recent
            if freq_a > freq_b:
                self._mark_as_volatile(b, a)
                background_logger.info(
                    f"🔥 Marked lower-frequency fact as volatile: {b.subject} {b.predicate} {b.object}"
                )
            elif freq_b > freq_a:
                self._mark_as_volatile(a, b)
                background_logger.info(
                    f"🔥 Marked lower-frequency fact as volatile: {a.subject} {a.predicate} {a.object}"
                )
            else:
                # If frequencies equal, prefer more recent
                if ts_a > ts_b:
                    self._mark_as_volatile(b, a)
                    background_logger.info(
                        f"🔥 Marked older fact as volatile: {b.subject} {b.predicate} {b.object}"
                    )
                else:
                    self._mark_as_volatile(a, b)
                    background_logger.info(
                        f"🔥 Marked older fact as volatile: {a.subject} {a.predicate} {a.object}"
                    )

        # After reconciliation, consolidate facts
        self.memory_log.consolidate_facts()

    def _find_all_contradictions(
        self, facts: List[TripletFact]
    ) -> List[Tuple[TripletFact, TripletFact, float]]:
        """Find all contradictions between facts"""
        contradictions = []

        for i, fact_a in enumerate(facts):
            for fact_b in facts[i + 1 :]:
                # Skip if both facts have low contradiction scores
                if (
                    fact_a.contradiction_score < 0.3
                    and fact_b.contradiction_score < 0.3
                ):
                    continue

                # Calculate contradiction score between these two facts
                contradiction_score = self._calculate_contradiction_score(
                    fact_a, fact_b
                )

                if contradiction_score > 0.3:  # Threshold for contradiction
                    contradictions.append((fact_a, fact_b, contradiction_score))

        return contradictions

    def _calculate_contradiction_score(
        self, fact_a: TripletFact, fact_b: TripletFact
    ) -> float:
        """Calculate contradiction score between two facts"""
        return calculate_contradiction_score(fact_a, fact_b)

    def _rank_contradictions(
        self, contradictions: List[Tuple[TripletFact, TripletFact, float]]
    ) -> List[ContradictionRanking]:
        """Rank contradictions by recency, emotion, and confidence"""
        ranked = []

        for fact_a, fact_b, contradiction_score in contradictions:
            # Calculate recency score (newer facts get higher scores)
            recency_a = self._calculate_recency_score(fact_a.timestamp)
            recency_b = self._calculate_recency_score(fact_b.timestamp)
            recency_score = max(recency_a, recency_b)

            # Calculate emotion score (average of both facts)
            emotion_a = getattr(fact_a, "emotion_score", 0.0)
            emotion_b = getattr(fact_b, "emotion_score", 0.0)
            emotion_score = (emotion_a + emotion_b) / 2

            # Calculate confidence score (average of both facts)
            confidence_a = getattr(fact_a, "decayed_confidence", 1.0)
            confidence_b = getattr(fact_b, "decayed_confidence", 1.0)
            confidence_score = (confidence_a + confidence_b) / 2

            # Calculate overall rank (weighted combination)
            overall_rank = (
                contradiction_score * 0.4
                + recency_score * 0.3
                + emotion_score * 0.2
                + confidence_score * 0.1
            )

            ranked.append(
                ContradictionRanking(
                    fact_a=fact_a,
                    fact_b=fact_b,
                    contradiction_score=contradiction_score,
                    recency_score=recency_score,
                    emotion_score=emotion_score,
                    confidence_score=confidence_score,
                    overall_rank=overall_rank,
                )
            )

        # Sort by overall rank (highest first)
        ranked.sort(key=lambda x: x.overall_rank, reverse=True)
        return ranked

    def _calculate_recency_score(self, timestamp: str) -> float:
        """Calculate recency score based on timestamp (newer = higher score)"""
        try:
            from datetime import datetime

            fact_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            current_time = datetime.now()
            days_old = (current_time - fact_time).days

            # Exponential decay: newer facts get higher scores
            recency_score = max(0.0, 1.0 - (days_old / 30.0))  # 30 days = 0 score
            return recency_score
        except Exception:
            return 0.5  # Default score if timestamp parsing fails

    def _process_contradiction(self, ranked_contradiction: ContradictionRanking):
        """Process a single contradiction based on ranking factors"""
        fact_a = ranked_contradiction.fact_a
        fact_b = ranked_contradiction.fact_b
        contradiction_score = ranked_contradiction.contradiction_score

        background_logger.info(
            f"⚖️ Processing contradiction (score: {contradiction_score:.2f}):"
        )
        background_logger.info(
            f"   A: {fact_a.subject} {fact_a.predicate} {fact_a.object}"
        )
        background_logger.info(
            f"   B: {fact_b.subject} {fact_b.predicate} {fact_b.object}"
        )

        # Strategy 1: Auto-resolve low-confidence + old facts
        if self._should_auto_delete(fact_a, fact_b):
            fact_to_delete = self._select_fact_to_delete(fact_a, fact_b)
            self._delete_fact(fact_to_delete)
            background_logger.info(
                f"🗑️ Auto-deleted low-confidence/old fact: {fact_to_delete.subject} {fact_to_delete.predicate} {fact_to_delete.object}"
            )
            return

        # Strategy 2: LLM escalation for strong contradictions
        if contradiction_score > self.llm_escalation_threshold:
            self._escalate_to_llm(fact_a, fact_b)
            return

        # Strategy 3: Mark as volatile for manual review
        self._mark_as_volatile(fact_a, fact_b)
        background_logger.info(f"🔥 Marked facts as volatile for manual review")

    def _should_auto_delete(self, fact_a: TripletFact, fact_b: TripletFact) -> bool:
        """Determine if facts should be auto-deleted based on confidence and age"""
        confidence_a = getattr(fact_a, "decayed_confidence", 1.0)
        confidence_b = getattr(fact_b, "decayed_confidence", 1.0)

        # Check if one fact is low confidence and old
        age_a = self._get_fact_age_days(fact_a.timestamp)
        age_b = self._get_fact_age_days(fact_b.timestamp)

        low_conf_old_a = (
            confidence_a < self.low_confidence_threshold
            and age_a > self.old_fact_threshold_days
        )
        low_conf_old_b = (
            confidence_b < self.low_confidence_threshold
            and age_b > self.old_fact_threshold_days
        )

        return low_conf_old_a or low_conf_old_b

    def _get_fact_age_days(self, timestamp: str) -> int:
        """Get age of fact in days"""
        try:
            from datetime import datetime

            fact_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            current_time = datetime.now()
            return (current_time - fact_time).days
        except Exception:
            return 0

    def _select_fact_to_delete(
        self, fact_a: TripletFact, fact_b: TripletFact
    ) -> TripletFact:
        """Select which fact to delete based on confidence and age"""
        confidence_a = getattr(fact_a, "decayed_confidence", 1.0)
        confidence_b = getattr(fact_b, "decayed_confidence", 1.0)
        age_a = self._get_fact_age_days(fact_a.timestamp)
        age_b = self._get_fact_age_days(fact_b.timestamp)

        # Delete the fact with lower confidence and older age
        score_a = confidence_a - (age_a / 100.0)  # Age penalty
        score_b = confidence_b - (age_b / 100.0)

        return fact_a if score_a < score_b else fact_b

    def _delete_fact(self, fact: TripletFact):
        """Delete a fact from the database"""
        with sqlite3.connect(self.memory_log.db_path) as conn:
            conn.execute("DELETE FROM facts WHERE id = ?", (fact.id,))
            conn.commit()

    def _escalate_to_llm(self, fact_a: TripletFact, fact_b: TripletFact):
        """Escalate contradiction to LLM for resolution"""
        prompt = f"""You have contradictory beliefs:

A: "{fact_a.subject} {fact_a.predicate} {fact_a.object}"
B: "{fact_b.subject} {fact_b.predicate} {fact_b.object}"

Should I forget A or B? Consider:
- Recency (A: {fact_a.timestamp}, B: {fact_b.timestamp})
- Confidence (A: {getattr(fact_a, 'decayed_confidence', 1.0):.2f}, B: {getattr(fact_b, 'decayed_confidence', 1.0):.2f})
- Emotion (A: {getattr(fact_a, 'emotion_score', 0.0):.2f}, B: {getattr(fact_b, 'emotion_score', 0.0):.2f})

Answer with just "A" or "B":"""

        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=20,
            )
            response.raise_for_status()
            decision = response.json()["response"].strip().upper()

            if decision == "A":
                self._delete_fact(fact_a)
                background_logger.info(
                    f"🤖 LLM decided to delete A: {fact_a.subject} {fact_a.predicate} {fact_a.object}"
                )
            elif decision == "B":
                self._delete_fact(fact_b)
                background_logger.info(
                    f"🤖 LLM decided to delete B: {fact_b.subject} {fact_b.predicate} {fact_b.object}"
                )
            else:
                background_logger.info(f"🤖 LLM couldn't decide, marking as volatile")
                self._mark_as_volatile(fact_a, fact_b)

        except Exception as e:
            background_logger.exception("❌ LLM escalation failed")
            self._mark_as_volatile(fact_a, fact_b)

    def _mark_as_volatile(self, fact_a: TripletFact, fact_b: TripletFact):
        """Mark facts as volatile for manual review"""
        with sqlite3.connect(self.memory_log.db_path) as conn:
            # Increase volatility scores
            conn.execute(
                "UPDATE facts SET volatility_score = MIN(1.0, volatility_score + 0.3) WHERE id IN (?, ?)",
                (fact_a.id, fact_b.id),
            )
            conn.commit()

    def reconcile_by_trajectory(
        self,
        subject: str,
        object_: str,
        trajectory_slope: float,
        suggested_deletions: List[int],
    ) -> Dict[str, Any]:
        """
        Reconcile contradictions based on sentiment trajectory analysis.

        Args:
            subject: Subject of the facts
            object_: Object of the facts
            trajectory_slope: Slope of sentiment trajectory
            suggested_deletions: List of fact IDs suggested for deletion

        Returns:
            Dictionary with reconciliation results
        """
        background_logger.info(
            f"🔄 Starting trajectory-based reconciliation for '{subject} {object_}' (slope: {trajectory_slope:.3f})"
        )

        results = {
            "subject": subject,
            "object": object_,
            "trajectory_slope": trajectory_slope,
            "suggested_deletions": suggested_deletions,
            "deleted_facts": [],
            "kept_facts": [],
            "errors": [],
        }

        try:
            # Get all facts for this subject-object pair
            all_facts = self.memory_log.get_all_facts(prune_contradictions=False)
            target_facts = [
                f
                for f in all_facts
                if f.subject.lower().strip() == subject.lower().strip()
                and f.object.lower().strip() == object_.lower().strip()
            ]

            if not target_facts:
                background_logger.warning(f"No facts found for '{subject} {object_}'")
                results["errors"].append("No facts found")
                return results

            # Determine trajectory direction
            trajectory_direction = "positive" if trajectory_slope > 0 else "negative"
            background_logger.info(f"Trajectory direction: {trajectory_direction}")

            # Process suggested deletions
            for fact_id in suggested_deletions:
                try:
                    # Find the fact
                    fact = next((f for f in target_facts if f.id == fact_id), None)
                    if not fact:
                        background_logger.warning(f"Fact {fact_id} not found")
                        results["errors"].append(f"Fact {fact_id} not found")
                        continue

                    # Verify the fact still opposes the trajectory
                    sentiment_score = get_sentiment_score(fact.predicate)
                    fact_is_positive = sentiment_score > 0
                    opposes_trajectory = (
                        trajectory_slope > 0 and not fact_is_positive
                    ) or (trajectory_slope < 0 and fact_is_positive)

                    if opposes_trajectory:
                        # Delete the fact
                        if self.memory_log.delete_fact(fact_id):
                            background_logger.info(
                                f"✅ Deleted opposing fact {fact_id}: '{fact.subject} {fact.predicate} {fact.object}'"
                            )
                            results["deleted_facts"].append(
                                {
                                    "id": fact_id,
                                    "text": f"{fact.subject} {fact.predicate} {fact.object}",
                                    "sentiment": sentiment_score,
                                    "timestamp": fact.timestamp,
                                }
                            )
                        else:
                            background_logger.error(
                                f"❌ Failed to delete fact {fact_id}"
                            )
                            results["errors"].append(f"Failed to delete fact {fact_id}")
                    else:
                        background_logger.info(
                            f"ℹ️ Fact {fact_id} no longer opposes trajectory, keeping"
                        )
                        results["kept_facts"].append(fact_id)

                except Exception as e:
                    background_logger.error(f"Error processing fact {fact_id}: {e}")
                    results["errors"].append(
                        f"Error processing fact {fact_id}: {str(e)}"
                    )

            # Consolidate facts after reconciliation
            self.memory_log.consolidate_facts()

            background_logger.info(
                f"✅ Trajectory reconciliation completed: {len(results['deleted_facts'])} deleted, {len(results['kept_facts'])} kept"
            )

        except Exception as e:
            background_logger.error(f"❌ Error in trajectory reconciliation: {e}")
            results["errors"].append(f"Reconciliation error: {str(e)}")

        return results
