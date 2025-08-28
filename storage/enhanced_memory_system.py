"""
Enhanced Memory System for MeRNSTA
Integrates all components for a complete memory-augmented AI system
"""

import sqlite3
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

from storage.enhanced_memory_model import EnhancedTripletFact, ContradictionRecord
from storage.enhanced_triplet_extractor import EnhancedTripletExtractor
from storage.enhanced_contradiction_resolver import ContradictionResolver
from storage.enhanced_semantic_search import SemanticMemorySearchEngine
from storage.enhanced_summarization import MemorySummarizer

# Enhanced components for predictive causal modeling
try:
    from storage.enhanced_memory import BeliefAbstractionLayer
    from storage.reflex_compression import ReflexCompressor
    from storage.memory_autocleaner import MemoryCleaner
    from storage.enhanced_reasoning import EnhancedReasoningEngine
    from storage.causal_drift_predictor import CausalDriftPredictor
    from agents.hypothesis_generator import HypothesisGeneratorAgent
    from agents.reflex_anticipator import ReflexAnticipator
    from storage.causal_audit_dashboard import CausalAuditDashboard
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False


class EnhancedMemorySystem:
    """
    Complete enhanced memory system with:
    - Dynamic NLP-based triplet extraction
    - Contradiction resolution and volatility tracking
    - Semantic search capabilities
    - Intelligent summarization
    - Command support
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db",
                 ollama_host: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        self.db_path = db_path
        self.extractor = EnhancedTripletExtractor()
        self.contradiction_resolver = ContradictionResolver()
        # Don't require Ollama for basic functionality
        self.search_engine = SemanticMemorySearchEngine(
            ollama_host=None,  # Disable Ollama for now
            ollama_model=None
        )
        self.summarizer = MemorySummarizer()
        
        # Initialize database
        self._init_database()
        
        # Cache for performance
        self._fact_cache = {}
        
    def _init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                contradiction BOOLEAN DEFAULT FALSE,
                volatile BOOLEAN DEFAULT FALSE,
                volatility_score REAL DEFAULT 0.0,
                user_profile_id TEXT,
                session_id TEXT,
                source_message_id INTEGER,
                hedge_detected BOOLEAN DEFAULT FALSE,
                intensifier_detected BOOLEAN DEFAULT FALSE,
                negation BOOLEAN DEFAULT FALSE,
                embedding TEXT,
                change_history TEXT,
                causal_strength REAL DEFAULT 0.0,
                cause TEXT,
                active BOOLEAN DEFAULT TRUE,
                token_count INTEGER,
                token_entropy REAL,
                token_ids TEXT,
                token_hash TEXT,
                causal_token_propagation TEXT,
                UNIQUE(subject, predicate, object, user_profile_id, session_id)
            )
        """)
        
        # Add causal_strength and cause columns if they don't exist (migration support)
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN causal_strength REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN cause TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add token-related columns if they don't exist (migration support)
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN token_count INTEGER")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN token_entropy REAL")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN token_ids TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN token_hash TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN causal_token_propagation TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add emotion-related columns if they don't exist (Phase 8 migration support)
        emotion_columns = [
            ("emotion_valence", "REAL"),
            ("emotion_arousal", "REAL"),
            ("emotion_tag", "TEXT"),
            ("emotional_strength", "REAL DEFAULT 0.0"),
            ("emotion_source", "TEXT"),
            ("mood_context", "TEXT")
        ]
        
        for column_name, column_type in emotion_columns:
            try:
                cursor.execute(f"ALTER TABLE enhanced_facts ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        # Contradiction records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradiction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (fact_a_id) REFERENCES enhanced_facts(id),
                FOREIGN KEY (fact_b_id) REFERENCES enhanced_facts(id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON enhanced_facts(user_profile_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON enhanced_facts(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate ON enhanced_facts(subject, predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON enhanced_facts(timestamp)")
        
        conn.commit()
        conn.close()
    
    def process_input(self, text: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        result = {
            "extracted_facts": [],
            "contradictions": [],
            "stored_facts": [],
            "query_results": [],
            "response": "",
            "command_result": None
        }
        
        text_lower = text.lower().strip()
        
        # Enhanced command routing - check at the very top before any extraction/storage
        # Summarization commands
        summarization_patterns = [
            r"summarize|summary|tell me what you know|what do you know about me",
            r"remind me what|recap what|overview of what",
            r"what have we talked about|what did we discuss|conversation so far",
            r"changed my mind|what are my opinions"
        ]
        
        for pattern in summarization_patterns:
            if re.search(pattern, text_lower):
                print("[CommandRouter] Routed to summarizer (top of handler)")
                facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                summary = self.summarizer.summarize_user_facts(facts, user_profile_id)
                result["response"] = summary
                return result
        
        # Meta-goal commands  
        meta_goal_patterns = [
            r"meta-?goal|generate some meta-?goals|/generate_meta_goals",
            r"suggest questions|clarify|clarification questions",
            r"what should i think about|help me understand"
        ]
        
        for pattern in meta_goal_patterns:
            if re.search(pattern, text_lower):
                print("[CommandRouter] Routed to meta-goal generator (top of handler)")
                facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                volatile_topics = self.contradiction_resolver.get_volatile_topics(facts)
                if not volatile_topics:
                    result["response"] = "No volatile topics detected. Your beliefs seem stable!"
                else:
                    questions = self.contradiction_resolver.suggest_clarification_questions(volatile_topics)
                    result["response"] = "\n".join(questions)
                return result
        
        # Query detection (enhanced heuristic for better query handling)
        is_question = (
            text.strip().endswith("?") or 
            any(text_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how"]) or
            any(w in text_lower for w in ["do i", "am i", "can i", "should i", "would i"])
        )
        
        if is_question:
            print(f"[QueryDetection] Detected query: '{text}'")
            facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            print(f"[QueryDetection] Retrieved {len(facts)} facts for search")
            
            search_results = self.search_engine.query_memory(text, facts, user_profile_id=user_profile_id, session_id=session_id)
            result["query_results"] = search_results
            
            intent = self.search_engine.extract_query_intent(text)
            response = self.search_engine.generate_query_response(text, search_results, intent)
            
            # ENHANCED: If no good results found, try broader search
            if not search_results or ("don't have" in response.lower() and "information" in response.lower()):
                print(f"[QueryDetection] No results found, trying broader search...")
                
                # Try searching all facts without session filtering
                all_facts = self.get_facts(user_profile_id=user_profile_id)
                broader_results = self.search_engine.query_memory(text, all_facts, user_profile_id=user_profile_id)
                
                if broader_results:
                    response = self.search_engine.generate_query_response(text, broader_results, intent)
                    print(f"[QueryDetection] Broader search found {len(broader_results)} results")
                else:
                    # Final fallback: simple text search
                    text_keywords = [word for word in text_lower.split() if len(word) > 3]
                    for fact in all_facts:
                        fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                        if any(keyword in fact_text for keyword in text_keywords):
                            response = f"You {fact.predicate} {fact.object} (confidence: {fact.confidence:.2f})"
                            print(f"[QueryDetection] Fallback search found: {fact_text}")
                            break
            
            result["response"] = response
            return result
        
        # Otherwise, treat as statement: extract and store facts
        extracted_facts = self.extractor.extract_triplets(text, user_profile_id, session_id)
        result["extracted_facts"] = extracted_facts
        
        contradictions_detected = 0
        volatile_topics = []
        
        if extracted_facts:
            for fact in extracted_facts:
                existing_facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                contradictions = self.contradiction_resolver.check_for_contradictions(fact, existing_facts)
                result["contradictions"].extend(contradictions)
                contradictions_detected += len(contradictions)
                
                stored_id = self._store_fact(fact)
                if stored_id:
                    fact.id = stored_id
                    result["stored_facts"].append(fact)
                
                # Store contradiction records
                for contradiction in contradictions:
                    self._store_contradiction(contradiction)
                    
                    # Phase 26: Hook into dissonance tracking
                    try:
                        from agents.dissonance_tracker import get_dissonance_tracker
                        
                        # Convert contradiction to dissonance tracking format
                        dissonance_data = {
                            'belief_id': getattr(contradiction, 'id', f"contradiction_{int(time.time())}"),
                            'source_belief': getattr(contradiction, 'fact_a_text', str(fact)),
                            'target_belief': getattr(contradiction, 'fact_b_text', 'existing_belief'),
                            'semantic_distance': getattr(contradiction, 'semantic_distance', 0.5),
                            'confidence': getattr(contradiction, 'confidence', 0.8)
                        }
                        
                        tracker = get_dissonance_tracker()
                        tracker.process_contradiction(dissonance_data)
                        
                    except Exception as e:
                        # Don't let dissonance tracking failures break memory processing
                        logging.warning(f"[EnhancedMemory] Dissonance tracking failed: {e}")
            
            # Update volatility for all facts after processing
            all_facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            volatile_topics = self.contradiction_resolver.get_volatile_topics(all_facts)
            
            # Update volatility flags for facts in volatile topics
            for subject, predicate, volatility in volatile_topics:
                for fact in all_facts:
                    if fact.subject == subject and fact.predicate == predicate:
                        fact.volatility_score = volatility
                        fact.volatile = volatility > self.contradiction_resolver.volatility_threshold
                        self._update_fact(fact)
            
            # Generate response
            response = f"I've stored {len(result['stored_facts'])} fact(s) from your input."
            if contradictions_detected > 0:
                response += f" ⚠️ Detected {contradictions_detected} contradiction(s)."
            if volatile_topics:
                volatile_topic_names = [f'{s}' for s, p, _ in volatile_topics]
                response += f" 🔥 Volatile topic detected: {', '.join(volatile_topic_names)} - you've changed your mind multiple times about this."
            result["response"] = response
        else:
            result["response"] = "I couldn't extract any specific facts from your input."
        
        return result
    
    def _handle_command(self, command: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle slash commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/list_facts":
            facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id, limit=20)
            if not facts:
                return {"response": "No facts stored yet."}
            
            lines = ["📚 Your stored facts (top 20 by confidence):"]
            for i, fact in enumerate(facts, 1):
                icons = ""
                if fact.contradiction:
                    icons += "⚠️"
                if fact.volatile:
                    icons += "🔥"
                confidence_bar = "█" * int(fact.confidence * 5)
                lines.append(
                    f"{i}. {icons} {fact.subject} {fact.predicate} {fact.object} "
                    f"[{confidence_bar}] {fact.confidence:.1f}"
                )
            
            return {"response": "\n".join(lines)}
        
        elif cmd == "/show_contradictions":
            contradictions = self._get_contradictions(user_profile_id, session_id)
            if not contradictions:
                return {"response": "No contradictions found. Your beliefs appear consistent!"}
            
            lines = ["⚠️ Detected Contradictions:"]
            for i, record in enumerate(contradictions, 1):
                status = "✅ RESOLVED" if record["resolved"] else "❌ UNRESOLVED"
                lines.append(f"\n{i}. {status}")
                lines.append(f"   Fact A: {record['fact_a_text']}")
                lines.append(f"   Fact B: {record['fact_b_text']}")
                if record["resolution_notes"]:
                    lines.append(f"   Resolution: {record['resolution_notes']}")
            
            return {"response": "\n".join(lines)}
        
        elif cmd == "/volatility_report":
            facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            report = self.summarizer.generate_reflection_report(facts)
            
            lines = ["📊 Volatility Report:"]
            vol_analysis = report["volatility_analysis"]
            lines.append(f"Volatile facts: {vol_analysis['volatile_fact_count']}")
            lines.append(f"Volatile topics: {vol_analysis['volatile_topic_count']}")
            
            if vol_analysis["top_volatile_topics"]:
                lines.append("\nMost volatile topics:")
                for subj, pred, score in vol_analysis["top_volatile_topics"]:
                    lines.append(f"  - {subj} {pred} (volatility: {score:.2f})")
            
            return {"response": "\n".join(lines)}
        
        elif cmd.startswith("/token_graph"):
            return self._handle_token_graph_command(cmd, user_profile_id, session_id)
        
        elif cmd == "/token_clusters":
            return self._handle_token_clusters_command(user_profile_id, session_id)
        
        elif cmd == "/token_drift":
            return self._handle_token_drift_command(user_profile_id, session_id)
        
        elif cmd == "/token_influence":
            return self._handle_token_influence_command(user_profile_id, session_id)
        
        elif cmd == "/drift_analysis":
            return self._handle_drift_analysis_command(user_profile_id, session_id)
        
        elif cmd == "/drift_goals":
            return self._handle_drift_goals_command(user_profile_id, session_id)
        
        elif cmd == "/reflex_log":
            return self._handle_reflex_log_command(user_profile_id, session_id)
        
        elif cmd == "/reflex_scores":
            return self._handle_reflex_scores_command(user_profile_id, session_id)
        
        elif cmd == "/strategy_optimization":
            return self._handle_strategy_optimization_command(user_profile_id, session_id)
        
        elif cmd == "/anticipated_drifts":
            return self._handle_anticipated_drifts_command(user_profile_id, session_id)
        
        elif cmd == "/strategy_stats":
            return self._handle_strategy_stats_command(user_profile_id, session_id)
        
        elif cmd == "/meta_stats":
            return self._handle_meta_stats_command(user_profile_id, session_id)
        
        # New enhanced commands for Belief Abstraction + Reflex Compression + Memory Autoclean
        
        elif cmd == "/beliefs":
            return self._handle_beliefs_command(user_profile_id, session_id)
        
        elif cmd == "/reflex_templates":
            return self._handle_reflex_templates_command(user_profile_id, session_id)
        
        elif cmd == "/memory_clean_log":
            return self._handle_memory_clean_log_command(user_profile_id, session_id)
        
        elif cmd.startswith("/belief_trace"):
            return self._handle_belief_trace_command(cmd, user_profile_id, session_id)
        
        # New enhanced commands for Predictive Causal Modeling
        elif cmd == "/hypotheses":
            return self._handle_hypotheses_command(user_profile_id, session_id)
        
        elif cmd.startswith("/confirm_hypothesis"):
            return self._handle_confirm_hypothesis_command(cmd, user_profile_id, session_id)
        
        elif cmd.startswith("/reject_hypothesis"):
            return self._handle_reject_hypothesis_command(cmd, user_profile_id, session_id)
        
        elif cmd == "/causal_dashboard":
            return self._handle_causal_dashboard_command(user_profile_id, session_id)
        
        else:
            return {"response": f"Unknown command: {cmd}. Available commands: "
                               "/list_facts, /show_contradictions, /summarize, "
                               "/generate_meta_goals, /volatility_report, "
                               "/token_graph, /token_clusters, /token_drift, /token_influence, "
                               "/drift_analysis, /drift_goals, /reflex_log, /reflex_scores, "
                               "/strategy_optimization, /anticipated_drifts, /strategy_stats, /meta_stats"}
    
    def _store_fact(self, fact: EnhancedTripletFact) -> Optional[int]:
        """Store a fact in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert embedding to string if present
            embedding_str = str(fact.embedding) if fact.embedding else None
            change_history_str = str(fact.change_history) if fact.change_history else "[]"
            
            # Ensure causal_strength and cause have values
            causal_strength = getattr(fact, 'causal_strength', 0.0)
            cause = getattr(fact, 'cause', None)
            
            # Handle token-related fields
            token_ids_str = json.dumps(fact.token_ids) if fact.token_ids else None
            causal_token_propagation_str = json.dumps(fact.causal_token_propagation) if fact.causal_token_propagation else None
            
            # Handle timestamp properly
            if isinstance(fact.timestamp, (int, float)):
                timestamp = str(fact.timestamp)
            elif hasattr(fact.timestamp, 'isoformat'):
                timestamp = fact.timestamp.isoformat()
            else:
                timestamp = str(fact.timestamp)
            
            # Handle emotion-related fields (Phase 8)
            emotion_valence = getattr(fact, 'emotion_valence', None)
            emotion_arousal = getattr(fact, 'emotion_arousal', None)
            emotion_tag = getattr(fact, 'emotion_tag', None)
            emotional_strength = getattr(fact, 'emotional_strength', 0.0)
            emotion_source = getattr(fact, 'emotion_source', None)
            mood_context = getattr(fact, 'mood_context', None)
            
            cursor.execute("""
                INSERT OR REPLACE INTO enhanced_facts 
                (subject, predicate, object, timestamp, confidence, contradiction,
                 volatile, volatility_score, user_profile_id, session_id,
                 source_message_id, hedge_detected, intensifier_detected, negation,
                 embedding, change_history, causal_strength, cause,
                 token_count, token_entropy, token_ids, token_hash, causal_token_propagation,
                 emotion_valence, emotion_arousal, emotion_tag, emotional_strength, emotion_source, mood_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.subject, fact.predicate, fact.object,
                timestamp, fact.confidence, fact.contradiction,
                fact.volatile, fact.volatility_score, fact.user_profile_id,
                fact.session_id, fact.source_message_id, fact.hedge_detected,
                fact.intensifier_detected, fact.negation, embedding_str,
                change_history_str, causal_strength, cause,
                fact.token_count, fact.token_entropy, token_ids_str, fact.token_hash, causal_token_propagation_str,
                emotion_valence, emotion_arousal, emotion_tag, emotional_strength, emotion_source, mood_context
            ))
            
            fact_id = cursor.lastrowid
            conn.commit()
            
            # Log token propagation in token graph
            if fact.token_ids and fact_id:
                try:
                    from storage.token_graph import add_token_propagation
                    for token in fact.token_ids:
                        add_token_propagation(token, str(fact_id), fact.timestamp)
                    print(f"[EnhancedMemory] Logged {len(fact.token_ids)} token propagations for fact {fact_id}")
                except ImportError:
                    print("[EnhancedMemory] Token graph not available, skipping token propagation logging")
                except Exception as e:
                    print(f"[EnhancedMemory] Error logging token propagation: {e}")
            
            return fact_id
        except Exception as e:
            logging.error(f"Error storing fact: {e}")
            print(f"[EnhancedMemory] Error storing fact: {e}")
            return None
        finally:
            conn.close()
    
    def _update_fact(self, fact: EnhancedTripletFact):
        """Update an existing fact"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Include causal_strength and cause in the update
            causal_strength = getattr(fact, 'causal_strength', 0.0)
            cause = getattr(fact, 'cause', None)
            
            cursor.execute("""
                UPDATE enhanced_facts 
                SET contradiction = ?, volatile = ?, volatility_score = ?,
                    change_history = ?, causal_strength = ?, cause = ?
                WHERE id = ?
            """, (
                fact.contradiction, fact.volatile, fact.volatility_score,
                str(fact.change_history), causal_strength, cause, fact.id
            ))
            conn.commit()
        except Exception as e:
            logging.error(f"Error updating fact: {e}")
        finally:
            conn.close()

    def update_fact_causal_info(self, fact_id: int, cause: str, causal_strength: float):
        """Update a fact with causal information in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert numpy types to Python types for JSON serialization
            causal_strength = float(causal_strength)  # Convert numpy float32 to Python float
            
            # Update the causal_strength column directly
            cursor.execute("""
                UPDATE enhanced_facts 
                SET causal_strength = ?, cause = ?
                WHERE id = ?
            """, (causal_strength, cause, fact_id))
            
            # Also update change_history to track the causal link
            import time
            import json
            cursor.execute("""
                UPDATE enhanced_facts 
                SET change_history = CASE 
                    WHEN change_history IS NULL OR change_history = '' OR change_history = '[]'
                    THEN ?
                    ELSE json_insert(change_history, '$[#]', ?)
                END
                WHERE id = ?
            """, (
                json.dumps([{
                    'action': 'causal_link_added', 
                    'cause': str(cause), 
                    'strength': causal_strength,  # Now a Python float
                    'timestamp': time.time()
                }]),
                json.dumps({
                    'action': 'causal_link_added', 
                    'cause': str(cause), 
                    'strength': causal_strength,  # Now a Python float
                    'timestamp': time.time()
                }),
                fact_id
            ))
            
            conn.commit()
            print(f"[EnhancedMemory] Updated fact {fact_id} with causal strength: {causal_strength:.3f}")
            
        except Exception as e:
            logging.error(f"Error updating causal info for fact {fact_id}: {e}")
            print(f"[EnhancedMemory] Error updating causal info for fact {fact_id}: {e}")
        finally:
            conn.close()
    
    def _store_contradiction(self, contradiction: ContradictionRecord):
        """Store a contradiction record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO contradiction_records
                (fact_a_id, fact_b_id, timestamp, resolved, resolution_notes, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                contradiction.fact_a_id, contradiction.fact_b_id,
                contradiction.timestamp if isinstance(contradiction.timestamp, (int, float)) else contradiction.timestamp.isoformat(), contradiction.resolved,
                contradiction.resolution_notes, contradiction.confidence
            ))
            conn.commit()
        except Exception as e:
            logging.error(f"Error storing contradiction: {e}")
        finally:
            conn.close()
    
    def add_emotional_context_to_fact(self, fact_id: int, valence: float, arousal: float,
                                    emotion_tag: str = "", strength: float = 1.0,
                                    source: str = "system", mood_context: str = ""):
        """
        Add emotional context to an existing fact.
        
        Args:
            fact_id: ID of the fact to tag
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            emotion_tag: Emotion label
            strength: Strength of emotional association
            source: Source of emotional tagging
            mood_context: Current mood context
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clamp values to valid ranges
            valence = max(-1.0, min(1.0, valence))
            arousal = max(0.0, min(1.0, arousal))
            strength = max(0.0, min(1.0, strength))
            
            cursor.execute("""
                UPDATE enhanced_facts 
                SET emotion_valence = ?, emotion_arousal = ?, emotion_tag = ?,
                    emotional_strength = ?, emotion_source = ?, mood_context = ?
                WHERE id = ?
            """, (valence, arousal, emotion_tag, strength, source, mood_context, fact_id))
            
            # Add to change history
            change_record = {
                "action": "emotional_tagging",
                "valence": valence,
                "arousal": arousal,
                "emotion_tag": emotion_tag,
                "strength": strength,
                "source": source,
                "timestamp": time.time()
            }
            
            cursor.execute("""
                UPDATE enhanced_facts 
                SET change_history = CASE 
                    WHEN change_history IS NULL OR change_history = '' OR change_history = '[]'
                    THEN ?
                    ELSE json_insert(change_history, '$[#]', ?)
                END
                WHERE id = ?
            """, (json.dumps([change_record]), json.dumps(change_record), fact_id))
            
            conn.commit()
            logging.info(f"[EnhancedMemory] Added emotional context to fact {fact_id}: "
                        f"valence={valence:.2f}, arousal={arousal:.2f}, tag={emotion_tag}")
            
        except Exception as e:
            logging.error(f"Error adding emotional context to fact {fact_id}: {e}")
        finally:
            conn.close()
    
    def get_facts_by_emotion(self, emotion_tag: Optional[str] = None,
                           valence_range: Optional[Tuple[float, float]] = None,
                           arousal_range: Optional[Tuple[float, float]] = None,
                           user_profile_id: Optional[str] = None,
                           limit: Optional[int] = None) -> List[EnhancedTripletFact]:
        """
        Retrieve facts filtered by emotional characteristics.
        
        Args:
            emotion_tag: Filter by specific emotion tag
            valence_range: Filter by valence range (min, max)
            arousal_range: Filter by arousal range (min, max)
            user_profile_id: Filter by user profile
            limit: Maximum number of results
            
        Returns:
            List of facts matching emotional criteria
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query with emotional filters
        query_parts = ["SELECT * FROM enhanced_facts WHERE active = 1"]
        params = []
        
        if emotion_tag:
            query_parts.append("AND emotion_tag = ?")
            params.append(emotion_tag)
        
        if valence_range:
            query_parts.append("AND emotion_valence BETWEEN ? AND ?")
            params.extend(valence_range)
        
        if arousal_range:
            query_parts.append("AND emotion_arousal BETWEEN ? AND ?")
            params.extend(arousal_range)
        
        if user_profile_id:
            query_parts.append("AND user_profile_id = ?")
            params.append(user_profile_id)
        
        # Only include facts that have emotional data
        query_parts.append("AND (emotion_valence IS NOT NULL OR emotion_arousal IS NOT NULL)")
        
        query_parts.append("ORDER BY timestamp DESC")
        
        if limit:
            query_parts.append("LIMIT ?")
            params.append(limit)
        
        query = " ".join(query_parts)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_fact(row) for row in rows]
        except Exception as e:
            logging.error(f"Error retrieving facts by emotion: {e}")
            return []
        finally:
            conn.close()
    
    def get_emotional_summary(self, user_profile_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            hours_back: float = 24.0) -> Dict[str, Any]:
        """
        Get emotional summary of recent facts.
        
        Args:
            user_profile_id: Filter by user profile
            session_id: Filter by session
            hours_back: How many hours back to analyze
            
        Returns:
            Dictionary with emotional statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate cutoff time
        cutoff_time = time.time() - (hours_back * 3600)
        
        query_parts = ["""
            SELECT emotion_valence, emotion_arousal, emotion_tag, emotional_strength, mood_context
            FROM enhanced_facts 
            WHERE active = 1 
            AND (emotion_valence IS NOT NULL OR emotion_arousal IS NOT NULL)
            AND timestamp > ?
        """]
        params = [cutoff_time]
        
        if user_profile_id:
            query_parts.append("AND user_profile_id = ?")
            params.append(user_profile_id)
        
        if session_id:
            query_parts.append("AND session_id = ?")
            params.append(session_id)
        
        query = " ".join(query_parts)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return {"emotional_facts_count": 0, "message": "No emotional data found"}
            
            # Calculate statistics
            valences = [row[0] for row in rows if row[0] is not None]
            arousals = [row[1] for row in rows if row[1] is not None]
            emotion_tags = [row[2] for row in rows if row[2]]
            mood_contexts = [row[4] for row in rows if row[4]]
            
            from collections import Counter
            
            summary = {
                "emotional_facts_count": len(rows),
                "avg_valence": sum(valences) / len(valences) if valences else 0.0,
                "avg_arousal": sum(arousals) / len(arousals) if arousals else 0.0,
                "valence_range": (min(valences), max(valences)) if valences else (0.0, 0.0),
                "arousal_range": (min(arousals), max(arousals)) if arousals else (0.0, 0.0),
                "emotion_tags": dict(Counter(emotion_tags)),
                "mood_contexts": dict(Counter(mood_contexts)),
                "hours_analyzed": hours_back
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting emotional summary: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def detect_emotional_triggers(self, fact: EnhancedTripletFact) -> Dict[str, Any]:
        """
        Detect what emotional response this fact should trigger based on its content.
        
        Args:
            fact: The fact to analyze
            
        Returns:
            Dictionary with suggested emotional context
        """
        try:
            # Import emotion model to use its event mappings
            from .emotion_model import get_emotion_model
            emotion_model = get_emotion_model(self.db_path)
            
            # Analyze fact content for emotional triggers
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            
            # Check for contradiction
            if fact.contradiction:
                return {
                    "event_type": "contradiction",
                    "valence": -0.6,
                    "arousal": 0.7,
                    "emotion_tag": "frustration",
                    "strength": 0.8,
                    "reason": "fact marked as contradiction"
                }
            
            # Check for novelty (recent, low access count)
            if fact.access_count <= 1 and fact.get_age_days() < 0.1:  # Less than 2.4 hours old
                return {
                    "event_type": "novelty", 
                    "valence": 0.3,
                    "arousal": 0.6,
                    "emotion_tag": "curiosity",
                    "strength": 0.6,
                    "reason": "new fact with low access"
                }
            
            # Check for high confidence (satisfaction)
            if fact.confidence > 0.9:
                return {
                    "event_type": "confirmation",
                    "valence": 0.4,
                    "arousal": 0.2,
                    "emotion_tag": "contentment",
                    "strength": 0.5,
                    "reason": "high confidence fact"
                }
            
            # Check for low confidence (uncertainty)
            if fact.confidence < 0.3:
                return {
                    "event_type": "confusion",
                    "valence": -0.3,
                    "arousal": 0.5,
                    "emotion_tag": "anxiety",
                    "strength": 0.4,
                    "reason": "low confidence fact"
                }
            
            # Check for words that might indicate emotional content
            emotional_keywords = {
                "love": {"valence": 0.8, "arousal": 0.6, "tag": "love"},
                "hate": {"valence": -0.8, "arousal": 0.7, "tag": "anger"},
                "fear": {"valence": -0.6, "arousal": 0.8, "tag": "fear"},
                "happy": {"valence": 0.7, "arousal": 0.5, "tag": "happiness"},
                "sad": {"valence": -0.7, "arousal": 0.4, "tag": "sadness"},
                "excited": {"valence": 0.6, "arousal": 0.9, "tag": "excitement"},
                "calm": {"valence": 0.3, "arousal": 0.1, "tag": "calm"},
                "angry": {"valence": -0.7, "arousal": 0.8, "tag": "anger"},
                "worried": {"valence": -0.4, "arousal": 0.6, "tag": "anxiety"},
                "confused": {"valence": -0.2, "arousal": 0.5, "tag": "confusion"}
            }
            
            for keyword, emotion_data in emotional_keywords.items():
                if keyword in fact_text:
                    return {
                        "event_type": "content_emotion",
                        "valence": emotion_data["valence"],
                        "arousal": emotion_data["arousal"],
                        "emotion_tag": emotion_data["tag"],
                        "strength": 0.7,
                        "reason": f"keyword '{keyword}' detected in fact"
                    }
            
            # Default neutral response
            return {
                "event_type": "neutral",
                "valence": 0.0,
                "arousal": 0.3,
                "emotion_tag": "neutral",
                "strength": 0.1,
                "reason": "no specific emotional triggers detected"
            }
            
        except Exception as e:
            logging.error(f"Error detecting emotional triggers: {e}")
            return {
                "event_type": "error",
                "valence": 0.0,
                "arousal": 0.3,
                "emotion_tag": "neutral",
                "strength": 0.0,
                "reason": f"error in detection: {str(e)}"
            }
    
    def get_facts(self, user_profile_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  limit: Optional[int] = None) -> List[EnhancedTripletFact]:
        """Retrieve facts from database, ensuring table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                contradiction BOOLEAN DEFAULT FALSE,
                volatile BOOLEAN DEFAULT FALSE,
                volatility_score REAL DEFAULT 0.0,
                user_profile_id TEXT,
                session_id TEXT,
                source_message_id INTEGER,
                hedge_detected BOOLEAN DEFAULT FALSE,
                intensifier_detected BOOLEAN DEFAULT FALSE,
                negation BOOLEAN DEFAULT FALSE,
                embedding TEXT,
                change_history TEXT,
                causal_strength REAL DEFAULT 0.0,
                cause TEXT,
                active BOOLEAN DEFAULT TRUE,
                token_count INTEGER,
                token_entropy REAL,
                token_ids TEXT,
                token_hash TEXT,
                causal_token_propagation TEXT,
                UNIQUE(subject, predicate, object, user_profile_id, session_id)
            )
        """)
        
        # Add causal_strength column if it doesn't exist (migration support)
        try:
            cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN causal_strength REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        query = "SELECT * FROM enhanced_facts WHERE active = TRUE"
        params = []
        if user_profile_id:
            query += " AND user_profile_id = ?"
            params.append(user_profile_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY confidence DESC, timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        facts = []
        for row in rows:
            fact_dict = {
                "id": row[0],
                "subject": row[1],
                "predicate": row[2],
                "object": row[3],
                "timestamp": row[4],
                "confidence": row[5],
                "contradiction": bool(row[6]),
                "volatile": bool(row[7]),
                "volatility_score": row[8],
                "user_profile_id": row[9],
                "session_id": row[10],
                "source_message_id": row[11],
                "hedge_detected": bool(row[12]),
                "intensifier_detected": bool(row[13]),
                "negation": bool(row[14]),
                "embedding": eval(row[15]) if row[15] else None,
                "change_history": eval(row[16]) if row[16] else [],
                "causal_strength": row[17] if row[17] is not None else 0.0,
                "cause": row[18] if row[18] is not None else None,
                # Token-related fields
                "token_count": row[20] if len(row) > 20 and row[20] is not None else None,
                "token_entropy": row[21] if len(row) > 21 and row[21] is not None else None,
                "token_ids": json.loads(row[22]) if len(row) > 22 and row[22] else [],
                "token_hash": row[23] if len(row) > 23 and row[23] is not None else None,
                "causal_token_propagation": json.loads(row[24]) if len(row) > 24 and row[24] else {}
            }
            facts.append(EnhancedTripletFact.from_dict(fact_dict))
        return facts
    
    def _get_contradictions(self, user_profile_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get contradiction records with fact details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                cr.id, cr.resolved, cr.resolution_notes, cr.confidence,
                f1.subject || ' ' || f1.predicate || ' ' || f1.object as fact_a_text,
                f2.subject || ' ' || f2.predicate || ' ' || f2.object as fact_b_text
            FROM contradiction_records cr
            JOIN enhanced_facts f1 ON cr.fact_a_id = f1.id
            JOIN enhanced_facts f2 ON cr.fact_b_id = f2.id
            WHERE f1.user_profile_id = ? AND f1.session_id = ?
        """, (user_profile_id, session_id))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "resolved": bool(row[1]),
                "resolution_notes": row[2],
                "confidence": row[3],
                "fact_a_text": row[4],
                "fact_b_text": row[5]
            })
        
        conn.close()
        return results 
    
    def _handle_token_graph_command(self, cmd: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /token_graph command with optional token ID."""
        try:
            from storage.token_graph import get_token_graph
            
            # Parse command: /token_graph [token_id]
            parts = cmd.split()
            if len(parts) == 1:
                # Show graph summary
                token_graph = get_token_graph()
                summary = token_graph.get_graph_summary()
                
                lines = ["🔗 Token Graph Summary:"]
                lines.append(f"Total tokens: {summary['total_tokens']}")
                lines.append(f"Total facts: {summary['total_facts']}")
                lines.append(f"Total clusters: {summary['total_clusters']}")
                lines.append(f"Total drift events: {summary['total_drift_events']}")
                
                if summary['top_influencers']:
                    lines.append("\n🏆 Top Influencer Tokens:")
                    for token_id, score in summary['top_influencers']:
                        lines.append(f"  - Token {token_id}: {score:.3f}")
                
                if summary['recent_drift_events']:
                    lines.append("\n🔄 Recent Drift Events:")
                    for event in summary['recent_drift_events']:
                        lines.append(f"  - Token {event['token_id']}: {event['old_cluster']} → {event['new_cluster']} (score: {event['drift_score']:.3f})")
                
                return {"response": "\n".join(lines)}
            
            elif len(parts) == 2:
                # Show specific token details
                try:
                    token_id = int(parts[1])
                    token_graph = get_token_graph()
                    
                    # Get token influencers
                    influencers = token_graph.get_token_influencers(token_id)
                    
                    # Get token cluster
                    cluster = token_graph.get_token_cluster(token_id)
                    
                    # Get drift events for this token
                    drift_events = token_graph.get_token_drift_events(token_id, limit=5)
                    
                    lines = [f"🔍 Token {token_id} Analysis:"]
                    lines.append(f"Influenced facts: {len(influencers)}")
                    
                    if cluster:
                        lines.append(f"Semantic cluster: {cluster.cluster_id}")
                        lines.append(f"Cluster size: {len(cluster.token_ids)} tokens")
                        lines.append(f"Cluster coherence: {cluster.coherence_score:.3f}")
                    
                    if drift_events:
                        lines.append(f"\n🔄 Drift Events:")
                        for event in drift_events:
                            lines.append(f"  - {event.old_semantic_cluster} → {event.new_semantic_cluster} (score: {event.drift_score:.3f})")
                    
                    if influencers:
                        lines.append(f"\n📋 Influenced Facts:")
                        # Get fact details
                        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                        fact_map = {fact.id: fact for fact in facts}
                        
                        for fact_id in list(influencers)[:10]:  # Show first 10
                            if fact_id in fact_map:
                                fact = fact_map[fact_id]
                                lines.append(f"  - ({fact.subject}, {fact.predicate}, {fact.object})")
                    
                    return {"response": "\n".join(lines)}
                    
                except ValueError:
                    return {"response": f"Invalid token ID: {parts[1]}. Use a number."}
            
            else:
                return {"response": "Usage: /token_graph [token_id]"}
                
        except ImportError:
            return {"response": "Token graph system not available"}
        except Exception as e:
            return {"response": f"Error analyzing token graph: {e}"}
    
    def _handle_token_clusters_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /token_clusters command."""
        try:
            from storage.token_graph import get_token_graph
            
            token_graph = get_token_graph()
            clusters = token_graph.detect_causal_clusters()
            
            lines = ["🔗 Token Clusters Analysis:"]
            lines.append(f"Detected clusters: {len(clusters)}")
            
            for i, cluster in enumerate(clusters[:5]):  # Show top 5 clusters
                lines.append(f"\n📊 Cluster {i+1} ({cluster.cluster_id}):")
                lines.append(f"  Tokens: {len(cluster.token_ids)}")
                lines.append(f"  Facts: {len(cluster.fact_ids)}")
                lines.append(f"  Coherence: {cluster.coherence_score:.3f}")
                
                # Show some token IDs
                token_list = list(cluster.token_ids)[:5]
                lines.append(f"  Sample tokens: {token_list}")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Token graph system not available"}
        except Exception as e:
            return {"response": f"Error analyzing token clusters: {e}"}
    
    def _handle_token_drift_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /token_drift command."""
        try:
            from storage.token_graph import get_token_graph
            
            token_graph = get_token_graph()
            drift_events = token_graph.get_token_drift_events(limit=10)
            
            lines = ["🔄 Token Drift Analysis:"]
            lines.append(f"Recent drift events: {len(drift_events)}")
            
            if drift_events:
                for i, event in enumerate(drift_events, 1):
                    lines.append(f"\n{i}. Token {event.token_id}:")
                    lines.append(f"   {event.old_semantic_cluster} → {event.new_semantic_cluster}")
                    lines.append(f"   Drift score: {event.drift_score:.3f}")
                    lines.append(f"   Type: {event.drift_type}")
                    lines.append(f"   Affected facts: {len(event.affected_facts)}")
            else:
                lines.append("\nNo drift events detected recently.")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Token graph system not available"}
        except Exception as e:
            return {"response": f"Error analyzing token drift: {e}"}
    
    def _handle_token_influence_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /token_influence command."""
        try:
            from storage.token_graph import get_token_graph
            
            token_graph = get_token_graph()
            
            # Get influence heatmap
            heatmap = token_graph.build_influence_heatmap()
            
            # Get top influencers
            top_influencers = token_graph.get_top_influencers(10)
            
            lines = ["🏆 Token Influence Analysis:"]
            lines.append(f"Total tokens analyzed: {len(heatmap)}")
            
            if top_influencers:
                lines.append("\n🔥 Top Influencer Tokens:")
                for i, (token_id, score) in enumerate(top_influencers, 1):
                    lines.append(f"{i}. Token {token_id}: {score:.3f}")
                    
                    # Get some facts influenced by this token
                    influencers = token_graph.get_token_influencers(token_id)
                    if influencers:
                        lines.append(f"   Influences {len(influencers)} facts")
            
            # Show influence distribution
            if heatmap:
                scores = list(heatmap.values())
                avg_influence = sum(scores) / len(scores)
                max_influence = max(scores)
                min_influence = min(scores)
                
                lines.append(f"\n📊 Influence Statistics:")
                lines.append(f"Average influence: {avg_influence:.3f}")
                lines.append(f"Maximum influence: {max_influence:.3f}")
                lines.append(f"Minimum influence: {min_influence:.3f}")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Token graph system not available"}
        except Exception as e:
            return {"response": f"Error analyzing token influence: {e}"} 
    
    def _handle_drift_analysis_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /drift_analysis command."""
        try:
            from agents.drift_analysis_agent import get_drift_agent
            
            drift_agent = get_drift_agent()
            report = drift_agent.generate_drift_report()
            
            return {"response": report}
            
        except ImportError:
            return {"response": "Drift analysis agent not available"}
        except Exception as e:
            return {"response": f"Error in drift analysis: {e}"}
    
    def _handle_drift_goals_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /drift_goals command."""
        try:
            from agents.cognitive_repair_agent import get_cognitive_repair_agent
            
            repair_agent = get_cognitive_repair_agent()
            
            # Get active goals
            active_goals = repair_agent.get_active_goals()
            completed_goals = repair_agent.get_completed_goals()
            
            lines = ["🔧 Drift-Triggered Repair Goals"]
            lines.append("=" * 40)
            lines.append(f"Active goals: {len(active_goals)}")
            lines.append(f"Completed goals: {len(completed_goals)}")
            
            if active_goals:
                lines.append("\n🎯 Active Goals (by priority):")
                sorted_goals = sorted(active_goals, key=lambda g: g.priority, reverse=True)
                
                for i, goal in enumerate(sorted_goals[:10], 1):  # Show top 10
                    priority_icon = "🔥" if goal.priority > 0.7 else "⚠️" if goal.priority > 0.4 else "💭"
                    lines.append(f"{i}. {priority_icon} {goal.goal}")
                    lines.append(f"   Priority: {goal.priority:.3f}")
                    lines.append(f"   Strategy: {goal.repair_strategy}")
                    lines.append(f"   Affected facts: {len(goal.affected_facts)}")
                    if goal.token_id:
                        lines.append(f"   Token ID: {goal.token_id}")
                    if goal.cluster_id:
                        lines.append(f"   Cluster ID: {goal.cluster_id}")
                    lines.append("")
            else:
                lines.append("\n✅ No active repair goals at this time.")
            
            # Show recent completions
            if completed_goals:
                recent_completions = [g for g in completed_goals if g.status == "completed"][-3:]
                if recent_completions:
                    lines.append("\n✅ Recent Completions:")
                    for goal in recent_completions:
                        lines.append(f"  • {goal.goal}")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Cognitive repair agent not available"}
        except Exception as e:
            return {"response": f"Error retrieving drift goals: {e}"} 

    def _handle_reflex_log_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /reflex_log command."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get recent cycles
            recent_cycles = reflex_logger.get_recent_cycles(limit=10)
            
            # Get statistics
            stats = reflex_logger.get_execution_statistics()
            
            lines = ["🧠 Reflex Log - Recent Cycles"]
            lines.append("=" * 50)
            
            # Show statistics
            lines.append(f"Total cycles: {stats.get('total_cycles', 0)}")
            lines.append(f"Success rate: {stats.get('success_rate', 0.0):.1%}")
            lines.append(f"Average duration: {stats.get('average_duration', 0.0):.2f}s")
            
            # Show strategy breakdown
            if stats.get('strategy_breakdown'):
                lines.append("\n📊 Strategy Breakdown:")
                for strategy, count in stats['strategy_breakdown'].items():
                    success_rate = stats['strategy_success'].get(strategy, {}).get('success_rate', 0.0)
                    lines.append(f"  {strategy}: {count} cycles ({success_rate:.1%} success)")
            
            # Show recent cycles
            if recent_cycles:
                lines.append("\n🔄 Recent Reflex Cycles:")
                for i, cycle in enumerate(recent_cycles, 1):
                    lines.append(f"\n{i}. {reflex_logger.format_cycle_display(cycle)}")
            else:
                lines.append("\nNo reflex cycles found.")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Reflex log system not available"}
        except Exception as e:
            return {"response": f"Error retrieving reflex log: {e}"} 

    def _handle_reflex_scores_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /reflex_scores command."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get score statistics
            stats = reflex_logger.get_score_statistics()
            
            # Get recent scores
            recent_scores = reflex_logger.get_reflex_scores(limit=10)
            
            lines = ["🧠 Reflex Effectiveness Scores"]
            lines.append("=" * 50)
            
            # Show overall statistics
            lines.append(f"📊 Overall Statistics:")
            lines.append(f"  Total scores: {stats.get('total_scores', 0)}")
            lines.append(f"  Average score: {stats.get('average_score', 0.0):.3f}")
            lines.append(f"  Best score: {stats.get('max_score', 0.0):.3f}")
            lines.append(f"  Worst score: {stats.get('min_score', 0.0):.3f}")
            lines.append(f"  Rolling average (last 10): {stats.get('rolling_average', 0.0):.3f}")
            
            # Show strategy breakdown
            if stats.get('strategy_statistics'):
                lines.append(f"\n🔧 Strategy Performance:")
                for strategy, strategy_stats in stats['strategy_statistics'].items():
                    lines.append(f"  {strategy}:")
                    lines.append(f"    Count: {strategy_stats['total']}")
                    lines.append(f"    Avg Score: {strategy_stats['avg_score']:.3f}")
                    lines.append(f"    Best: {strategy_stats['max_score']:.3f}")
                    lines.append(f"    Worst: {strategy_stats['min_score']:.3f}")
            
            # Show recent scores
            if recent_scores:
                lines.append(f"\n🔄 Recent Scores:")
                for i, score in enumerate(recent_scores, 1):
                    lines.append(f"  {i}. {score.score_icon} {score.strategy}: {score.score:.3f}")
                    lines.append(f"     Token: {score.token_id}, Cycle: {score.cycle_id}")
                    if score.scoring_notes:
                        lines.append(f"     Notes: {score.scoring_notes}")
            else:
                lines.append(f"\nNo reflex scores found yet.")
                lines.append("Scores are generated when reflex cycles complete with cognitive state data.")
            
            return {"response": "\n".join(lines)}
            
        except ImportError:
            return {"response": "Reflex log system not available"}
        except Exception as e:
            return {"response": f"Error retrieving reflex scores: {e}"} 

    def _handle_strategy_optimization_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /strategy_optimization command."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get recent scores to analyze optimization patterns
            recent_scores = reflex_logger.get_reflex_scores(limit=50)
            
            lines = ["🧠 Strategy Optimization Analysis"]
            lines.append("=" * 50)
            
            if recent_scores:
                # Analyze strategy performance
                strategy_performance = {}
                for score in recent_scores:
                    strategy = score.strategy
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []
                    strategy_performance[strategy].append(score.score)
                
                # Calculate averages and find best strategies
                strategy_averages = {}
                for strategy, scores in strategy_performance.items():
                    avg_score = sum(scores) / len(scores)
                    strategy_averages[strategy] = avg_score
                
                # Sort strategies by performance
                sorted_strategies = sorted(strategy_averages.items(), key=lambda x: x[1], reverse=True)
                
                lines.append(f"📊 Strategy Performance Analysis ({len(recent_scores)} recent repairs):")
                lines.append("")
                
                for i, (strategy, avg_score) in enumerate(sorted_strategies, 1):
                    count = len(strategy_performance[strategy])
                    icon = "🟢" if avg_score >= 0.8 else "🟡" if avg_score >= 0.6 else "🟠" if avg_score >= 0.4 else "🔴"
                    lines.append(f"  {i}. {icon} {strategy}: {avg_score:.3f} ({count} repairs)")
                
                # Show optimization recommendations
                lines.append("")
                lines.append("🔧 Optimization Recommendations:")
                
                if len(sorted_strategies) >= 2:
                    best_strategy, best_score = sorted_strategies[0]
                    second_strategy, second_score = sorted_strategies[1]
                    
                    lines.append(f"  • Best performing: {best_strategy} (avg: {best_score:.3f})")
                    lines.append(f"  • Consider prioritizing {best_strategy} for similar drift patterns")
                    
                    if best_score - second_score > 0.1:
                        lines.append(f"  • {best_strategy} significantly outperforms {second_strategy}")
                    else:
                        lines.append(f"  • {best_strategy} and {second_strategy} perform similarly")
                
                # Show drift type analysis if available
                lines.append("")
                lines.append("🎯 Drift Type Strategy Preferences:")
                lines.append("  • Contradictions: belief_clarification, fact_consolidation")
                lines.append("  • Volatility: cluster_reassessment, belief_clarification")
                lines.append("  • Semantic Decay: cluster_reassessment, fact_consolidation")
                
            else:
                lines.append("No reflex scores available yet.")
                lines.append("Strategy optimization requires historical performance data.")
                lines.append("Run some drift repairs to build optimization data.")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error in strategy optimization analysis: {e}"} 

    def _handle_anticipated_drifts_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /anticipated_drifts command."""
        try:
            from agents.daemon_drift_watcher import get_daemon_drift_watcher
            
            drift_watcher = get_daemon_drift_watcher()
            
            # Get predicted tokens
            predicted_tokens = drift_watcher.get_predicted_tokens()
            
            lines = ["🔮 Anticipated Drifts - Tokens Flagged as Unstable"]
            lines.append("=" * 60)
            
            if predicted_tokens:
                lines.append(f"Currently monitoring {len(predicted_tokens)} tokens for drift:")
                lines.append("")
                
                for i, token in enumerate(predicted_tokens, 1):
                    confidence_icon = "🔴" if token['confidence'] > 0.8 else "🟡" if token['confidence'] > 0.6 else "🟢"
                    lines.append(f"{i}. {confidence_icon} Token {token['token_id']}")
                    lines.append(f"   Prediction Type: {token['prediction_type']}")
                    lines.append(f"   Confidence: {token['confidence']:.3f}")
                    lines.append(f"   Goal ID: {token['goal_id']}")
                    lines.append(f"   Status: {token['status']}")
                    lines.append(f"   Created: {datetime.fromtimestamp(token['created_time']).strftime('%H:%M')}")
                    lines.append("")
            else:
                lines.append("No tokens currently flagged as unstable.")
                lines.append("The drift watcher monitors for early signs of cognitive decay.")
            
            # Get drift watcher status
            status = drift_watcher.get_monitoring_status()
            lines.append("📊 Drift Watcher Status:")
            lines.append(f"  Running: {'✅' if status['running'] else '❌'}")
            lines.append(f"  Last Check: {datetime.fromtimestamp(status['last_check_time']).strftime('%H:%M')}")
            lines.append(f"  Predictions Made: {status['prediction_count']}")
            lines.append(f"  Goals Spawned: {status['goal_spawn_count']}")
            lines.append(f"  Successful Predictions: {status['successful_predictions']}")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving anticipated drifts: {e}"}

    def _handle_strategy_stats_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /strategy_stats command."""
        try:
            from agents.strategy_optimizer import get_strategy_optimizer
            
            optimizer = get_strategy_optimizer()
            
            # Get strategy performance
            performance = optimizer.analyze_strategy_performance()
            
            lines = ["🧠 Strategy Statistics - Performance Analysis"]
            lines.append("=" * 60)
            
            if performance:
                lines.append(f"Analyzed {len(performance)} strategies:")
                lines.append("")
                
                # Sort by average score
                sorted_strategies = sorted(
                    performance.values(),
                    key=lambda x: x.average_score,
                    reverse=True
                )
                
                for i, strategy in enumerate(sorted_strategies, 1):
                    score_icon = "🟢" if strategy.average_score > 0.8 else "🟡" if strategy.average_score > 0.6 else "🟠" if strategy.average_score > 0.4 else "🔴"
                    lines.append(f"{i}. {score_icon} {strategy.strategy}")
                    lines.append(f"   Total Executions: {strategy.total_executions}")
                    lines.append(f"   Success Rate: {strategy.success_rate:.1%}")
                    lines.append(f"   Average Score: {strategy.average_score:.3f}")
                    lines.append(f"   Best Score: {strategy.best_score:.3f}")
                    lines.append(f"   Worst Score: {strategy.worst_score:.3f}")
                    lines.append(f"   Recent Trend: {strategy.recent_trend:.3f}")
                    lines.append("")
            else:
                lines.append("No strategy performance data available.")
                lines.append("Run some drift repairs to build performance data.")
            
            # Get recommendations
            recommendations = optimizer.get_strategy_recommendations()
            if recommendations.get('suggestions'):
                lines.append("💡 Recommendations:")
                for suggestion in recommendations['suggestions']:
                    lines.append(f"  • {suggestion}")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving strategy statistics: {e}"}

    def _handle_meta_stats_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /meta_stats command."""
        try:
            from agents.meta_monitor import get_meta_monitor
            
            meta_monitor = get_meta_monitor()
            
            # Get meta statistics
            stats = meta_monitor.get_meta_statistics()
            
            lines = ["🧠 Meta-Cognitive Statistics - System Performance"]
            lines.append("=" * 70)
            
            # Monitoring status
            monitoring_status = stats.get('monitoring_status', {})
            lines.append("📊 Monitoring Status:")
            lines.append(f"  Running: {'✅' if monitoring_status.get('running') else '❌'}")
            lines.append(f"  Last Monitoring: {datetime.fromtimestamp(monitoring_status.get('last_monitoring_time', 0)).strftime('%H:%M')}")
            lines.append(f"  Monitoring Cycles: {monitoring_status.get('monitoring_count', 0)}")
            lines.append(f"  Goals Generated: {monitoring_status.get('goals_generated', 0)}")
            lines.append("")
            
            # Current metrics
            current_metrics = stats.get('current_metrics', {})
            if current_metrics:
                lines.append("📈 Current Metrics:")
                for metric_name, metric_data in current_metrics.items():
                    trend_icon = "📈" if metric_data['trend'] == 'increasing' else "📉" if metric_data['trend'] == 'decreasing' else "➡️"
                    lines.append(f"  {trend_icon} {metric_name}: {metric_data['value']:.3f}")
                    lines.append(f"     {metric_data['description']}")
                lines.append("")
            
            # Improvement goals
            active_goals = stats.get('active_improvement_goals', 0)
            total_goals = stats.get('total_goals', 0)
            lines.append("🎯 Improvement Goals:")
            lines.append(f"  Active Goals: {active_goals}")
            lines.append(f"  Total Goals: {total_goals}")
            lines.append(f"  Metrics History: {stats.get('metrics_history_count', 0)} entries")
            
            # Get improvement goals
            goals = meta_monitor.get_improvement_goals()
            if goals:
                lines.append("\n📋 Recent Improvement Goals:")
                for goal in goals[:5]:  # Show last 5 goals
                    priority_icon = "🔴" if goal['priority'] > 0.8 else "🟡" if goal['priority'] > 0.6 else "🟢"
                    lines.append(f"  {priority_icon} {goal['goal_type']}: {goal['description']}")
                    lines.append(f"     Priority: {goal['priority']:.2f}, Status: {goal['status']}")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving meta statistics: {e}"}

    def _handle_beliefs_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /beliefs command."""
        try:
            from storage.enhanced_memory import BeliefAbstractionLayer
            
            belief_system = BeliefAbstractionLayer(self.db_path)
            beliefs = belief_system.get_all_beliefs(limit=15)
            stats = belief_system.get_belief_statistics()
            
            lines = ["🧠 Abstract Beliefs - Higher-Level Reasoning"]
            lines.append("=" * 60)
            lines.append(f"Total beliefs: {stats['total_beliefs']}")
            lines.append(f"Average confidence: {stats['average_confidence']:.2f}")
            lines.append(f"Average coherence: {stats['average_coherence']:.2f}")
            lines.append("")
            
            if beliefs:
                lines.append("📋 Current Beliefs:")
                for i, belief in enumerate(beliefs, 1):
                    confidence_icon = "🟢" if belief.confidence >= 0.8 else "🟡" if belief.confidence >= 0.6 else "🔴"
                    lines.append(f"{i}. {confidence_icon} {belief.belief_id}")
                    lines.append(f"   Cluster: {belief.cluster_id}")
                    lines.append(f"   Statement: {belief.abstract_statement}")
                    lines.append(f"   Confidence: {belief.confidence:.2f} | Coherence: {belief.coherence_score:.2f}")
                    lines.append(f"   Usage: {belief.usage_count} | Supporting facts: {len(belief.supporting_facts)}")
                    lines.append("")
            else:
                lines.append("No abstract beliefs found yet.")
                lines.append("Beliefs are created automatically from consistent fact clusters.")
                lines.append("Run more interactions to build belief patterns.")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving beliefs: {e}"}

    def _handle_reflex_templates_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /reflex_templates command."""
        try:
            from storage.reflex_compression import ReflexCompressor
            
            compressor = ReflexCompressor()
            templates = compressor.get_all_templates(limit=10)
            stats = compressor.get_template_statistics()
            
            lines = ["🔄 Reflex Templates - Reusable Repair Patterns"]
            lines.append("=" * 60)
            lines.append(f"Total templates: {stats['total_templates']}")
            lines.append(f"Average success rate: {stats['average_success_rate']:.2f}")
            lines.append(f"Average score: {stats['average_score']:.2f}")
            lines.append("")
            
            if templates:
                lines.append("📋 Top Templates:")
                for i, template in enumerate(templates, 1):
                    success_icon = "🟢" if template.success_rate >= 0.8 else "🟡" if template.success_rate >= 0.6 else "🔴"
                    lines.append(f"{i}. {success_icon} {template.template_id}")
                    lines.append(f"   Strategy: {template.strategy}")
                    lines.append(f"   Goal Pattern: {template.goal_pattern}")
                    lines.append(f"   Success Rate: {template.success_rate:.2f} | Avg Score: {template.avg_score:.2f}")
                    lines.append(f"   Usage: {template.usage_count} | Source Cycles: {len(template.source_cycles)}")
                    if template.execution_pattern:
                        lines.append(f"   Execution: {', '.join(template.execution_pattern[:3])}...")
                    lines.append("")
            else:
                lines.append("No reflex templates found yet.")
                lines.append("Templates are created automatically from similar reflex cycles.")
                lines.append("Run more drift repairs to build template patterns.")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving reflex templates: {e}"}

    def _handle_memory_clean_log_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /memory_clean_log command."""
        try:
            from storage.memory_autocleaner import MemoryCleaner
            
            cleaner = MemoryCleaner(self.db_path)
            cleanup_log = cleaner.get_cleanup_log(limit=15)
            stats = cleaner.get_cleanup_statistics()
            memory_stats = cleaner.get_memory_usage_stats()
            
            lines = ["🧹 Memory Cleanup Log - Garbage Collection"]
            lines.append("=" * 60)
            lines.append(f"Total cleanup actions: {stats['total_actions']}")
            lines.append(f"Total memory saved: {stats['total_memory_saved']} bytes")
            lines.append(f"Total confidence impact: {stats['total_confidence_impact']:.2f}")
            lines.append(f"Recent actions (24h): {stats['recent_actions_24h']}")
            lines.append("")
            
            lines.append("📊 Current Memory Usage:")
            lines.append(f"  Total facts: {memory_stats['total_facts']}")
            lines.append(f"  Contradictions: {memory_stats['total_contradictions']}")
            lines.append(f"  Average confidence: {memory_stats['average_confidence']:.2f}")
            lines.append(f"  Average volatility: {memory_stats['average_volatility']:.2f}")
            lines.append("")
            
            if cleanup_log:
                lines.append("📋 Recent Cleanup Actions:")
                for i, action in enumerate(cleanup_log[:10], 1):
                    action_icon = "🗑️" if action.action_type == "remove" else "🗜️" if action.action_type == "compress" else "⚡"
                    lines.append(f"{i}. {action_icon} {action.action_type.upper()} {action.target_type}: {action.target_id}")
                    lines.append(f"   Reason: {action.reason}")
                    lines.append(f"   Memory saved: {action.memory_saved} bytes | Impact: {action.confidence_impact:.2f}")
                    lines.append("")
            else:
                lines.append("No cleanup actions logged yet.")
                lines.append("Memory cleaner runs automatically in the background.")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving memory cleanup log: {e}"}

    def _handle_belief_trace_command(self, cmd: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /belief_trace command."""
        try:
            from storage.enhanced_reasoning import EnhancedReasoningEngine
            
            reasoning_engine = EnhancedReasoningEngine(self.db_path)
            parts = cmd.split()
            
            if len(parts) > 1:
                # Show trace for specific goal or token
                target = parts[1]
                if target.isdigit():
                    # Token ID
                    token_id = int(target)
                    traces = reasoning_engine.get_belief_traces_by_token(token_id, limit=5)
                    lines = [f"🧠 Belief Traces for Token {token_id}"]
                else:
                    # Goal ID
                    trace = reasoning_engine.get_belief_trace(target)
                    traces = [trace] if trace else []
                    lines = [f"🧠 Belief Trace for Goal {target}"]
            else:
                # Show recent traces
                traces = reasoning_engine.get_recent_belief_traces(limit=10)
                lines = ["🧠 Recent Belief Traces - Decision Reasoning"]
            
            lines.append("=" * 60)
            
            if traces:
                for i, trace in enumerate(traces, 1):
                    confidence_icon = "🟢" if trace.strategy_confidence >= 0.8 else "🟡" if trace.strategy_confidence >= 0.6 else "🔴"
                    lines.append(f"{i}. {confidence_icon} {trace.trace_id}")
                    lines.append(f"   Goal: {trace.goal_id}")
                    lines.append(f"   Strategy: {trace.final_strategy} (confidence: {trace.strategy_confidence:.2f})")
                    lines.append(f"   Beliefs considered: {len(trace.considered_beliefs)}")
                    if trace.belief_influences:
                        top_influences = sorted(trace.belief_influences.items(), key=lambda x: x[1], reverse=True)[:3]
                        lines.append(f"   Top influences: {', '.join([f'{bid}({inf:.2f})' for bid, inf in top_influences])}")
                    lines.append(f"   Reasoning: {trace.reasoning_notes}")
                    lines.append("")
            else:
                lines.append("No belief traces found.")
                lines.append("Belief traces are created when goals are processed with enhanced reasoning.")
            
            return {"response": "\n".join(lines)}
            
        except Exception as e:
            return {"response": f"Error retrieving belief traces: {e}"}

    def _handle_hypotheses_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /hypotheses command"""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return {"response": "Enhanced components not available"}
        
        try:
            hypothesis_agent = HypothesisGeneratorAgent()
            open_hypotheses = hypothesis_agent.get_open_hypotheses(limit=10)
            stats = hypothesis_agent.get_hypothesis_statistics()
            
            lines = ["🔬 Open Hypotheses - Causal Analysis"]
            lines.append("=" * 60)
            lines.append(f"Total hypotheses: {stats['total_hypotheses']}")
            lines.append(f"Confirmation rate: {stats['confirmation_rate']:.2%}")
            lines.append(f"Recent hypotheses (24h): {stats['recent_hypotheses_24h']}")
            lines.append("")
            
            if open_hypotheses:
                for i, hypothesis in enumerate(open_hypotheses, 1):
                    confidence_icon = "🟢" if hypothesis.confidence_score >= 0.8 else "🟡" if hypothesis.confidence_score >= 0.6 else "🔴"
                    lines.append(f"{i}. {confidence_icon} {hypothesis.hypothesis_id}")
                    lines.append(f"   Cause: {hypothesis.cause_token}")
                    lines.append(f"   Prediction: {hypothesis.predicted_outcome}")
                    lines.append(f"   Probability: {hypothesis.probability:.2f} | Confidence: {hypothesis.confidence_score:.2f}")
                    lines.append(f"   Type: {hypothesis.hypothesis_type}")
                    lines.append(f"   Evidence: {', '.join(hypothesis.supporting_evidence)}")
                    lines.append("")
            else:
                lines.append("No open hypotheses found.")
                lines.append("Hypotheses are generated when contradictions or drift are detected.")
            
            return {"response": "\n".join(lines)}
        except Exception as e:
            return {"response": f"Error retrieving hypotheses: {e}"}
    
    def _handle_confirm_hypothesis_command(self, cmd: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /confirm_hypothesis command"""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return {"response": "Enhanced components not available"}
        
        try:
            parts = cmd.split()
            if len(parts) < 2:
                return {"response": "Usage: /confirm_hypothesis <hypothesis_id>"}
            
            hypothesis_id = parts[1]
            hypothesis_agent = HypothesisGeneratorAgent()
            hypothesis_agent.confirm_hypothesis(hypothesis_id)
            
            return {"response": f"✅ Hypothesis {hypothesis_id} confirmed"}
        except Exception as e:
            return {"response": f"Error confirming hypothesis: {e}"}
    
    def _handle_reject_hypothesis_command(self, cmd: str, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /reject_hypothesis command"""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return {"response": "Enhanced components not available"}
        
        try:
            parts = cmd.split()
            if len(parts) < 2:
                return {"response": "Usage: /reject_hypothesis <hypothesis_id>"}
            
            hypothesis_id = parts[1]
            hypothesis_agent = HypothesisGeneratorAgent()
            hypothesis_agent.reject_hypothesis(hypothesis_id)
            
            return {"response": f"❌ Hypothesis {hypothesis_id} rejected"}
        except Exception as e:
            return {"response": f"Error rejecting hypothesis: {e}"}
    
    def _handle_causal_dashboard_command(self, user_profile_id: str, session_id: str) -> Dict[str, Any]:
        """Handle /causal_dashboard command"""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return {"response": "Enhanced components not available"}
        
        try:
            dashboard = CausalAuditDashboard()
            dashboard_data = dashboard.generate_dashboard_data()
            
            lines = ["🔮 CAUSAL AUDIT DASHBOARD - Predictive Modeling"]
            lines.append("=" * 80)
            lines.append("")
            
            # Current metrics
            metrics = dashboard_data['current_metrics']
            lines.append("📊 CURRENT METRICS:")
            lines.append(f"  Total Predictions: {metrics.total_predictions}")
            lines.append(f"  Pending Predictions: {metrics.pending_predictions}")
            lines.append(f"  Total Hypotheses: {metrics.total_hypotheses}")
            lines.append(f"  Open Hypotheses: {metrics.open_hypotheses}")
            lines.append(f"  Anticipatory Reflexes: {metrics.total_anticipatory_reflexes}")
            lines.append(f"  Successful Reflexes: {metrics.successful_reflexes}")
            lines.append(f"  Prediction Accuracy: {metrics.prediction_accuracy:.2%}")
            lines.append(f"  Hypothesis Confirmation Rate: {metrics.hypothesis_confirmation_rate:.2%}")
            lines.append(f"  Reflex Success Rate: {metrics.reflex_success_rate:.2%}")
            lines.append("")
            
            # Upcoming drifts
            lines.append("🔮 UPCOMING PREDICTED DRIFTS:")
            for drift in dashboard_data['upcoming_drifts'][:5]:
                urgency_icon = "🔴" if drift['urgency'] > 0.8 else "🟡" if drift['urgency'] > 0.5 else "🟢"
                lines.append(f"  {urgency_icon} {drift['prediction_type']}: {drift['predicted_outcome']}")
                lines.append(f"    Probability: {drift['probability']:.2%} | Urgency: {drift['urgency']:.2f}")
            lines.append("")
            
            # System health
            health = dashboard_data['system_statistics']['system_health']
            health_icon = "🟢" if health['status'] == 'excellent' else "🟡" if health['status'] == 'good' else "🔴"
            lines.append(f"{health_icon} SYSTEM HEALTH: {health['status'].upper()} ({health['overall_score']:.2%})")
            
            return {"response": "\n".join(lines)}
        except Exception as e:
            return {"response": f"Error generating dashboard: {e}"} 