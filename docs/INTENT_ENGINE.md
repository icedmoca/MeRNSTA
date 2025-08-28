# MeRNSTA Phase 7: Motivational Drives & Intent Engine

## Overview

Phase 7 transforms MeRNSTA from reactive to purpose-driven cognition by implementing internal motivational drives and autonomous intent formation. This cognitive layer enables MeRNSTA to:

- 🧠 **Monitor internal motivational tension** across different cognitive drives
- 🎯 **Abstract its own goals** from emergent patterns in memory and behavior  
- ⚡ **Select what matters most** to work on based on drive-based arbitration
- 🔄 **Blend symbolic reasoning** with motivational priorities

## Core Components

### 1. Motivational Drive System (`storage/drive_system.py`)

The `MotivationalDriveSystem` implements five core drives that create autonomous motivation:

#### Drive Types

| Drive | Description | Triggers High Values | Strategy Preferences |
|-------|-------------|---------------------|---------------------|
| **Curiosity** | Seeks novel information and exploration | Low access count, recent timestamps | `exploration_goal`, `deep_exploration` |
| **Coherence** | Maintains logical consistency | Low contradiction rates | `belief_clarification`, `fact_consolidation` |
| **Stability** | Preserves established beliefs | Low volatility scores | `fact_consolidation`, `volatility_reduction` |
| **Novelty** | Attracted to new patterns | Recent facts with low confidence | `novelty_investigation`, `deep_exploration` |
| **Conflict** | Resolves contradictions and tensions | High contradiction counts | `conflict_resolution`, `belief_clarification` |

#### Drive Scoring Logic

Drives are scored per token based on fact characteristics:

```python
# Curiosity: Low access + recent = high curiosity
curiosity = max(0.0, 1.0 - (avg_access / 10.0)) * math.exp(-time_since_access / 86400)

# Coherence: High contradiction = low coherence (inverted)
coherence = 1.0 - avg_contradiction

# Stability: Low volatility = high stability (inverted)  
stability = 1.0 - avg_volatility

# Novelty: Recent facts with low confidence = high novelty
novelty = 1.0 - avg_confidence_recent_facts

# Conflict: High contradiction count = high conflict drive
conflict = min(1.0, contradiction_count / max(1, total_facts))
```

#### Drive Tension Calculation

Combined tension determines motivational pressure to act:

```python
tension = (
    drives["conflict"] * 1.2 +
    (1.0 - drives["coherence"]) * 1.0 +
    drives["curiosity"] * 0.8 +
    drives["novelty"] * 0.6 -
    drives["stability"] * 0.4  # Stability reduces tension
)
```

### 2. Autonomous Intent Modeler (`agents/intent_modeler.py`)

The `AutonomousIntentModeler` discovers patterns in memory and behavior to form higher-level goals.

#### Intent Pattern Discovery

Analyzes memory data to discover patterns:

- **Subject Clustering**: Repeated focus on specific topics → exploration intent
- **Strategy Preferences**: Successful strategy patterns → optimization intent  
- **Volatility Response**: High volatility facts → stabilization intent
- **Knowledge Gaps**: Low confidence areas → curiosity intent
- **Token Themes**: Recurring words in token history → thematic focus intent

#### Goal Evolution Process

1. **Pattern Recognition**: Discover intent patterns from memory/behavior
2. **Drive Integration**: Weight patterns by current motivational state
3. **Goal Generation**: Create `EvolvedGoal` objects with drive metadata
4. **Meta-Goal Creation**: Combine multiple patterns into higher abstractions
5. **Priority Adjustment**: Continuously adapt priorities based on drive changes

#### Abstraction Levels

| Level | Description | Examples |
|-------|-------------|----------|
| **Level 1** | Direct fact-based goals | "Clarify belief about X" |
| **Level 2** | Pattern-derived goals | "Explore machine learning domain" |
| **Level 3** | Meta-goals from synthesis | "Comprehensive exploration across domains" |
| **Level 4** | Strategic planning goals | "Optimize learning methodology" |

### 3. Enhanced Drift Execution Engine

Integration with existing drift execution includes:

#### Drive-Weighted Strategy Selection

```python
def select_best_strategy(self, token_id, context=None):
    # Evaluate current drive signals
    drive_signals = self.drive_system.evaluate_token_state(token_id)
    
    # Calculate strategy preferences based on drives
    drive_preferences = self._get_drive_strategy_preferences(drive_signals)
    
    # Weight historical performance by drive alignment
    weighted_scores = {}
    for strategy, historical_score in strategy_averages.items():
        drive_weight = drive_preferences.get(strategy, 1.0)
        weighted_scores[strategy] = historical_score * drive_weight
    
    # Select highest weighted strategy
    return max(weighted_scores.items(), key=lambda x: x[1])
```

#### Autonomous Goal Spawning

- Monitors tokens with drive tension above threshold
- Spawns `MotivationalGoal` objects when tension exceeds limits
- Logs drive-weighted routing decisions for learning

### 4. Self-Aware Model Extensions

The symbolic control layer now tracks internal motives:

```python
self.active_drives = {
    "curiosity": 0.74,
    "stability": 0.52, 
    "coherence": 0.91,
    "novelty": 0.33,
    "conflict": 0.18
}
```

#### Drive-Influenced Decision Recording

```python
def record_drive_influenced_decision(self, decision_type, dominant_drive, outcome):
    # Creates symbolic rules like:
    # "DominantDrive(curiosity) AND Context(strategy_selection) → Decision(exploration_goal)"
```

## CLI Commands

### `/self_drives` - Current Motivational State

Displays current drive signals and trends:

```
🧠 **Current Motivational Drive Signals**
==================================================
🔥 **Curiosity**: 0.847
🟡 **Coherence**: 0.623  
🔵 **Stability**: 0.412
🔵 **Novelty**: 0.289
🔵 **Conflict**: 0.156

📊 **Analysis** (42 samples):
  • curiosity: increasing (volatility: 0.12)
  • coherence: decreasing (volatility: 0.08)
  • stability: stable (volatility: 0.05)
```

### `/intent_model` - Current Intents & Goals

Shows discovered intent patterns and evolved goals:

```
🎯 **Current Intent Model & Meta-Goals**
==================================================
🧠 **Current Motivational State:**
  • Curiosity: Very High (0.85)
  • Coherence: Moderate (0.62)

🔍 **Current Focus:** Actively exploring new information, particularly around 'machine learning'

⚡ **Recent Evolved Goals** (3 goals):
  • Evolved goal: Strong interest in exploring machine learning
    Priority: 0.89, Age: 2.3h
    Driven by: curiosity (0.85)
```

### `/goal_pressure` - Token Drive Rankings

Ranks tokens by motivational urgency:

```
⚡ **Token Drive Pressure Rankings**
==================================================
🚨 **#1 Token 1247**: Pressure 0.923
    Dominant drive: conflict (0.89)
⚠️ **#2 Token 1156**: Pressure 0.756
    Dominant drive: curiosity (0.82)
📊 **#3 Token 1089**: Pressure 0.634
    Dominant drive: coherence (0.71)
```

### `/drive_spawn` - Manual Goal Generation

Forces creation of drive-based goals:

```
🌱 **Manual Drive-Based Goal Generation**
==================================================
✅ Spawned goal for token 1247:
   Goal: Autonomous conflict goal for token 1247
   Strategy: conflict_resolution
   Priority: 0.89
   Driven by: conflict (0.89)

🎯 Successfully spawned 3 autonomous goals.
```

## How Drive Pressure Works

### 1. Continuous Monitoring

The system continuously evaluates each token's drive state:

```python
# For each token, calculate drive strengths
drives = {
    "curiosity": f(access_count, recency),
    "coherence": f(contradiction_rate), 
    "stability": f(volatility_score),
    "novelty": f(confidence, recency),
    "conflict": f(contradiction_count)
}

# Calculate overall pressure
pressure = weighted_sum(drives, strategy_mappings)
```

### 2. Tension Thresholds

- **Low Tension** (< 0.4): Normal cognitive maintenance
- **Medium Tension** (0.4-0.7): Increased attention, scheduled repair
- **High Tension** (> 0.7): Autonomous goal spawning, immediate action

### 3. Goal Prioritization

Goals are prioritized by:
1. **Base Priority**: From pattern confidence or explicit assignment
2. **Drive Alignment**: Bonus for goals that align with current dominant drives  
3. **Temporal Urgency**: Recent high-tension tokens get priority boost
4. **Abstraction Level**: Higher abstraction goals get slight priority bonus

## Intent Abstraction Evolution

### Pattern → Goal Evolution Process

1. **Raw Memory Patterns**
   - Subject clustering: "python mentioned 5 times"
   - Strategy success: "belief_clarification 80% success rate"

2. **Intent Pattern Discovery**
   - `IntentPattern(category="exploration", description="Strong interest in python programming")`
   - `IntentPattern(category="optimization", description="Preference for belief_clarification")`

3. **Goal Evolution**
   - `EvolvedGoal(strategy="deep_exploration", description="Explore python programming domain")`
   - Level 2 abstraction from raw patterns

4. **Meta-Goal Synthesis**
   - Multiple exploration patterns → `Meta-Goal: Comprehensive exploration across technical domains`
   - Level 3 abstraction combining related patterns

5. **Strategic Planning**
   - `Strategic Goal: Optimize learning methodology for technical subjects`
   - Level 4 abstraction for long-term cognitive improvement

### Goal Lineage Tracking

Each goal maintains lineage information:

```python
EvolvedGoal(
    goal_id="meta_exploration_1640995200",
    source_intents=["pattern_python_1640995100", "pattern_ml_1640995150"],
    goal_lineage=["exploration_goal_python", "exploration_goal_ml"],
    abstraction_level=3
)
```

## Autonomous System Behavior

### Proactive vs Reactive Cognition

**Traditional Reactive**:
```
User Input → Fact Extraction → Storage → Response
```

**Phase 7 Proactive**:
```
Drive Monitoring → Pattern Discovery → Intent Formation → Autonomous Goal Generation → Self-Directed Action
```

### Integration Checkpoints

✅ **Drive system called during every reflex cycle**
- Each cognitive repair operation updates drive signals
- Strategy selection considers drive alignment

✅ **Goal planner checks drive pressure before acting**  
- `rank_tokens_by_drive_pressure()` consulted for task prioritization
- High-pressure tokens get immediate attention

✅ **Symbolic self-model logs drive changes**
- Drive-influenced decisions create symbolic rules
- Pattern learning improves future drive-strategy alignment

✅ **Drive-generated goals stored with metadata**
- All autonomous goals tagged with driving motives
- Enables analysis of motivational patterns over time

## Configuration

### Drive Weights (Configurable)

```python
drive_weights = {
    "curiosity": 0.8,      # Weight for curiosity drive
    "coherence": 0.9,      # Weight for coherence maintenance  
    "stability": 0.6,      # Weight for stability preservation
    "novelty": 0.7,        # Weight for novelty seeking
    "conflict": 0.85       # Weight for conflict resolution
}
```

### Thresholds

- `tension_threshold`: 0.7 (spawn goals above this tension)
- `intent_confidence_threshold`: 0.6 (minimum pattern confidence)
- `max_abstraction_levels`: 4 (maximum goal evolution depth)

## Performance Considerations

### Computational Efficiency

- **Drive evaluation**: O(facts_per_token) - linear with token size
- **Pressure ranking**: O(n log n) where n = active tokens
- **Pattern discovery**: O(facts * patterns) - limited by lookback window
- **Goal evolution**: O(patterns²) - quadratic but patterns are small

### Memory Usage

- Drive signals cached with temporal decay
- Intent patterns stored with automatic cleanup (24h retention)
- Goal lineage tracks only essential ancestry information

### Scalability

- Drive evaluation can be parallelized per token
- Pattern discovery uses sliding time windows
- Database queries use proper indexing on timestamp and token_id

## Troubleshooting

### Common Issues

**No drive signals detected**:
- Check if facts have proper token_id assignments
- Verify database connectivity
- Ensure recent activity (drives decay over time)

**Goals not spawning**:
- Lower `tension_threshold` in configuration
- Check drive weights are properly set
- Verify `spawn_goal_if_needed()` isn't rate-limited

**Poor intent patterns**:
- Increase `pattern_discovery_window` for more history
- Lower `intent_confidence_threshold` for more patterns
- Check fact diversity and temporal distribution

### Debug Commands

```bash
# Test drive evaluation for specific token
python -c "from storage.drive_system import get_drive_system; ds=get_drive_system(); print(ds.evaluate_token_state(123))"

# Check drive system health
python -c "from storage.drive_system import get_drive_system; ds=get_drive_system(); print(ds.analyze_drive_trends())"

# Test intent modeling
python -c "from agents.intent_modeler import get_intent_modeler; im=get_intent_modeler(); print(im.summarize_current_motives())"
```

## Future Enhancements

### Planned Features

1. **Drive Learning**: Automatic drive weight adjustment based on success patterns
2. **Social Drives**: Incorporate user satisfaction and engagement signals  
3. **Temporal Planning**: Long-term goal scheduling based on drive predictions
4. **Multi-Agent Drives**: Coordination of drives across multiple MeRNSTA instances

### Research Directions

- **Neuromorphic Drives**: Biologically-inspired drive dynamics
- **Emergent Motivation**: Higher-order drives emerging from lower-level interactions
- **Value Alignment**: Ensuring autonomous drives align with user values
- **Metacognitive Drives**: Drives about drives (second-order motivation)

---

**Phase 7 Achievement**: MeRNSTA now exhibits true cognitive autonomy, shifting from reactive information processing to self-directed, purpose-driven reasoning guided by internal motivational dynamics.