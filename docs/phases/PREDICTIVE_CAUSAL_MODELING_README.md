# Predictive Causal Modeling & Hypothesis Generation

## Overview

MeRNSTA now includes a comprehensive **Predictive Causal Modeling & Hypothesis Generation** system that makes it anticipatory and self-repairing. This system enables MeRNSTA to:

- **Forecast its own instability** before symptoms appear
- **Generate hypotheses** about the causes of drift and contradictions
- **Take preventive action** using anticipatory reflexes
- **Refine predictive models** over time based on outcomes

## 🧠 Core Components

### 1. Causal Drift Predictor (`storage/causal_drift_predictor.py`)

**Purpose**: Maintains time-series data for each token and predicts future drift patterns.

**Key Features**:
- Records metrics for each token: volatility, drift_score, coherence, consistency_rate
- Fits predictive models using linear regression and trend analysis
- Spawns predictive goals when thresholds are exceeded
- Links predictions to reflex cycles for tracking

**Prediction Types**:
- **Volatility Prediction**: Forecasts when token volatility will exceed thresholds
- **Contradiction Likelihood**: Predicts probability of contradictions emerging
- **Coherence Decay**: Forecasts when semantic coherence will deteriorate

**Example**:
```python
from storage.causal_drift_predictor import CausalDriftPredictor

predictor = CausalDriftPredictor()
predictor.record_token_metrics(token_id=42, metrics={
    'volatility': 0.6,
    'drift_score': 0.7,
    'coherence': 0.5,
    'consistency_rate': 0.8
})

# Automatically spawns predictive goals when thresholds exceeded
pending_goals = predictor.get_pending_predictive_goals()
```

### 2. Hypothesis Generator Agent (`agents/hypothesis_generator.py`)

**Purpose**: Generates hypotheses about the causes of drift and contradictions.

**Key Features**:
- Analyzes contradictions to identify root causes
- Generates hypotheses for semantic drift patterns
- Links hypotheses to reflex cycles and drift goals
- Tracks hypothesis confirmation/rejection rates

**Hypothesis Types**:
- **Contradiction Causes**: Subject mismatches, predicate conflicts, object contradictions
- **Drift Causes**: High drift scores, cluster instability, semantic context shifts
- **Semantic Decay**: Low coherence, high volatility, cluster fragmentation

**Example**:
```python
from agents.hypothesis_generator import HypothesisGeneratorAgent

agent = HypothesisGeneratorAgent()

# Generate hypotheses for contradiction
contradiction_data = {
    'fact_a': {'subject': 'Python', 'predicate': 'is', 'object': 'easy'},
    'fact_b': {'subject': 'Python', 'predicate': 'is', 'object': 'difficult'}
}
hypotheses = agent.generate_hypotheses_for_contradiction(contradiction_data)

# Confirm or reject hypotheses
agent.confirm_hypothesis(hypothesis_id)
agent.reject_hypothesis(hypothesis_id)
```

### 3. Reflex Anticipator (`agents/reflex_anticipator.py`)

**Purpose**: Enables proactive repair before drift symptoms appear.

**Key Features**:
- Checks for predictive goals before triggering drift cycles
- Uses reflex templates for immediate execution
- Skips LLM when pattern matches are found
- Tracks anticipatory reflex effectiveness

**Anticipatory Logic**:
1. Check for pending predictive goals relevant to current context
2. Find matching reflex templates based on goal characteristics
3. Execute anticipatory reflex using template or generate new strategy
4. Record outcomes for model refinement

**Example**:
```python
from agents.reflex_anticipator import ReflexAnticipator

anticipator = ReflexAnticipator()

# Check if anticipatory reflex should be triggered
reflex = anticipator.check_for_anticipatory_reflex(
    token_id=42,
    cluster_id="programming_languages",
    drift_score=0.8
)

if reflex:
    print(f"Anticipatory reflex triggered: {reflex.strategy}")
```

### 4. Causal Audit Dashboard (`storage/causal_audit_dashboard.py`)

**Purpose**: Comprehensive monitoring and visualization of predictive causal modeling.

**Key Features**:
- Shows predicted upcoming drifts with urgency levels
- Displays active hypotheses and their statuses
- Tracks anticipatory reflex templates triggered
- Provides timeline of predicted vs actual outcomes
- Calculates overall system health metrics

**Dashboard Sections**:
- **Current Metrics**: Real-time statistics across all systems
- **Upcoming Drifts**: Prioritized list of predicted issues
- **Active Hypotheses**: Open hypotheses with confidence scores
- **System Health**: Overall performance and trend analysis

## 🚀 CLI Commands

### New Commands Added

#### `/hypotheses`
Shows open hypotheses with their causes, predictions, and confidence scores.

```bash
/hypotheses
```

**Output**:
```
🔬 Open Hypotheses - Causal Analysis
============================================================
Total hypotheses: 15
Confirmation rate: 73.33%
Recent hypotheses (24h): 8

1. 🟢 hypothesis_contradiction_cause_1234567890
   Cause: object_contradiction_easy_difficult
   Prediction: This contradiction suggests object_contradiction_easy_difficult is unstable
   Probability: 0.90 | Confidence: 0.85
   Type: contradiction_cause
   Evidence: Contradictory objects: easy vs difficult, High confidence difference: 0.10
```

#### `/confirm_hypothesis <hypothesis_id>`
Confirms a hypothesis as correct.

```bash
/confirm_hypothesis hypothesis_contradiction_cause_1234567890
```

#### `/reject_hypothesis <hypothesis_id>`
Rejects a hypothesis as incorrect.

```bash
/reject_hypothesis hypothesis_contradiction_cause_1234567890
```

#### `/causal_dashboard`
Shows comprehensive causal audit dashboard.

```bash
/causal_dashboard
```

**Output**:
```
🔮 CAUSAL AUDIT DASHBOARD - Predictive Modeling
================================================================================

📊 CURRENT METRICS:
  Total Predictions: 25
  Pending Predictions: 8
  Total Hypotheses: 15
  Open Hypotheses: 12
  Anticipatory Reflexes: 18
  Successful Reflexes: 15
  Prediction Accuracy: 83.33%
  Hypothesis Confirmation Rate: 73.33%
  Reflex Success Rate: 83.33%

🔮 UPCOMING PREDICTED DRIFTS:
  🔴 contradiction_prevention: Contradiction likelihood will reach 0.75
    Probability: 85.00% | Urgency: 0.92
  🟡 coherence_improvement: Coherence will decay to 0.25
    Probability: 72.00% | Urgency: 0.68

🟢 SYSTEM HEALTH: GOOD (80.00%)
```

## 🔧 Integration with Existing Systems

### Enhanced Memory System Integration

The predictive causal modeling system is fully integrated with the `EnhancedMemorySystem`:

```python
from storage.enhanced_memory_system import EnhancedMemorySystem

# Initialize enhanced system (automatically includes predictive components)
memory_system = EnhancedMemorySystem()

# Process input normally - predictive features work in background
result = memory_system.process_input("I love Python programming", "user1", "session1")

# Use new predictive commands
hypotheses = memory_system._handle_command("/hypotheses", "user1", "session1")
dashboard = memory_system._handle_command("/causal_dashboard", "user1", "session1")
```

### Automatic Integration Points

1. **Token Metrics Recording**: Automatically records metrics when facts are processed
2. **Contradiction Detection**: Triggers hypothesis generation when contradictions are found
3. **Drift Detection**: Spawns predictive goals when semantic drift is detected
4. **Reflex Cycles**: Links predictions to reflex cycles for outcome tracking

## 📊 Real-Time Learning

### Model Refinement

The system continuously learns and improves:

1. **Prediction Accuracy**: Tracks how well predictions match actual outcomes
2. **Hypothesis Validation**: Confirms or rejects hypotheses based on evidence
3. **Reflex Effectiveness**: Measures success rates of anticipatory reflexes
4. **Template Evolution**: Refines reflex templates based on performance

### Learning Metrics

- **Prediction Accuracy**: Percentage of predictions that came true
- **Hypothesis Confirmation Rate**: Percentage of hypotheses confirmed vs rejected
- **Reflex Success Rate**: Percentage of anticipatory reflexes that succeeded
- **System Health Score**: Overall performance across all components

## 🧪 Testing

### Run Comprehensive Tests

```bash
python test_predictive_causal_modeling.py
```

This test script verifies:
- ✅ Causal Drift Predictor functionality
- ✅ Hypothesis Generator Agent
- ✅ Reflex Anticipator
- ✅ Causal Audit Dashboard
- ✅ Integration with Enhanced Memory System

### Test Output Example

```
🚀 STARTING PREDICTIVE CAUSAL MODELING TESTS
================================================================================

🧠 TESTING CAUSAL DRIFT PREDICTOR
============================================================
📊 Recording token metrics...
🎯 Checking for predictive goals...
Found 3 pending predictive goals
  Goal: predictive_volatility_1_1234567890
    Type: volatility_reduction
    Outcome: Volatility will increase to 0.85
    Probability: 0.75
    Urgency: 0.82

📈 Prediction Statistics:
  Total predictions: 15
  Pending predictions: 3
  Average accuracy: 0.83

✅ PASSED

📋 TEST SUMMARY
================================================================================
causal_drift_predictor    ✅ PASSED
hypothesis_generator      ✅ PASSED
reflex_anticipator        ✅ PASSED
causal_audit_dashboard    ✅ PASSED
integration               ✅ PASSED

Overall: 5/5 tests passed
🎉 All tests passed! Predictive causal modeling system is working correctly.
```

## 🎯 Use Cases

### 1. Proactive Contradiction Prevention

**Scenario**: System predicts that a contradiction is likely to emerge in the "programming_languages" cluster.

**System Response**:
1. Causal Drift Predictor spawns predictive goal
2. Reflex Anticipator finds matching template
3. Executes anticipatory reflex to clarify beliefs
4. Prevents contradiction before it occurs

### 2. Semantic Drift Anticipation

**Scenario**: System detects increasing volatility in a concept cluster.

**System Response**:
1. Analyzes drift patterns and generates hypotheses
2. Predicts coherence decay in next 5 tasks
3. Triggers cluster reassessment reflex
4. Stabilizes concept before fragmentation

### 3. Adaptive Strategy Learning

**Scenario**: System learns which strategies work best for different drift patterns.

**System Response**:
1. Tracks reflex success rates by pattern type
2. Refines reflex templates based on outcomes
3. Improves prediction accuracy over time
4. Optimizes anticipatory responses

## 🔮 Future Enhancements

### Planned Features

1. **Advanced ML Models**: Replace simple linear regression with more sophisticated predictive models
2. **Causal Graph Analysis**: Build causal graphs to understand deeper relationships
3. **Multi-Step Prediction**: Predict cascading effects across multiple tokens
4. **Confidence Calibration**: Improve prediction confidence estimates
5. **External Data Integration**: Incorporate external context for better predictions

### Research Directions

- **Causal Discovery**: Automatically discover causal relationships in the knowledge base
- **Temporal Modeling**: Better modeling of how concepts evolve over time
- **Uncertainty Quantification**: More sophisticated uncertainty estimates
- **Explainable AI**: Generate explanations for predictions and hypotheses

## 🏗️ Architecture

### Database Schema

The system uses `causal_predictions.db` with the following tables:

- **time_series_data**: Token metrics over time
- **predictive_goals**: Goals spawned from predictions
- **prediction_history**: Tracked outcomes of predictions
- **hypotheses**: Generated hypotheses with status
- **anticipatory_reflexes**: Executed anticipatory reflexes
- **dashboard_metrics**: Historical dashboard data
- **outcome_timeline**: Predicted vs actual outcomes

### Component Interactions

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Enhanced      │    │   Causal Drift   │    │   Hypothesis    │
│   Memory        │◄──►│   Predictor      │◄──►│   Generator     │
│   System        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Reflex        │    │   Reflex         │    │   Causal Audit  │
│   Anticipator   │◄──►│   Compressor     │◄──►│   Dashboard     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎉 Conclusion

The Predictive Causal Modeling & Hypothesis Generation system transforms MeRNSTA from a reactive system into an **anticipatory, self-repairing cognitive system**. It no longer waits to fail - it thinks ahead, like a living system.

**Key Benefits**:
- 🚀 **Proactive Repair**: Fixes issues before they become problems
- 🧠 **Deeper Understanding**: Generates hypotheses about root causes
- 📈 **Continuous Learning**: Improves predictions and strategies over time
- 🔮 **Future Awareness**: Anticipates and prepares for likely scenarios
- 🛡️ **Self-Protection**: Maintains cognitive stability automatically

This system represents a significant step toward truly intelligent, self-aware AI systems that can maintain their own cognitive health and adapt to changing circumstances. 