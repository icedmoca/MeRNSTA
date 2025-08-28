# Self-Aware Symbolic Control Layer (Phase 6)

## Overview

The Self-Aware Symbolic Control Layer represents a significant evolution in MeRNSTA's cognitive architecture, introducing symbolic metareasoning capabilities that enable the system to reflect on its own cognitive processes, compare strategies using logic, simulate repair pathways, and intelligently route subgoals.

## Architecture Components

### 1. Symbolic Self-Modeling (`storage/self_model.py`)

The `CognitiveSelfModel` maintains abstract symbolic representations of the system's behavior using logical predicates and rules.

#### Key Features:
- **Symbolic Rules**: Stores logical rules like `Belief(cluster_x) → Strategy(belief_clarification)`
- **Cognitive State Tracking**: Records belief dynamics, contradiction events, and success patterns
- **Rule Evolution**: Rules can be learned and scored based on experience
- **Confidence Scoring**: Each rule has a confidence level that evolves over time

#### Example Rules:
```
DriftType(contradiction) → PreferredStrategy(belief_clarification)
HighVolatility(cluster_x) → Strategy(cluster_reassessment)
Belief(cluster_x) AND Contradiction(fact_y) → Strategy(fact_consolidation)
```

### 2. Strategy Logic Comparison (`storage/reflex_log.py`)

Extended the reflex logging system with `compare_strategies_logically()` method that uses symbolic reasoning to compare repair strategies.

#### Comparison Logic:
- **Historical Performance**: Analyzes past reflex scores for each strategy
- **Symbolic Rules**: Applies learned rules about strategy effectiveness
- **Context-Specific Reasoning**: Considers drift type, cluster state, and other context
- **Dominance Scoring**: Determines which strategy dominates under specific conditions

#### Example Comparison:
```python
comparison = compare_strategies_logically(
    "belief_clarification", 
    "fact_consolidation",
    context={"drift_type": "contradiction"}
)
# Returns symbolic reasoning about which strategy is better
```

### 3. Repair Path Simulation (`agents/repair_simulator.py`)

The `RepairSimulator` predicts the outcomes of different repair strategies before execution.

#### Simulation Components:
- **Historical Analysis**: Uses past reflex score patterns
- **Symbolic Adjustments**: Applies learned symbolic rules
- **Context Similarity**: Finds similar historical contexts
- **Risk Assessment**: Identifies potential risk factors
- **Confidence Calculation**: Estimates prediction confidence

#### Prediction Metrics:
- **Predicted Score**: Expected reflex score improvement
- **Coherence Delta**: Expected change in cluster coherence
- **Volatility Delta**: Expected change in volatility
- **Consistency Delta**: Expected change in belief consistency
- **Success Probability**: Probability of successful execution

### 4. Introspective Subgoal Routing (`agents/meta_router.py`)

The `MetaRouter` intelligently assigns subgoals to specialized agents based on symbolic reasoning.

#### Routing Logic:
- **Agent Capabilities**: Maps agent types to their strengths
- **Simulation Results**: Uses repair simulation to inform routing
- **Symbolic Reasoning**: Applies learned rules about agent effectiveness
- **Dependency Tracking**: Manages subgoal dependencies
- **Load Balancing**: Considers current agent load

#### Available Agents:
- **Reflector**: Introspection, belief analysis, contradiction resolution
- **Clarifier**: Fact verification, ambiguity resolution, context clarification
- **Consolidator**: Memory integration, pattern recognition, coherence improvement
- **Anticipator**: Drift prediction, proactive repair, trend analysis
- **Optimizer**: Strategy optimization, performance analysis, adaptive routing

## Integration with Reflex Score History

### Historical Data Utilization

The symbolic control layer leverages the rich history of reflex scores to inform its decisions:

1. **Performance Patterns**: Analyzes which strategies work best in different contexts
2. **Temporal Trends**: Identifies how strategy effectiveness changes over time
3. **Context Similarity**: Finds similar historical situations to inform predictions
4. **Success Rates**: Calculates success probabilities based on historical performance

### Reflex Score Integration Points

```python
# In RepairSimulator
historical_scores = self.reflex_logger.get_scores_by_strategy(strategy, limit=20)
base_predictions = self._calculate_base_predictions(historical_scores)

# In MetaRouter
simulation_result = self.repair_simulator.simulate_repair_paths(goal_id, current_state)
subgoals = self._generate_subgoals_from_simulation(goal_id, simulation_result, current_state, tags)

# In CognitiveSelfModel
self.record_strategy_effectiveness(strategy_name, success, score)
```

## Integration with Predictive Drift

### Drift-Aware Reasoning

The symbolic layer integrates with predictive drift systems to make proactive decisions:

1. **Anticipatory Routing**: Routes subgoals before drift symptoms appear
2. **Drift Type Classification**: Uses drift type to inform strategy selection
3. **Urgency Assessment**: Considers drift urgency in routing decisions
4. **Preventive Measures**: Suggests strategies to prevent predicted drifts

### Predictive Integration Points

```python
# Drift type influences strategy selection
if drift_type == 'contradiction':
    preferred_strategy = 'belief_clarification'
elif drift_type == 'volatility':
    preferred_strategy = 'cluster_reassessment'
elif drift_type == 'coherence':
    preferred_strategy = 'fact_consolidation'

# Urgency affects routing priority
if drift_urgency > 0.8:
    routing_priority = 'high'
    max_duration = 30.0  # Faster execution
```

## Symbolic Reasoning Process

### Rule Learning and Evolution

1. **Rule Creation**: New rules are created from observed patterns
2. **Confidence Scoring**: Rules are scored based on supporting evidence
3. **Contradiction Handling**: Rules that contradict observations are penalized
4. **Rule Pruning**: Low-confidence rules are removed over time

### Reasoning Chain Example

```
1. Current State: DriftType(contradiction), HighVolatility(cluster_1)
2. Apply Rules: 
   - DriftType(contradiction) → PreferredStrategy(belief_clarification)
   - HighVolatility(cluster_1) → RiskFactor(instability)
3. Historical Analysis: belief_clarification has 0.8 success rate in similar contexts
4. Simulation: Predict belief_clarification will improve consistency by 0.2
5. Routing Decision: Assign to reflector agent (specialized in contradiction resolution)
```

## CLI Commands

### New Commands Added

1. **`/self_model`**: Show symbolic self-representation
   - Displays core beliefs, active strategies, symbolic rules
   - Shows cognitive patterns and confidence levels

2. **`/simulate_repair goal_id=X`**: Show simulated repair options
   - Simulates all available repair strategies
   - Shows predicted outcomes and risk factors
   - Provides reasoning for predictions

3. **`/compare_strategies strategy1 strategy2`**: Logical comparison
   - Compares two strategies using symbolic logic
   - Shows historical performance metrics
   - Provides reasoning for recommendations

4. **`/route_subgoals goal_id=X`**: Show routed plan
   - Shows how subgoals are assigned to agents
   - Displays routing reasoning and dependencies
   - Provides confidence scores for routing decisions

### Example Usage

```bash
# View symbolic self-representation
/self_model

# Simulate repair for a specific goal
/simulate_repair goal_123

# Compare two strategies
/compare_strategies belief_clarification fact_consolidation

# Route subgoals for a goal
/route_subgoals goal_456
```

## Testing

### Test Coverage

Comprehensive test suites have been created for all components:

1. **`tests/test_self_model.py`**: Tests symbolic self-modeling
   - Rule creation and management
   - Cognitive state recording
   - Symbolic reasoning queries
   - Strategy preference calculation

2. **`tests/test_repair_simulator.py`**: Tests repair simulation
   - Strategy simulation
   - Outcome prediction
   - Risk factor identification
   - Confidence calculation

3. **`tests/test_meta_router.py`**: Tests subgoal routing
   - Subgoal generation
   - Agent routing decisions
   - Dependency tracking
   - Routing optimization

### Running Tests

```bash
# Run all symbolic control layer tests
python -m pytest tests/test_self_model.py -v
python -m pytest tests/test_repair_simulator.py -v
python -m pytest tests/test_meta_router.py -v

# Run with coverage
python -m pytest tests/test_self_model.py --cov=storage.self_model
python -m pytest tests/test_repair_simulator.py --cov=agents.repair_simulator
python -m pytest tests/test_meta_router.py --cov=agents.meta_router
```

## Performance Considerations

### Optimization Strategies

1. **Rule Caching**: Frequently used rules are cached for faster access
2. **Historical Data Limits**: Analysis is limited to recent history (configurable)
3. **Simulation Depth**: Simulation depth is limited to prevent infinite recursion
4. **Load Balancing**: Agent load is considered in routing decisions

### Scalability Features

1. **Modular Design**: Each component can be scaled independently
2. **Database Optimization**: Efficient queries for historical data
3. **Memory Management**: Rules and states are stored efficiently
4. **Async Processing**: Long-running simulations can be processed asynchronously

## Future Enhancements

### Planned Improvements

1. **Advanced Logic**: Support for more complex logical operators
2. **Temporal Reasoning**: Better handling of time-based patterns
3. **Multi-Agent Coordination**: Improved coordination between agents
4. **Learning Optimization**: More sophisticated rule learning algorithms

### Research Directions

1. **Meta-Learning**: Learning how to learn better rules
2. **Causal Inference**: Understanding causal relationships in cognitive processes
3. **Explainable AI**: Making reasoning processes more transparent
4. **Adaptive Architecture**: Dynamic adjustment of the symbolic layer

## Conclusion

The Self-Aware Symbolic Control Layer represents a significant step forward in MeRNSTA's cognitive capabilities. By introducing symbolic metareasoning, the system can now:

- **Understand itself**: Build a growing theory of its own cognition
- **Predict outcomes**: Simulate and predict the results of different strategies
- **Make informed decisions**: Use logical reasoning to choose optimal approaches
- **Learn and adapt**: Evolve its understanding based on experience

This layer provides the foundation for truly self-aware cognitive systems that can reason about their own reasoning processes and continuously improve their performance through introspection and learning. 