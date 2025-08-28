# MeRNSTA Enhanced Features

## Overview

This document describes the enhanced features added to MeRNSTA to make it smarter, leaner, and more abstract-aware:

1. **Belief Abstraction Layer** - Creates abstract beliefs from consistent fact clusters
2. **Reflex Compression & Reuse** - Groups reflex cycles into reusable patterns
3. **Memory Autocleaner (Garbage Collector)** - Automatically cleans up memory
4. **Enhanced Reasoning from Abstract Beliefs** - Uses beliefs to inform decision making

## 🧠 Belief Abstraction Layer

### Overview
The Belief Abstraction Layer automatically creates higher-level abstract beliefs from consistent patterns in fact clusters. These beliefs enable more sophisticated reasoning and pattern recognition.

### Features
- **Automatic Belief Creation**: Scans clusters every 10 goals for consistent patterns
- **Belief Criteria**: ≥3 consistent facts, 0 contradictions, low volatility
- **Abstract Statements**: Generates human-readable belief statements
- **Confidence Scoring**: Calculates belief confidence based on fact consistency
- **Usage Tracking**: Monitors how often beliefs are used in reasoning

### Implementation
- **File**: `storage/enhanced_memory.py`
- **Class**: `BeliefAbstractionLayer`
- **Database**: `belief_facts` table in `enhanced_memory.db`

### Usage
```python
from storage.enhanced_memory import BeliefAbstractionLayer

# Initialize belief system
belief_system = BeliefAbstractionLayer()

# Trigger belief scan
new_beliefs = belief_system.scan_for_beliefs()

# Get all beliefs
beliefs = belief_system.get_all_beliefs()

# Get beliefs for specific cluster
cluster_beliefs = belief_system.get_beliefs_by_cluster("cluster_1")
```

### CLI Command
```
/beliefs
```
Shows current abstracted beliefs with confidence scores and usage statistics.

## 🔄 Reflex Compression & Reuse

### Overview
The Reflex Compression system groups similar reflex cycles into reusable templates, improving response efficiency and effectiveness for similar drift scenarios.

### Features
- **Pattern Recognition**: Identifies similar reflex cycles based on strategy and goals
- **Template Creation**: Merges cycles into reusable patterns with effectiveness metrics
- **Strategy Suggestions**: Recommends templates for similar drift scenarios
- **Performance Tracking**: Monitors template success rates and usage patterns

### Implementation
- **File**: `storage/reflex_compression.py`
- **Class**: `ReflexCompressor`
- **Database**: `reflex_templates` table in `reflex_cycles.db`

### Usage
```python
from storage.reflex_compression import ReflexCompressor

# Initialize compressor
compressor = ReflexCompressor()

# Compress reflex cycles into templates
templates = compressor.compress_reflex_cycles(cycles)

# Get template suggestions for drift
suggestions = compressor.suggest_templates_for_drift(
    drift_score=0.8, token_id=123, cluster_id="cluster_1"
)
```

### CLI Command
```
/reflex_templates
```
Shows reusable repair patterns with success rates and usage statistics.

## 🧹 Memory Autocleaner (Garbage Collector)

### Overview
The Memory Autocleaner automatically cleans up memory by removing orphaned facts, dead contradictions, and compressing duplicates, keeping the system lean and efficient.

### Features
- **Orphaned Fact Removal**: Removes unlinked, low-score, unused facts
- **Dead Contradiction Cleanup**: Removes resolved or negated contradictions
- **Duplicate Compression**: Compresses facts with high cosine similarity
- **Background Operation**: Runs automatically every 5 minutes or 10 tasks
- **Comprehensive Logging**: Tracks all cleanup actions and their impact

### Implementation
- **File**: `storage/memory_autocleaner.py`
- **Class**: `MemoryCleaner`
- **Database**: `cleanup_log` table in `enhanced_memory.db`
- **Log File**: `memory_clean_log.jsonl`

### Usage
```python
from storage.memory_autocleaner import MemoryCleaner

# Initialize cleaner
cleaner = MemoryCleaner()

# Start background daemon
cleaner.start_daemon()

# Perform manual cleanup
stats = cleaner.perform_cleanup()

# Get cleanup statistics
stats = cleaner.get_cleanup_statistics()
```

### CLI Command
```
/memory_clean_log
```
Shows memory cleanup results and current memory usage statistics.

## 🧠 Enhanced Reasoning from Abstract Beliefs

### Overview
The Enhanced Reasoning Engine leverages abstract beliefs to inform strategy routing and execution planning for new goals, creating a more intelligent decision-making process.

### Features
- **Belief Retrieval**: Finds relevant beliefs for current goal context
- **Strategy Routing**: Uses beliefs to determine optimal repair strategies
- **Template Suggestions**: Recommends reflex templates based on belief patterns
- **Belief Tracing**: Tracks which beliefs influenced each decision
- **Confidence Scoring**: Calculates strategy confidence based on belief analysis

### Implementation
- **File**: `storage/enhanced_reasoning.py`
- **Class**: `EnhancedReasoningEngine`
- **Database**: `belief_traces` table in `enhanced_memory.db`

### Usage
```python
from storage.enhanced_reasoning import EnhancedReasoningEngine

# Initialize reasoning engine
reasoning_engine = EnhancedReasoningEngine()

# Process goal with enhanced reasoning
result = reasoning_engine.process_goal_with_beliefs(
    goal_id="goal_123",
    drift_score=0.8,
    token_id=456,
    cluster_id="cluster_1"
)

# Get belief trace for goal
trace = reasoning_engine.get_belief_trace("goal_123")
```

### CLI Command
```
/belief_trace [goal_id|token_id]
```
Shows beliefs that informed a specific decision or recent belief traces.

## 🔗 System Integration

### Enhanced Memory System Integration
All new features are integrated into the existing `EnhancedMemorySystem` class:

```python
from storage.enhanced_memory_system import EnhancedMemorySystem

# Initialize enhanced system
memory_system = EnhancedMemorySystem()

# New commands available
result = memory_system._handle_command("/beliefs", user_id, session_id)
result = memory_system._handle_command("/reflex_templates", user_id, session_id)
result = memory_system._handle_command("/memory_clean_log", user_id, session_id)
result = memory_system._handle_command("/belief_trace", user_id, session_id)
```

### CLI Integration
New commands added to the main CLI interface:

- `beliefs` - Show current abstracted beliefs per cluster
- `reflex_templates` - View reusable repair patterns
- `memory_clean_log` - View memory autocleaner results
- `belief_trace [goal_id|token_id]` - Show beliefs that informed a decision

## 📊 Database Schema

### New Tables

#### `belief_facts`
```sql
CREATE TABLE belief_facts (
    belief_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    abstract_statement TEXT NOT NULL,
    supporting_facts TEXT NOT NULL,
    vector TEXT,
    created_at REAL NOT NULL,
    confidence REAL DEFAULT 0.0,
    volatility_score REAL DEFAULT 0.0,
    last_updated REAL NOT NULL,
    usage_count INTEGER DEFAULT 0,
    coherence_score REAL DEFAULT 0.0
);
```

#### `reflex_templates`
```sql
CREATE TABLE reflex_templates (
    template_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    pattern_signature TEXT NOT NULL,
    goal_pattern TEXT NOT NULL,
    execution_pattern TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    avg_score REAL DEFAULT 0.0,
    cluster_overlap REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL,
    source_cycles TEXT NOT NULL,
    effectiveness_notes TEXT
);
```

#### `cleanup_log`
```sql
CREATE TABLE cleanup_log (
    action_id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    timestamp REAL NOT NULL,
    memory_saved INTEGER DEFAULT 0,
    confidence_impact REAL DEFAULT 0.0
);
```

#### `belief_traces`
```sql
CREATE TABLE belief_traces (
    trace_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    token_id INTEGER,
    cluster_id TEXT,
    considered_beliefs TEXT NOT NULL,
    belief_influences TEXT NOT NULL,
    final_strategy TEXT NOT NULL,
    strategy_confidence REAL NOT NULL,
    timestamp REAL NOT NULL,
    reasoning_notes TEXT
);
```

## 🧪 Testing

### Test Script
Run the comprehensive test script to verify all enhanced features:

```bash
python test_enhanced_systems.py
```

This script tests:
- Belief Abstraction Layer functionality
- Reflex Compression & Reuse system
- Memory Autocleaner operations
- Enhanced Reasoning engine
- System integration

### Individual Testing
Each system can be tested independently:

```python
# Test belief abstraction
from storage.enhanced_memory import BeliefAbstractionLayer
belief_system = BeliefAbstractionLayer()
beliefs = belief_system.scan_for_beliefs()

# Test reflex compression
from storage.reflex_compression import ReflexCompressor
compressor = ReflexCompressor()
templates = compressor.get_all_templates()

# Test memory cleaner
from storage.memory_autocleaner import MemoryCleaner
cleaner = MemoryCleaner()
stats = cleaner.perform_cleanup()

# Test enhanced reasoning
from storage.enhanced_reasoning import EnhancedReasoningEngine
reasoning = EnhancedReasoningEngine()
result = reasoning.process_goal_with_beliefs("test_goal", 0.8)
```

## 🚀 Getting Started

### 1. Initialize Enhanced Systems
The enhanced systems are automatically initialized when you create an `EnhancedMemorySystem` instance:

```python
from storage.enhanced_memory_system import EnhancedMemorySystem

# This automatically initializes all enhanced features
memory_system = EnhancedMemorySystem()
```

### 2. Start Memory Cleaner Daemon
```python
from storage.memory_autocleaner import MemoryCleaner

cleaner = MemoryCleaner()
cleaner.start_daemon()  # Runs in background
```

### 3. Use Enhanced Commands
```python
# Process input normally - enhanced features work automatically
result = memory_system.process_input("I love Python programming", "user1", "session1")

# Use new commands
beliefs_result = memory_system._handle_command("/beliefs", "user1", "session1")
templates_result = memory_system._handle_command("/reflex_templates", "user1", "session1")
```

### 4. Monitor System Performance
```python
# Get belief statistics
from storage.enhanced_memory import BeliefAbstractionLayer
belief_stats = BeliefAbstractionLayer().get_belief_statistics()

# Get template statistics
from storage.reflex_compression import ReflexCompressor
template_stats = ReflexCompressor().get_template_statistics()

# Get cleanup statistics
from storage.memory_autocleaner import MemoryCleaner
cleanup_stats = MemoryCleaner().get_cleanup_statistics()
```

## 📈 Performance Benefits

### Memory Efficiency
- **Automatic cleanup** reduces memory footprint
- **Duplicate compression** saves storage space
- **Orphaned fact removal** maintains data quality

### Reasoning Quality
- **Abstract beliefs** enable higher-level reasoning
- **Pattern recognition** improves decision making
- **Template reuse** increases repair effectiveness

### System Intelligence
- **Belief-driven decisions** are more contextually aware
- **Learning from patterns** improves over time
- **Automatic optimization** reduces manual intervention

## 🔧 Configuration

### Belief Abstraction
```python
# Configure belief creation thresholds
belief_system = BeliefAbstractionLayer()
belief_system.min_facts_for_belief = 3  # Minimum facts for belief
belief_system.max_contradictions = 0    # Maximum contradictions allowed
belief_system.max_volatility = 0.3      # Maximum volatility threshold
```

### Reflex Compression
```python
# Configure template creation
compressor = ReflexCompressor()
compressor.similarity_threshold = 0.7      # Similarity for grouping
compressor.min_cycles_for_template = 3     # Minimum cycles for template
compressor.max_templates_per_strategy = 10 # Templates per strategy
```

### Memory Cleaner
```python
# Configure cleanup thresholds
cleaner = MemoryCleaner()
cleaner.orphan_score_threshold = 0.3           # Low score threshold
cleaner.orphan_age_threshold = 7 * 24 * 3600   # Age threshold (7 days)
cleaner.duplicate_similarity_threshold = 0.95  # Duplicate similarity
cleaner.contradiction_resolution_threshold = 0.8 # Resolution threshold
```

## 🐛 Troubleshooting

### Common Issues

1. **No beliefs created**: Ensure you have enough facts with consistent patterns
2. **No templates generated**: Run more reflex cycles to build template patterns
3. **Cleanup not working**: Check if the daemon is running and thresholds are appropriate
4. **Import errors**: Ensure all required dependencies are installed

### Debug Mode
Enable debug logging to see detailed system operation:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Triggers
```python
# Manually trigger belief scan
belief_system.scan_for_beliefs()

# Manually trigger cleanup
cleaner.perform_cleanup()

# Manually compress reflex cycles
compressor.compress_reflex_cycles(cycles)
```

## 📚 API Reference

### BeliefAbstractionLayer
- `scan_for_beliefs()` - Scan for new beliefs
- `get_all_beliefs(limit=None)` - Get all beliefs
- `get_beliefs_by_cluster(cluster_id)` - Get beliefs for cluster
- `get_belief_statistics()` - Get belief statistics
- `increment_usage(belief_id)` - Increment belief usage

### ReflexCompressor
- `compress_reflex_cycles(cycles)` - Compress cycles into templates
- `get_all_templates(limit=None)` - Get all templates
- `suggest_templates_for_drift(drift_score, token_id, cluster_id)` - Get suggestions
- `get_template_statistics()` - Get template statistics
- `increment_usage(template_id)` - Increment template usage

### MemoryCleaner
- `start_daemon()` - Start background cleanup daemon
- `stop_daemon()` - Stop background daemon
- `perform_cleanup()` - Perform manual cleanup
- `get_cleanup_log(limit=None)` - Get cleanup log
- `get_cleanup_statistics()` - Get cleanup statistics
- `get_memory_usage_stats()` - Get memory usage statistics

### EnhancedReasoningEngine
- `process_goal_with_beliefs(goal_id, drift_score, token_id, cluster_id)` - Process goal
- `get_belief_trace(goal_id)` - Get belief trace for goal
- `get_belief_traces_by_token(token_id, limit=None)` - Get traces for token
- `get_recent_belief_traces(limit=20)` - Get recent traces
- `get_belief_reasoning_statistics()` - Get reasoning statistics

## 🤝 Contributing

When contributing to the enhanced features:

1. **Follow existing patterns** in the codebase
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for any API changes
4. **Maintain backward compatibility** where possible
5. **Use type hints** for better code clarity

## 📄 License

These enhanced features are part of the MeRNSTA project and follow the same licensing terms. 