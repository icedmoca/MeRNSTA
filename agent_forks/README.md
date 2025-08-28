# Agent Forks Directory

This directory contains forked and mutated agent instances created by the Phase 22 Recursive Self-Replication system.

## Directory Structure

```
agent_forks/
├── <fork_id_1>/
│   ├── <agent_name>_<short_id>.py    # Mutated agent source code
│   ├── test_results.json             # Test execution results
│   └── metadata.json                 # Fork metadata and lineage
├── <fork_id_2>/
│   ├── <agent_name>_<short_id>.py
│   ├── test_results.json
│   └── metadata.json
└── README.md                         # This file
```

## Fork Lifecycle

1. **Creation**: Agent source code is copied to a new UUID-named directory
2. **Mutation**: Code is mutated using various strategies (function renaming, logic tweaks, prompt adjustments)
3. **Testing**: Fork is tested for syntax validity and basic functionality
4. **Evaluation**: Performance is scored and compared against survival threshold
5. **Reintegration**: High-performing forks are kept, low-performing ones are pruned

## File Types

- **Agent Files**: `<agent_name>_<short_id>.py` - The actual mutated agent code
- **Test Results**: `test_results.json` - Results from testing the forked agent
- **Metadata**: `metadata.json` - Fork origin, mutations applied, lineage information

## CLI Commands

- `/fork_agent <agent_name>` - Create a new fork
- `/run_fork <fork_id>` - Execute and test a fork
- `/score_forks` - Evaluate all fork performance
- `/prune_forks` - Remove low-performing forks

## Safety Features

- Syntax validation before and after mutations
- Isolated execution environment
- Automatic backup of original agents
- Performance-based survival thresholds
- Rollback capabilities for failed mutations

## Monitoring

Fork activity is logged to `output/fork_logs.jsonl` with events including:
- Fork creation
- Mutations applied
- Test results
- Performance scores
- Pruning decisions

This system enables MeRNSTA to evolve and improve its agent population through
controlled genetic-style mutations and natural selection based on performance.