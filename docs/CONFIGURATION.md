# Configuration Reference

MeRNSTA uses a centralized configuration system with **zero hardcoding policy**. All settings are externalized and hot-reloadable.

## Configuration Files

### `config.yaml` - Main Configuration
The primary configuration file containing all system settings.

### `.env` - Environment Variables
Sensitive settings and environment-specific overrides.

## Core Configuration Sections

### Database Settings
```yaml
database:
  default_path: "memory.db"
  max_connections: 10
  retry_delay: 0.1
  retry_attempts: 5
  backup_interval: 3600
```

- `default_path`: SQLite database file path
- `max_connections`: Connection pool size
- `retry_delay`: Seconds between retry attempts
- `retry_attempts`: Max retries for database operations
- `backup_interval`: Automatic backup interval (seconds)

### Network Configuration
```yaml
network:
  ollama_host: "http://127.0.0.1:11434"
  bind_host: "0.0.0.0"
  api_port: 8000
  dashboard_port: 8001
  port_retry_attempts: 5
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8000"
```

- `ollama_host`: Ollama API endpoint
- `bind_host`: API server bind address
- `api_port`: Primary API server port
- `dashboard_port`: Dashboard server port
- `port_retry_attempts`: Ports to try if default unavailable
- `cors_origins`: Allowed CORS origins

### Memory Behavior
```yaml
volatility_thresholds:
  stable: 0.3
  medium: 0.6
  high: 0.8
  clarification: 1.0

default_thresholds:
  stable_facts: 0.3
  unstable_facts: 0.8
  high_confidence: 0.9
```

- `volatility_thresholds`: Fact stability classification
- `default_thresholds`: General confidence thresholds

### Personality Profiles
```yaml
personality_profiles:
  neutral:
    name: "Neutral"
    multiplier: 1.0
    description: "Balanced memory behavior"
  
  loyal:
    name: "Loyal"
    multiplier: 0.7
    description: "Slower decay, stronger retention"
  
  skeptical:
    name: "Skeptical"
    multiplier: 1.5
    description: "Faster decay, questions everything"
  
  emotional:
    name: "Emotional"
    multiplier: 1.2
    description: "Emotion-biased reinforcement"
  
  analytical:
    name: "Analytical"
    multiplier: 0.8
    description: "Precise, methodical memory"
```

### Memory Routing Modes
```yaml
memory_routing_modes:
  MAC:
    name: "Memory-Augmented Context"
    description: "Inject relevant facts into context"
  
  MAG:
    name: "Memory-Augmented Generation"
    description: "Guide token generation with memory"
  
  MEL:
    name: "Memory-Enhanced Learning"
    description: "Full memory integration"
```

### Behavior Settings
```yaml
behavior:
  auto_reconcile: true
  emotion_bias: true
  enable_compression: true
  semantic_drift_threshold: 0.35
```

- `auto_reconcile`: Enable automatic contradiction resolution
- `emotion_bias`: Apply emotion-based memory biasing
- `enable_compression`: Enable automatic memory compression
- `semantic_drift_threshold`: Threshold for drift detection

### Multi-Modal Settings
```yaml
multimodal:
  embedding_model: "mistral"
  media_storage_path: "media/"
  similarity_threshold: 0.7
  max_cluster_size: 10
```

- `embedding_model`: Model for generating embeddings
- `media_storage_path`: Directory for uploaded media
- `similarity_threshold`: Minimum similarity for matching
- `max_cluster_size`: Maximum facts per cluster

## Environment Variables (.env)

### Security
```bash
API_SECURITY_TOKEN=your-secure-token-here
```

### Database
```bash
DATABASE_URL=sqlite:///memory.db
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost:5432/mernsta
```

### Cache
```bash
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
ENABLE_CACHING=true
```

### Monitoring
```bash
LOG_LEVEL=INFO
PROMETHEUS_ENABLED=true
ENABLE_TRACING=true
```

### Background Tasks
```bash
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
RECONCILIATION_INTERVAL=300
COMPRESSION_INTERVAL=3600
```

## Advanced Configuration

### Custom Fact Extraction Patterns
```yaml
fact_extraction_patterns:
  - pattern: "I (love|like|enjoy) (.+)"
    confidence: 0.9
  - pattern: "My (.+) is (.+)"
    confidence: 0.8
  - pattern: "I (hate|dislike) (.+)"
    confidence: 0.9
```

Add custom regex patterns for fact extraction with confidence scores.

### Subject Mapping
```yaml
subject_mapping:
  "fav color": "favorite color"
  "fav food": "favorite food"
  "my name": "name"
```

Normalize subject variations to canonical forms.

### Question Words
```yaml
question_words:
  - "what"
  - "which"
  - "where"
  - "when"
  - "who"
  - "how"
```

Words that indicate queries rather than statements.

## Configuration Validation

MeRNSTA validates all configuration on startup:

```python
# All required sections must be present
required_sections = [
    "volatility_thresholds", "personality_profiles", 
    "memory_routing_modes", "network", "database"
]

# Automatic validation ensures type safety
assert 0.0 <= compression_threshold <= 1.0
assert max_connections > 0
assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
```

## Hot-Reload

Configuration changes are automatically detected and applied:

```bash
# Edit config.yaml
vim config.yaml

# Changes are automatically applied to running services
# No restart required for most settings
```

Monitored files:
- `config.yaml`
- `.env`

## Configuration Examples

### Development Setup
```yaml
database:
  default_path: "dev_memory.db"
network:
  api_port: 8000
  ollama_host: "http://localhost:11434"
behavior:
  auto_reconcile: false  # Manual control for debugging
```

### Production Setup
```yaml
database:
  default_path: "/data/mernsta.db"
  max_connections: 50
network:
  bind_host: "0.0.0.0"
  api_port: 80
behavior:
  enable_compression: true
  auto_reconcile: true
```

### Enterprise Setup
```yaml
database:
  # Use PostgreSQL for scale
  # Set via DATABASE_URL environment variable
network:
  cors_origins:
    - "https://yourdomain.com"
    - "https://app.yourdomain.com"
```

## Troubleshooting

### Configuration Errors
```bash
# Check configuration validity
python -c "from config.settings import validate_config; validate_config()"
```

### View Loaded Configuration
```bash
# Display current configuration
python -c "from config.settings import _cfg; import json; print(json.dumps(_cfg, indent=2))"
```

### Reset to Defaults
```bash
# Backup current config
cp config.yaml config.yaml.backup

# Reset to defaults (if needed)
git checkout config.yaml
```

## Best Practices

1. **Always backup** `config.yaml` before major changes
2. **Use environment variables** for sensitive settings
3. **Test configuration** changes in development first
4. **Monitor logs** after configuration changes
5. **Use version control** for configuration files
6. **Document custom** patterns and mappings

## See Also

- [Installation Guide](INSTALLATION.md)
- [API Documentation](API.md)
- [Enterprise Features](ENTERPRISE.md) 