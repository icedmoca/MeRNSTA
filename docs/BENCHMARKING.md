# Performance Benchmarking Guide

This guide helps you understand MeRNSTA's performance characteristics on your specific hardware configuration.

## Quick Benchmark

Run the complete performance suite:

```bash
python benchmarks/performance_suite.py
```

This tests:
- **Memory Operations**: Storage, retrieval, search at different scales
- **Contradiction Detection**: Real-time performance
- **Embedding Generation**: Text processing latency
- **Concurrent Load**: Multi-user simulation

## Expected Performance

### Hardware Impact

**CPU Performance:**
- **High-end (16+ cores)**: 1-2ms per operation
- **Mid-range (8 cores)**: 2-4ms per operation  
- **Low-end (4 cores)**: 4-8ms per operation

**Memory Impact:**
- **16GB+ RAM**: Full caching, optimal performance
- **8GB RAM**: Partial caching, good performance
- **4GB RAM**: Limited caching, reduced performance

**Storage Impact:**
- **NVMe SSD**: 0.1-0.5ms database operations
- **SATA SSD**: 0.5-2ms database operations
- **HDD**: 5-20ms database operations

### Scale Expectations

| Facts | Storage (per fact) | Search | Memory Usage |
|-------|-------------------|--------|--------------|
| 1K    | 0.5-2ms          | 10-20ms | 50MB        |
| 10K   | 1-3ms            | 20-40ms | 200MB       |
| 100K  | 2-5ms            | 40-80ms | 800MB       |
| 1M    | 5-10ms           | 100-200ms | 4GB      |

## Optimization Tips

### For Better Performance

1. **Use SSD storage** for database files
2. **Increase connection pool** size in config
3. **Enable PostgreSQL** for large datasets
4. **Use Redis caching** for embeddings
5. **Run Ollama locally** to reduce network latency

### Configuration Tuning

```yaml
# config.yaml optimizations
database:
  max_connections: 20  # Increase for high concurrency

network:
  ollama_host: "http://localhost:11434"  # Local = faster

behavior:
  enable_compression: true  # Reduces memory usage
  
multimodal:
  embedding_cache_size: 20000  # Larger cache
```

## Benchmark Interpretation

### What Good Performance Looks Like

- **Storage**: <5ms per fact at 50k facts
- **Search**: <100ms for semantic search
- **API Response**: <200ms end-to-end
- **Contradiction Detection**: <50ms

### Warning Signs

- **Storage >20ms**: Check storage performance, database locks
- **Search >500ms**: Enable compression, check embedding cache
- **High Memory Usage**: Enable auto-cleanup, check for leaks
- **API Timeouts**: Scale workers, check database connections

## Hardware Recommendations

### Development

**Minimum:**
- 4GB RAM
- 2 CPU cores  
- 1GB available disk
- Any Python 3.10+ environment

**Recommended:**
- 8GB RAM
- 4 CPU cores
- SSD storage
- Local Ollama installation

### Production

**Small Scale (1K-50K facts):**
- 8GB RAM
- 4-8 CPU cores
- SSD storage
- PostgreSQL database

**Medium Scale (50K-500K facts):**
- 16GB RAM  
- 8-16 CPU cores
- NVMe storage
- PostgreSQL with read replicas
- Redis cache

**Large Scale (500K+ facts):**
- 32GB+ RAM
- 16+ CPU cores
- Enterprise NVMe storage
- PostgreSQL cluster
- Dedicated Redis instance
- Load balancing

## Troubleshooting Performance

### Slow Storage

**Symptoms:** High storage latency, timeouts
**Solutions:**
- Check disk I/O with `iostat -x 1`
- Switch to PostgreSQL from SQLite
- Increase `max_connections` in config
- Enable WAL mode (auto-enabled)

### Memory Issues

**Symptoms:** High RAM usage, OOM errors
**Solutions:**
- Enable memory compression
- Reduce `embedding_cache_size`
- Enable auto-cleanup tasks
- Monitor with `/metrics` endpoint

### API Slowness

**Symptoms:** High response times, timeouts
**Solutions:**
- Scale API workers: `--workers 4`
- Enable Redis caching
- Check database performance
- Monitor concurrent connections

### Embedding Latency

**Symptoms:** Slow text processing
**Solutions:**
- Run Ollama locally
- Use faster embedding model
- Increase embedding cache
- Batch process when possible

## Continuous Monitoring

Monitor performance in production:

```bash
# Real-time metrics
curl http://localhost:8000/metrics

# System health
curl http://localhost:8000/health/detailed

# Performance dashboard
open http://localhost:8000/dashboard
```

Set up alerting for:
- Response time >500ms
- Memory usage >80%
- Error rate >5%
- Database connections >80% of pool

## See Also

- [Enterprise Scaling Guide](ENTERPRISE.md)
- [Configuration Tuning](CONFIGURATION.md)
- [Production Deployment](INSTALLATION.md) 