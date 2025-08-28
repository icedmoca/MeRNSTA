# MeRNSTA Enterprise Deployment Guide

This guide covers deploying MeRNSTA as an enterprise-grade, scalable memory-aware AI system.

## 🏗️ Architecture Overview

MeRNSTA is now a production-ready system with:

- **Environment-driven configuration** with hot-reload capability
- **Celery + Redis task queue** for background processing
- **Structured logging** with correlation IDs and JSON output
- **Prometheus metrics** and comprehensive health checks
- **Redis caching** for embeddings and cluster centroids
- **Database indexing** for 1M+ facts performance
- **Security middleware** with rate limiting and input validation

## 📋 Prerequisites

### System Requirements
- Python 3.11+
- Redis 6.0+
- SQLite (dev) or PostgreSQL (prod)
- 4GB+ RAM
- 10GB+ disk space

### Dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### 1. Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database Configuration
DATABASE_URL=sqlite:///memory.db
MAX_CONNECTIONS=10
DATABASE_TIMEOUT=30.0

# Memory System Configuration
MAX_FACTS=1000000
COMPRESSION_THRESHOLD=0.8
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.7

# Background Tasks Configuration
RECONCILIATION_INTERVAL=300
COMPRESSION_INTERVAL=3600
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring Configuration
METRICS_PORT=9090
LOG_LEVEL=INFO
ENABLE_TRACING=true
PROMETHEUS_ENABLED=true

# Security Configuration
API_SECURITY_TOKEN=your-secure-token-here
RATE_LIMIT=100
RATE_LIMIT_WINDOW=60
DISABLE_RATE_LIMIT=false

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
ENABLE_CACHING=true

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=mistral

# Feature Flags
ENABLE_COMPRESSION=true
ENABLE_AUTO_RECONCILIATION=true
ENABLE_PERSONALITY_BIASING=true
ENABLE_EMOTION_ANALYSIS=true

# Performance Configuration
BATCH_SIZE=1000
EMBEDDING_CACHE_SIZE=10000
MAX_CONCURRENT_TASKS=10

# Environment
ENVIRONMENT=production
DEBUG=false
```

### 2. Production Configuration

For production deployment:

```bash
# Use PostgreSQL instead of SQLite
DATABASE_URL=postgresql://user:password@localhost:5432/mernsta

# Increase limits for high load
MAX_FACTS=5000000
RATE_LIMIT=1000
MAX_CONCURRENT_TASKS=50

# Enable all monitoring
ENABLE_TRACING=true
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO

# Secure token
API_SECURITY_TOKEN=your-super-secure-production-token
```

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://mernsta:password@db:5432/mernsta
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  worker:
    build: .
    command: celery -A tasks.task_queue worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://mernsta:password@db:5432/mernsta
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  beat:
    build: .
    command: celery -A tasks.task_queue beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://mernsta:password@db:5432/mernsta
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mernsta
      - POSTGRES_USER=mernsta
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 2: Kubernetes

Create `k8s/` directory with manifests:

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mernsta

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mernsta-config
  namespace: mernsta
data:
  DATABASE_URL: "postgresql://mernsta:password@postgres:5432/mernsta"
  REDIS_URL: "redis://redis:6379/0"
  CELERY_BROKER_URL: "redis://redis:6379/0"
  LOG_LEVEL: "INFO"
  ENABLE_TRACING: "true"
  PROMETHEUS_ENABLED: "true"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mernsta-secret
  namespace: mernsta
type: Opaque
data:
  API_SECURITY_TOKEN: <base64-encoded-token>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mernsta-api
  namespace: mernsta
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mernsta-api
  template:
    metadata:
      labels:
        app: mernsta-api
    spec:
      containers:
      - name: api
        image: mernsta:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: mernsta-config
        - secretRef:
            name: mernsta-secret
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mernsta-api
  namespace: mernsta
spec:
  selector:
    app: mernsta-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 📊 Monitoring Setup

### 1. Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mernsta-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### 2. Grafana Dashboards

Import these dashboards to Grafana:

1. **MeRNSTA Overview Dashboard**
   - Memory statistics
   - API performance
   - Background task status
   - System resources

2. **Memory Analytics Dashboard**
   - Fact count trends
   - Contradiction scores
   - Compression ratios
   - Cluster statistics

3. **System Health Dashboard**
   - Database performance
   - Cache hit ratios
   - Error rates
   - Response times

## 🔍 Health Checks

### Available Endpoints

- `/health` - Comprehensive health check
- `/health/live` - Liveness probe (Kubernetes)
- `/health/ready` - Readiness probe (Kubernetes)
- `/health/detailed` - Detailed health with performance metrics
- `/metrics` - Prometheus metrics

### Health Check Components

1. **Database Health**
   - Connection test
   - Query performance
   - Fact count

2. **Memory System Health**
   - Total facts
   - Contradiction count
   - Trust scores
   - Compression status

3. **Background Tasks Health**
   - Redis connection
   - Celery worker status
   - Task queue status

4. **System Resources**
   - CPU usage
   - Memory usage
   - Disk usage

5. **LLM Service Health**
   - Ollama connection
   - Available models
   - Response times

6. **Cache Health**
   - Redis connection
   - Cache operations
   - Hit ratios

## 🚦 Startup Sequence

### 1. Start Infrastructure

```bash
# Start Redis
redis-server

# Start PostgreSQL (if using)
docker run -d --name postgres \
  -e POSTGRES_DB=mernsta \
  -e POSTGRES_USER=mernsta \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

# Start Prometheus
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 2. Start Application Components

```bash
# Start Celery worker
celery -A tasks.task_queue worker --loglevel=info

# Start Celery beat (scheduler)
celery -A tasks.task_queue beat --loglevel=info

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics

# Check task queue
curl http://localhost:8000/health/detailed
```

## 🔧 Maintenance

### Database Maintenance

```bash
# Backup database
pg_dump mernsta > backup.sql

# Optimize database
VACUUM ANALYZE;

# Check indexes
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch 
FROM pg_stat_user_indexes;
```

### Cache Maintenance

```bash
# Clear cache
curl -X POST http://localhost:8000/api/memory/clear-cache

# Check cache stats
curl http://localhost:8000/api/memory/cache-stats
```

### Log Management

```bash
# Rotate logs
logrotate /etc/logrotate.d/mernsta

# Monitor logs
tail -f logs/mernsta.log
tail -f logs/background.log
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check DATABASE_URL
   - Verify PostgreSQL is running
   - Check connection pool settings

2. **Redis Connection Errors**
   - Check REDIS_URL
   - Verify Redis is running
   - Check network connectivity

3. **Celery Worker Issues**
   - Check CELERY_BROKER_URL
   - Verify Redis is accessible
   - Check worker logs

4. **Memory Issues**
   - Check MAX_FACTS setting
   - Monitor memory usage
   - Enable compression

5. **Performance Issues**
   - Check database indexes
   - Monitor cache hit ratios
   - Adjust batch sizes

### Debug Commands

```bash
# Check configuration
python -c "from config.environment import get_settings; print(get_settings().dict())"

# Test database connection
python -c "from storage.db_utils import get_connection; print(get_connection())"

# Test Redis connection
python -c "import redis; r = redis.from_url('redis://localhost:6379/0'); print(r.ping())"

# Check task queue
python -c "from tasks.task_queue import get_queue_stats; print(get_queue_stats())"
```

## 📈 Scaling

### Horizontal Scaling

1. **API Servers**
   - Deploy multiple API instances
   - Use load balancer
   - Share database and Redis

2. **Celery Workers**
   - Deploy multiple workers
   - Use different queues for different task types
   - Monitor worker performance

3. **Database**
   - Use read replicas
   - Implement connection pooling
   - Consider sharding for very large datasets

### Vertical Scaling

1. **Memory**
   - Increase MAX_FACTS
   - Optimize compression settings
   - Use larger Redis instances

2. **CPU**
   - Increase MAX_CONCURRENT_TASKS
   - Optimize batch processing
   - Use faster embedding models

3. **Storage**
   - Use SSD storage
   - Implement data archiving
   - Optimize database indexes

## 🔐 Security

### Production Security Checklist

- [ ] Change default API_SECURITY_TOKEN
- [ ] Use HTTPS in production
- [ ] Implement proper CORS policy
- [ ] Set up rate limiting
- [ ] Enable input validation
- [ ] Use secure database connections
- [ ] Implement audit logging
- [ ] Regular security updates

### Security Headers

The application automatically sets security headers:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000; includeSubDomains

## 📚 Additional Resources

- [Celery Documentation](https://docs.celeryproject.org/)
- [Redis Documentation](https://redis.io/documentation)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 🆘 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review health check endpoints
3. Check monitoring dashboards
4. Consult this deployment guide
5. Review the test suite in `tests/test_enterprise_features.py` 