# API Documentation

MeRNSTA provides a comprehensive REST API for memory operations, agent interactions, and system monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

All protected endpoints require Bearer token authentication:

```bash
Authorization: Bearer <your-token>
```

Set your token in `.env`:
```bash
API_SECURITY_TOKEN=your-secure-token-here
```

## Memory API

### Add Memory
Store text in memory and extract triplets.

```http
POST /api/memory/add
Content-Type: application/json
Authorization: Bearer <token>

{
  "text": "I love pizza"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Stored 1 triplets",
  "fact_ids": [42]
}
```

### Get Facts
Retrieve stored facts with optional limit.

```http
GET /api/memory/facts?limit=10
Authorization: Bearer <token>
```

**Response:**
```json
{
  "facts": [
    {
      "id": 1,
      "subject": "I",
      "predicate": "love",
      "object": "pizza",
      "confidence": 0.95
    }
  ]
}
```

### Search Triplets
Semantic search for relevant triplets.

```http
GET /api/memory/search_triplets?query=what do I like&top_k=5
Authorization: Bearer <token>
```

**Response:**
```json
{
  "query": "what do I like",
  "triplets": [
    {
      "id": 1,
      "subject": "I",
      "predicate": "love",
      "object": "pizza",
      "confidence": 0.95,
      "similarity": 0.87
    }
  ],
  "count": 1
}
```

### Get Contradictions
List contradictions with optional filtering.

```http
GET /api/memory/contradictions?resolved=false
Authorization: Bearer <token>
```

**Response:**
```json
{
  "contradictions": [
    {
      "id": 1,
      "fact_a_text": "I love pizza",
      "fact_b_text": "I hate pizza",
      "confidence": 0.92,
      "resolved": false
    }
  ]
}
```

### Get Memory Report
Comprehensive memory system analysis.

```http
GET /api/memory/memory_report
Authorization: Bearer <token>
```

**Response:**
```json
{
  "report": {
    "total_facts": 150,
    "high_confidence_facts": 120,
    "contradictions": 5,
    "subjects": ["I", "user", "system"],
    "memory_health": "good"
  }
}
```

### Upload Media
Store image or audio with extracted metadata.

```http
POST /api/memory/upload_file
Content-Type: multipart/form-data
Authorization: Bearer <token>

media_type=image
file=<image_file>
description=Optional description
```

**Response:**
```json
{
  "success": true,
  "fact_id": 42,
  "message": "Image processed and stored"
}
```

## Agent API

### Get Context
Retrieve relevant context for agent operations.

```http
GET /api/agent/context?goal=What does the user like?
Authorization: Bearer <token>
```

**Response:**
```json
{
  "goal": "What does the user like?",
  "context": [
    {
      "subject": "user",
      "predicate": "likes",
      "object": "pizza",
      "confidence": 0.95,
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "count": 1
}
```

### Agent Reflection
Process agent task results and update memory.

```http
POST /api/agent/reflect
Content-Type: application/json
Authorization: Bearer <token>

{
  "task": "Find user preferences",
  "result": "User prefers Italian food"
}
```

**Response:**
```json
{
  "reflection": "Noted user's preference for Italian cuisine",
  "stored_facts": 1,
  "confidence": 0.85
}
```

### Get Trust Score
Retrieve trust score for a subject.

```http
GET /api/agent/trust_score/user
Authorization: Bearer <token>
```

**Response:**
```json
{
  "subject": "user",
  "trust_score": 0.92,
  "fact_count": 25,
  "contradiction_count": 2,
  "last_updated": "2024-01-01T12:00:00"
}
```

### Test Pattern Extraction
Test fact extraction on sample text.

```http
GET /api/agent/test_pattern?text=I enjoy coding
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "input_text": "I enjoy coding",
  "results": {
    "extracted_triplets": [
      ["I", "enjoy", "coding", 0.9]
    ],
    "patterns_matched": ["preference_pattern"],
    "confidence_scores": [0.9]
  }
}
```

## Chat API

### Chat Endpoint
Interactive chat with memory integration.

```http
POST /chat
Content-Type: application/json
Authorization: Bearer <token>

{
  "message": "What do I like?",
  "personality": "neutral"
}
```

**Response:**
```json
{
  "response": "Based on what I remember, you like pizza and Italian food.",
  "personality": "neutral",
  "facts_used": 2,
  "confidence": 0.89
}
```

## Dashboard API

### Get Dashboard Facts
Paginated facts for dashboard display.

```http
GET /dashboard/facts?page=1
```

**Response:**
```json
{
  "facts": [
    {
      "id": 1,
      "subject": "I",
      "predicate": "love", 
      "object": "pizza",
      "media_type": "text",
      "confidence": 0.95,
      "contradiction_score": 0.0,
      "volatility_score": 0.1
    }
  ]
}
```

### Get Dashboard Metrics
Real-time system metrics.

```http
GET /dashboard/metrics?minimal=true
```

**Response:**
```json
{
  "fact_count": 150,
  "contradiction_rate": 0.03,
  "memory_usage_mb": 45.2
}
```

### Get Clusters
Memory clustering information.

```http
GET /dashboard/clusters
```

**Response:**
```json
{
  "clusters": [
    {
      "id": 1,
      "subject": "food preferences",
      "cluster_size": 8,
      "trust_score": 0.92,
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

## Health & Monitoring

### Health Check
System health status (no auth required).

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "response_time_ms": 23.5,
  "checks": {
    "database": {"status": "healthy"},
    "memory": {"status": "healthy"},
    "llm": {"status": "healthy"}
  },
  "version": "1.0.0"
}
```

### Detailed Health Check
Comprehensive health report.

```http
GET /health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "health_percentage": 95.0,
  "checks": {
    "database": {
      "status": "healthy",
      "fact_count": 150,
      "response_time_ms": 5.2
    },
    "memory": {
      "status": "healthy",
      "usage_mb": 45.2,
      "cache_hit_ratio": 0.85
    }
  },
  "performance": {
    "response_time_ms": 23.5,
    "memory_usage_mb": 128.4,
    "cpu_percent": 15.2
  }
}
```

### Prometheus Metrics
Metrics in Prometheus format.

```http
GET /metrics
```

**Response:**
```
# HELP memory_operations_total Total memory operations
# TYPE memory_operations_total counter
memory_operations_total{operation="store",status="success"} 1250

# HELP api_requests_total Total API requests  
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/api/memory/facts",status_code="200"} 45
```

## Error Responses

### Authentication Error
```json
{
  "detail": "Invalid authentication credentials",
  "status_code": 401
}
```

### Validation Error
```json
{
  "detail": "Invalid input format",
  "status_code": 400,
  "errors": ["text field is required"]
}
```

### Rate Limit Error
```json
{
  "detail": "Rate limit exceeded",
  "status_code": 429,
  "retry_after": 60
}
```

### Server Error
```json
{
  "detail": "Internal server error",
  "status_code": 500,
  "error_id": "uuid-error-id"
}
```

## Rate Limiting

Default limits:
- **100 requests per minute** per IP
- **Configurable** via `RATE_LIMIT` environment variable
- **Disabled** in test environments

## SDK Examples

### Python SDK
```python
import requests

class MeRNSTAClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def add_memory(self, text):
        response = requests.post(
            f"{self.base_url}/api/memory/add",
            json={"text": text},
            headers=self.headers
        )
        return response.json()
    
    def search(self, query, top_k=5):
        response = requests.get(
            f"{self.base_url}/api/memory/search_triplets",
            params={"query": query, "top_k": top_k},
            headers=self.headers
        )
        return response.json()

# Usage
client = MeRNSTAClient("http://localhost:8000", "your-token")
result = client.add_memory("I love programming")
```

### JavaScript SDK
```javascript
class MeRNSTAClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async addMemory(text) {
    const response = await fetch(`${this.baseUrl}/api/memory/add`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ text })
    });
    return response.json();
  }

  async search(query, topK = 5) {
    const url = new URL(`${this.baseUrl}/api/memory/search_triplets`);
    url.searchParams.set('query', query);
    url.searchParams.set('top_k', topK);
    
    const response = await fetch(url, { headers: this.headers });
    return response.json();
  }
}

// Usage
const client = new MeRNSTAClient('http://localhost:8000', 'your-token');
const result = await client.addMemory('I love JavaScript');
```

## OpenAPI Specification

Interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

## WebSocket Support

Real-time memory updates (planned):

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/memory');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Memory update:', update);
};
```

## Best Practices

1. **Always handle errors** appropriately
2. **Use pagination** for large result sets
3. **Cache frequently used** data
4. **Monitor rate limits** to avoid throttling
5. **Use semantic search** for better relevance
6. **Include context** in agent operations
7. **Validate input** before sending requests

## See Also

- [Installation Guide](INSTALLATION.md)
- [Configuration Reference](CONFIGURATION.md)
- [Enterprise Features](ENTERPRISE.md) 