# Installation Guide

## Prerequisites

- **Python 3.10+** (required)
- **Git** (for cloning the repository)
- **Redis** (optional, for enterprise features)
- **PostgreSQL** (optional, for enterprise scale)

## Quick Install

### 1. Clone Repository
```bash
git clone https://github.com/icedmoca/mernsta.git
cd mernsta
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Test Installation
```bash
python cortex.py --help
```

### 6. Run MeRNSTA Service:
#### Start:
```bash
./start_mernsta.sh start
```
#### Restart:
```bash
./start_mernsta.sh restart
```
#### Stop:
```bash
./start_mernsta.sh stop
```

## or Docker Installation

### Pull and Run
```bash
docker run -it ghcr.io/icedmoca/mernsta:latest python3 cortex.py
```

### Build from Source
```bash
docker build -t mernsta .
docker run -it mernsta python3 cortex.py
```

## Enterprise Setup

### 1. Install Redis
```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

### 2. Install PostgreSQL (Optional)
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Create database
createdb mernsta
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Start Enterprise Services
```bash
python start_enterprise.py
```

## Configuration

### Basic Configuration
Edit `config.yaml` to customize:
- Database path
- Ollama host URL
- Memory thresholds
- Personality settings

### Environment Variables
Set in `.env` file:
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `API_SECURITY_TOKEN` - Authentication token

## Verification

### Run Tests
```bash
pytest tests/ -v
```

### Start Dashboard
```bash
python demos/memory_dashboard.py
```

### Test API
```bash
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

## Troubleshooting

### Common Issues

**Python version error:**
```bash
python3 --version  # Must be 3.10+
```

**spaCy model missing:**
```bash
python -m spacy download en_core_web_sm
```

**Redis connection error:**
```bash
redis-cli ping  # Should return PONG
```

**Port conflicts:**
- API server will automatically try ports 8000-8005
- Dashboard uses port 8001 by default
- Check `config.yaml` to modify ports

### Performance Optimization

**For better performance:**
1. Use PostgreSQL instead of SQLite for large datasets
2. Enable Redis caching
3. Increase `max_connections` in config
4. Use SSD storage for database files

### Memory Usage

- **SQLite:** ~100MB for 10k facts
- **PostgreSQL:** More efficient for large datasets
- **Redis cache:** ~50MB for embedding cache

## Next Steps

1. Read [Configuration Guide](CONFIGURATION.md)
2. Explore [API Documentation](API.md)
3. Check [Enterprise Features](ENTERPRISE.md)
4. Try the demos in `/demos` directory 