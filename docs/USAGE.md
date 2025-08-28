# Usage Examples

### **🚀 Full AGI Mode (Recommended)**
```bash
# Start everything at once
python main.py run

# Start with custom ports
python main.py run --web-port 8080 --api-port 8081

# Start without web interface (headless)
python main.py run --no-web

# Start with enterprise features
python main.py run --enterprise
```

**What you get:**
- **💬 Web Chat Interface** at http://localhost:8000/chat
- **🔌 REST API Server** at http://localhost:8001/docs
- **🤖 All 23 Cognitive Agents** running simultaneously
- **🔄 Background Tasks** (reflection, planning, memory consolidation)
- **🧠 Full Autonomous AGI** system ready to use

### **Web Chat Interface Only**
```bash
# Start just the web interface
python main.py web

# Open: http://localhost:8000/chat
# - Single Agent mode: Choose specific agent (planner, critic, etc.)
# - Debate Mode: Enable to hear from all agents simultaneously
```

### **API Integration Only**
```bash
# Start just the API server
python main.py api

# Test agent response
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze this system"}'

# Check system health
curl http://localhost:8001/health
```

### **Background Tasks Only**
```bash
# Start just background autonomous processes
python main.py integration --integration-mode daemon

# Features automatically running:
# - Memory consolidation
# - Drift prediction & execution  
# - Reflection cycles
# - Self-healing
# - Planning & goal generation
```

