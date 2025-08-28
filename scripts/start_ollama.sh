#!/bin/bash
# MeRNSTA Ollama Startup Script
# Starts the custom Ollama instance from external/ollama directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OLLAMA_DIR="$PROJECT_ROOT/external/ollama"
OLLAMA_BINARY="$OLLAMA_DIR/ollama"
PID_FILE="$PROJECT_ROOT/pids/ollama.pid"
LOG_FILE="$PROJECT_ROOT/logs/ollama.log"

# Default port from config
OLLAMA_PORT=11434
OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT"

echo -e "${BLUE}🧠 MeRNSTA Ollama Startup Script${NC}"
echo -e "${BLUE}================================${NC}"

# Function to check if Ollama is already running
check_ollama_running() {
    if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama is already running on $OLLAMA_HOST${NC}"
        return 0
    else
        return 1
    fi
}

# Function to wait for Ollama to be ready
wait_for_ollama() {
    echo -e "${YELLOW}⏳ Waiting for Ollama to be ready...${NC}"
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama is ready!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}❌ Ollama failed to start within 60 seconds${NC}"
    return 1
}

# Function to stop Ollama
stop_ollama() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}🛑 Stopping Ollama (PID: $pid)...${NC}"
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}🔄 Force killing Ollama...${NC}"
                kill -9 "$pid"
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# Function to start Ollama
start_ollama() {
    echo -e "${BLUE}🚀 Starting Ollama from: $OLLAMA_DIR${NC}"
    
    # Check if Ollama binary exists
    if [ ! -f "$OLLAMA_BINARY" ]; then
        echo -e "${RED}❌ Ollama binary not found at: $OLLAMA_BINARY${NC}"
        echo -e "${YELLOW}💡 Make sure you have built the custom Ollama with tokenizer/detokenizer support${NC}"
        exit 1
    fi
    
    # Check if binary is executable
    if [ ! -x "$OLLAMA_BINARY" ]; then
        echo -e "${YELLOW}🔧 Making Ollama binary executable...${NC}"
        chmod +x "$OLLAMA_BINARY"
    fi
    
    # Create necessary directories
    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Stop any existing Ollama process
    stop_ollama
    
    # Start Ollama in background
    echo -e "${BLUE}🔄 Starting Ollama with ./ollama serve...${NC}"
    cd "$OLLAMA_DIR"
    
    # Start Ollama and capture PID
    nohup ./ollama serve > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    echo -e "${GREEN}✅ Ollama started with PID: $pid${NC}"
    echo -e "${BLUE}📝 Logs: $LOG_FILE${NC}"
    echo -e "${BLUE}🔗 API: $OLLAMA_HOST${NC}"
    
    # Wait for Ollama to be ready
    if wait_for_ollama; then
        echo -e "${GREEN}🎉 Ollama is ready for use!${NC}"
        
        # Test tokenizer endpoints
        echo -e "${BLUE}🧪 Testing tokenizer endpoints...${NC}"
        if curl -s -X POST "$OLLAMA_HOST/api/tokenize" \
            -H "Content-Type: application/json" \
            -d '{"model": "tinyllama", "content": "test"}' > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Tokenizer endpoint working${NC}"
        else
            echo -e "${YELLOW}⚠️ Tokenizer endpoint test failed (this may be normal if no models are loaded)${NC}"
        fi
        
        if curl -s -X POST "$OLLAMA_HOST/api/detokenize" \
            -H "Content-Type: application/json" \
            -d '{"model": "tinyllama", "tokens": [1, 2, 3]}' > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Detokenizer endpoint working${NC}"
        else
            echo -e "${YELLOW}⚠️ Detokenizer endpoint test failed (this may be normal if no models are loaded)${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}❌ Failed to start Ollama${NC}"
        echo -e "${YELLOW}📝 Check logs: $LOG_FILE${NC}"
        return 1
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Ollama Status${NC}"
    echo -e "${BLUE}===============${NC}"
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✅ Ollama is running (PID: $pid)${NC}"
        else
            echo -e "${RED}❌ Ollama PID file exists but process is not running${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${YELLOW}⚠️ No PID file found${NC}"
    fi
    
    if check_ollama_running; then
        echo -e "${GREEN}✅ Ollama API is responding${NC}"
        echo -e "${BLUE}🔗 API URL: $OLLAMA_HOST${NC}"
    else
        echo -e "${RED}❌ Ollama API is not responding${NC}"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}📝 Ollama Logs (last 20 lines)${NC}"
        echo -e "${BLUE}============================${NC}"
        tail -n 20 "$LOG_FILE"
    else
        echo -e "${YELLOW}⚠️ No log file found${NC}"
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        if check_ollama_running; then
            echo -e "${GREEN}✅ Ollama is already running${NC}"
            exit 0
        fi
        start_ollama
        ;;
    stop)
        stop_ollama
        echo -e "${GREEN}✅ Ollama stopped${NC}"
        ;;
    restart)
        stop_ollama
        sleep 2
        start_ollama
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    check)
        if check_ollama_running; then
            echo -e "${GREEN}✅ Ollama is running and responding${NC}"
            exit 0
        else
            echo -e "${RED}❌ Ollama is not running or not responding${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${BLUE}Usage: $0 {start|stop|restart|status|logs|check}${NC}"
        echo -e "${BLUE}Commands:${NC}"
        echo -e "  start   - Start Ollama from external/ollama"
        echo -e "  stop    - Stop Ollama"
        echo -e "  restart - Restart Ollama"
        echo -e "  status  - Show Ollama status"
        echo -e "  logs    - Show recent logs"
        echo -e "  check   - Check if Ollama is running (exit code 0/1)"
        exit 1
        ;;
esac
