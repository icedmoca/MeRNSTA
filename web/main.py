#!/usr/bin/env python3
"""
FastAPI Web Interface for MeRNSTA Multi-Agent Cognitive System

Main application entry point providing live agent chat and debate mode.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_config

# Import routes
from web.routes.agents import router as agents_router
from web.routes.visualizer import router as visualizer_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()
network_config = config.get('network', {})

# Create FastAPI app
app = FastAPI(
    title="MeRNSTA Multi-Agent Cognitive System",
    description="Live agent chat interface with debate mode",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=network_config.get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
web_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(web_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

# Include routers
app.include_router(agents_router, prefix="/agents", tags=["agents"])
app.include_router(visualizer_router, prefix="/visualizer", tags=["visualizer"])

@app.get("/")
async def root():
    """Root endpoint - return JSON metadata for tests and clients."""
    config = get_config()
    visualizer_enabled = config.get('visualizer', {}).get('enable_visualizer', False)
    return {
        "message": "MeRNSTA Multi-Agent System",
        "chat_url": "/chat",
        "visualizer_url": "/visualizer/",
        "visualizer_enabled": visualizer_enabled
    }

@app.get("/chat")
async def chat_interface(request: Request):
    """Main chat interface."""
    try:
        # Get agent registry for UI
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        agent_names = registry.get_agent_names()
        system_status = registry.get_system_status()
        api_port = config.get('network', {}).get('api_port', 8001)
        api_base_url = f"http://127.0.0.1:{api_port}"
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "agent_names": agent_names,
            "system_status": system_status,
            "debate_enabled": system_status.get("debate_mode", True),
            "api_base_url": api_base_url
        })
    except Exception as e:
        logger.error(f"Error loading chat interface: {e}")
        return {"error": "Chat interface unavailable", "details": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        status = registry.get_system_status()
        
        return {
            "status": "healthy",
            "agents": status.get("total_agents", 0),
            "enabled": status.get("enabled", False)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/agents")
async def list_agents():
    """List all available agents and their capabilities."""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        return {
            "agents": registry.get_agent_names(),
            "capabilities": registry.get_agent_capabilities(),
            "system_status": registry.get_system_status()
        }
    except Exception as e:
        return {"error": "Could not load agents", "details": str(e)}

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🤖 Starting MeRNSTA Multi-Agent Cognitive System")
    
    try:
        # Initialize agent registry
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        status = registry.get_system_status()
        
        logger.info(f"✅ Initialized {status['total_agents']} agents: {status['agent_names']}")
        logger.info(f"🧠 Debate mode: {'enabled' if status['debate_mode'] else 'disabled'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("🔄 Shutting down MeRNSTA Multi-Agent System")

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from config
    host = network_config.get('bind_host', '0.0.0.0')
    port = network_config.get('api_port', 8000)
    
    logger.info(f"🚀 Starting server at http://{host}:{port}")
    logger.info(f"💬 Chat interface: http://{host}:{port}/chat")
    
    uvicorn.run(
        "web.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    ) 