#!/usr/bin/env python3
"""
🧠 MeRNSTA Memory Graph Visualizer Routes - Phase 34

Web routes for the interactive cognitive visualization system.
Provides endpoints for the D3.js frontend to access memory graph data.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config.settings import get_config

# Lazy import to avoid heavy cognitive system initialization
def get_system_bridge():
    """Lazy import of system bridge to avoid startup delays."""
    try:
        from api.system_bridge import SystemBridgeAPI
        return SystemBridgeAPI()
    except Exception as e:
        logger.warning(f"System Bridge not available: {e}")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()
visualizer_config = config.get('visualizer', {})

# Create router
router = APIRouter()

# Set up templates
web_dir = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(web_dir / "templates"))


@router.get("/", response_class=HTMLResponse)
async def visualizer_home(request: Request):
    """Main visualizer dashboard page."""
    # Check if visualizer is enabled
    if not visualizer_config.get('enable_visualizer', False):
        raise HTTPException(status_code=503, detail="Visualizer not enabled in configuration")
    
    # Get visualizer configuration for the frontend
    api_port = config.get('network', {}).get('api_port', 8001)
    frontend_config = {
        "host": visualizer_config.get('host', '127.0.0.1'),
        "port": visualizer_config.get('port', 8182),
        "update_interval": visualizer_config.get('update_interval', 5),
        "real_time_updates": visualizer_config.get('real_time_updates', True),
        "modules": visualizer_config.get('modules', {}),
        "ui_settings": visualizer_config.get('ui_settings', {}),
        "api_base_url": f"http://127.0.0.1:{api_port}"  # Dynamic API port from config
    }
    
    return templates.TemplateResponse("visualizer/dashboard.html", {
        "request": request,
        "config": frontend_config,
        "title": "MeRNSTA Memory Graph Visualizer"
    })


@router.get("/module/{module_name}", response_class=HTMLResponse)
async def visualizer_module(request: Request, module_name: str):
    """Individual module visualization pages."""
    # Check if visualizer is enabled
    if not visualizer_config.get('enable_visualizer', False):
        raise HTTPException(status_code=503, detail="Visualizer not enabled in configuration")
    
    # Validate module name
    valid_modules = ["contradiction_map", "personality_evolution", "task_flow_dag", "dissonance_heatmap"]
    if module_name not in valid_modules:
        raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found")
    
    # Check if module is enabled
    modules_config = visualizer_config.get('modules', {})
    if not modules_config.get(module_name, True):
        raise HTTPException(status_code=503, detail=f"Module '{module_name}' is disabled")
    
    # Module-specific configuration
    module_configs = {
        "contradiction_map": {
            "title": "Contradiction Map",
            "description": "Interactive visualization of memory contradictions and fact relationships",
            "graph_type": "force_directed",
            "physics_enabled": True,
            "node_types": ["fact", "contradiction"],
            "edge_types": ["contradicts", "supports", "relates_to"]
        },
        "personality_evolution": {
            "title": "Personality Evolution",
            "description": "Timeline visualization of personality trait changes and evolution triggers",
            "graph_type": "timeline",
            "physics_enabled": False,
            "node_types": ["personality_state", "trigger_event"],
            "edge_types": ["evolves_to", "triggered_by"]
        },
        "task_flow_dag": {
            "title": "Task Flow DAG",
            "description": "Directed acyclic graph of plan execution, memory formation, and scoring",
            "graph_type": "hierarchical",
            "physics_enabled": True,
            "node_types": ["plan", "execution", "memory", "score"],
            "edge_types": ["executes", "forms_memory", "generates_score"]
        },
        "dissonance_heatmap": {
            "title": "Dissonance Heatmap",
            "description": "Heat map visualization of cognitive dissonance intensity over time",
            "graph_type": "heatmap",
            "physics_enabled": False,
            "node_types": ["heatmap_cell"],
            "edge_types": []
        }
    }
    
    module_config = module_configs.get(module_name, {})
    api_port = config.get('network', {}).get('api_port', 8001)
    module_config.update({
        "module_name": module_name,
        "api_base_url": f"http://127.0.0.1:{api_port}",
        "update_interval": visualizer_config.get('update_interval', 5),
        "ui_settings": visualizer_config.get('ui_settings', {})
    })
    
    return templates.TemplateResponse("visualizer/module.html", {
        "request": request,
        "module_config": module_config,
        "title": f"MeRNSTA - {module_config.get('title', module_name)}"
    })


@router.get("/events", response_class=HTMLResponse)
async def visualizer_events(request: Request):
    """Real-time cognitive events visualization page."""
    # Check if visualizer is enabled
    if not visualizer_config.get('enable_visualizer', False):
        raise HTTPException(status_code=503, detail="Visualizer not enabled in configuration")
    
    # Check if cognitive events module is enabled
    modules_config = visualizer_config.get('modules', {})
    if not modules_config.get('cognitive_events', True):
        raise HTTPException(status_code=503, detail="Cognitive events module is disabled")
    
    api_port = config.get('network', {}).get('api_port', 8001)
    events_config = {
        "title": "Real-time Cognitive Events",
        "description": "Live stream of cognitive processes: memory formation, contradictions, planning",
        "api_base_url": f"http://127.0.0.1:{api_port}",
        "update_interval": visualizer_config.get('update_interval', 5),
        "real_time_updates": visualizer_config.get('real_time_updates', True),
        "ui_settings": visualizer_config.get('ui_settings', {}),
        "data_retention": visualizer_config.get('data_retention', {})
    }
    
    return templates.TemplateResponse("visualizer/events.html", {
        "request": request,
        "events_config": events_config,
        "title": "MeRNSTA - Real-time Cognitive Events"
    })


@router.get("/api/status")
async def visualizer_api_status():
    """Get visualizer status and configuration."""
    return {
        "enabled": visualizer_config.get('enable_visualizer', False),
        "modules": visualizer_config.get('modules', {}),
        "update_interval": visualizer_config.get('update_interval', 5),
        "real_time_updates": visualizer_config.get('real_time_updates', True),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/config")
async def visualizer_api_config():
    """Get visualizer configuration for frontend."""
    if not visualizer_config.get('enable_visualizer', False):
        raise HTTPException(status_code=503, detail="Visualizer not enabled")
    
    return {
        "visualizer": visualizer_config,
        "api_endpoints": {
            "data": "/visualizer/data",
            "events": "/visualizer/events",
            "status": "/status"
        },
        "frontend_settings": {
            "theme": visualizer_config.get('ui_settings', {}).get('theme', 'dark'),
            "auto_refresh": visualizer_config.get('ui_settings', {}).get('auto_refresh', True),
            "graph_physics": visualizer_config.get('ui_settings', {}).get('graph_physics', True),
            "zoom_levels": visualizer_config.get('ui_settings', {}).get('zoom_levels', [0.1, 0.5, 1.0, 1.5, 2.0, 5.0])
        }
    }