#!/usr/bin/env python3
"""
FastAPI Routes for Agent Interactions

Handles agent responses, debate mode, and agent management endpoints.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response models
class AgentRequest(BaseModel):
    agent: Optional[str] = None
    message: str
    debate: bool = False
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    agent: str
    response: str
    method: Optional[str] = None
    confidence: Optional[float] = None
    debate_results: Optional[list] = None

@router.post("/respond", response_model=AgentResponse)
async def agent_respond(request: AgentRequest):
    """
    Handle agent responses with support for debate mode.
    
    Args:
        request: Agent request with message and configuration
        
    Returns:
        Agent response or debate results
    """
    try:
        # Import here to avoid circular imports
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        if not registry.enabled:
            raise HTTPException(status_code=503, detail="Multi-agent system is disabled")
        
        # Handle debate mode
        if request.debate:
            logger.info(f"[AgentRoutes] Debate mode request: '{request.message[:50]}...'")
            
            debate_results = registry.execute_debate_mode(request.message, request.context)
            
            if not debate_results:
                raise HTTPException(status_code=503, detail="No agents available for debate")
            
            # Format debate response
            formatted_responses = []
            for result in debate_results:
                formatted_responses.append(f"**{result['agent'].title()}Agent**: {result['response']}")
            
            return AgentResponse(
                agent="debate",
                response="\n\n---\n\n".join(formatted_responses),
                debate_results=debate_results
            )
        
        # Handle single agent mode
        if not request.agent:
            # Default to planner if no specific agent requested
            request.agent = "planner"
        
        agent = registry.get_agent(request.agent)
        if not agent:
            available_agents = registry.get_agent_names()
            raise HTTPException(
                status_code=404, 
                detail=f"Agent '{request.agent}' not found. Available: {available_agents}"
            )
        
        logger.info(f"[AgentRoutes] {request.agent}Agent request: '{request.message[:50]}...'")
        
        # Get agent response
        response = agent.respond(request.message, request.context)
        
        # Try to extract additional metadata if available
        method = getattr(agent, 'last_method', None)
        confidence = getattr(agent, 'last_confidence', None)
        
        return AgentResponse(
            agent=request.agent,
            response=response,
            method=method,
            confidence=confidence
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[AgentRoutes] Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status")
async def agent_status():
    """Get system and agent status."""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        return registry.get_system_status()
    except Exception as e:
        logger.error(f"[AgentRoutes] Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def agent_capabilities(agent: Optional[str] = None):
    """Get agent capabilities."""
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        
        return registry.get_agent_capabilities(agent)
    except Exception as e:
        logger.error(f"[AgentRoutes] Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload")
async def reload_agents():
    """Reload all agents (useful for config changes)."""
    try:
        from agents.registry import reload_agent_registry
        reload_agent_registry()
        
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        status = registry.get_system_status()
        
        logger.info(f"[AgentRoutes] Reloaded {status['total_agents']} agents")
        
        return {
            "message": "Agents reloaded successfully",
            "status": status
        }
    except Exception as e:
        logger.error(f"[AgentRoutes] Error reloading agents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 