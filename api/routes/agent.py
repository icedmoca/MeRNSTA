#!/usr/bin/env python3
"""
Legacy compatibility for agent-related tests that import api.routes.agent
Provides minimal hooks used in tests.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

# Expose a minimal /respond for tests that might call it
@router.post("/agents/respond")
async def agent_respond(message: str):
    try:
        from agents.registry import get_agent_registry
        registry = get_agent_registry()
        agent = registry.get_agent("planner")
        resp = agent.respond(message) if agent else "[PlannerAgent] Agent unavailable"
        return JSONResponse({"agent": "planner", "response": resp})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

