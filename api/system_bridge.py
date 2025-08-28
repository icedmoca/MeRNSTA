#!/usr/bin/env python3
"""
🌉 MeRNSTA System Bridge API - Phase 30
Local FastAPI HTTP API server that provides OS integration endpoints.

This module provides REST API endpoints for external applications to interact
with the MeRNSTA cognitive system as a microservice:
- /ask - Query the cognitive system
- /memory - Access memory operations  
- /goal - Goal management
- /reflect - Trigger reflection
- /personality - Personality information
- /status - System status and health
"""

import asyncio
import os
import sys
import time
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem


# Pydantic models for API requests/responses
class AskRequest(BaseModel):
    query: str = Field(..., description="Question or statement to process")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class AskResponse(BaseModel):
    response: str = Field(..., description="The cognitive system's response")
    confidence: Optional[float] = Field(None, description="Response confidence score")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the response")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")


class MemoryQuery(BaseModel):
    query_type: str = Field(..., description="Type of memory query (search, recent, facts, contradictions)")
    query: Optional[str] = Field(None, description="Search query for memory lookup")
    limit: Optional[int] = Field(10, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class MemoryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Memory query results")
    total_count: Optional[int] = Field(None, description="Total number of matching items")
    query_info: Dict[str, Any] = Field(..., description="Information about the query")


class GoalRequest(BaseModel):
    action: str = Field(..., description="Action: add, remove, list, update")
    goal_text: Optional[str] = Field(None, description="Goal description for add/update")
    goal_id: Optional[str] = Field(None, description="Goal ID for remove/update")
    priority: Optional[float] = Field(None, description="Goal priority (0-1)")


class GoalResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Status message")
    goals: Optional[List[Dict[str, Any]]] = Field(None, description="Current goals list")


class ReflectionRequest(BaseModel):
    trigger_type: str = Field("manual", description="Type of reflection trigger")
    focus_area: Optional[str] = Field(None, description="Specific area to focus reflection on")


class ReflectionResponse(BaseModel):
    success: bool = Field(..., description="Whether reflection completed successfully")
    insights: Optional[List[str]] = Field(None, description="Key insights from reflection")
    duration_seconds: Optional[float] = Field(None, description="Time taken for reflection")
    timestamp: str = Field(..., description="Reflection completion timestamp")


class PersonalityResponse(BaseModel):
    traits: Dict[str, Any] = Field(..., description="Current personality traits")
    evolution_history: Optional[List[Dict[str, Any]]] = Field(None, description="Personality evolution history")
    stability_metrics: Optional[Dict[str, float]] = Field(None, description="Personality stability metrics")


class StatusResponse(BaseModel):
    system_status: str = Field(..., description="Overall system status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    cognitive_system_active: bool = Field(..., description="Whether cognitive system is active")
    background_tasks: Dict[str, Any] = Field(..., description="Background task status")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    last_activity: str = Field(..., description="Timestamp of last activity")


# === VISUALIZER API MODELS ===

class VisualizerDataRequest(BaseModel):
    module: str = Field(..., description="Visualizer module: contradiction_map, personality_evolution, task_flow_dag, dissonance_heatmap, cognitive_events")
    time_range: Optional[str] = Field("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, all")
    limit: Optional[int] = Field(100, description="Maximum number of items to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class GraphNode(BaseModel):
    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Node display label")
    type: str = Field(..., description="Node type")
    data: Dict[str, Any] = Field(..., description="Node data")
    position: Optional[Dict[str, float]] = Field(None, description="Node position (x, y)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GraphEdge(BaseModel):
    id: str = Field(..., description="Edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None, description="Edge label")
    weight: Optional[float] = Field(None, description="Edge weight/strength")
    data: Optional[Dict[str, Any]] = Field(None, description="Edge data")


class VisualizerDataResponse(BaseModel):
    module: str = Field(..., description="Requested module")
    timestamp: str = Field(..., description="Response timestamp")
    nodes: List[GraphNode] = Field(..., description="Graph nodes")
    edges: List[GraphEdge] = Field(..., description="Graph edges")
    metadata: Dict[str, Any] = Field(..., description="Graph metadata")
    update_count: Optional[int] = Field(None, description="Update counter for real-time sync")


class CognitiveEventResponse(BaseModel):
    events: List[Dict[str, Any]] = Field(..., description="Recent cognitive events")
    real_time_data: Dict[str, Any] = Field(..., description="Real-time cognitive state")
    timestamp: str = Field(..., description="Response timestamp")


# Global cognitive system instance
cognitive_system: Optional[Phase2AutonomousCognitiveSystem] = None
app_start_time = datetime.now()


class SystemBridgeAPI:
    """Main API class that manages the FastAPI application and cognitive system."""
    
    def __init__(self):
        self.config = get_config()
        self.os_config = self.config.get('os_integration', {})
        self.api_config = self.os_config.get('api', {})
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="MeRNSTA System Bridge API",
            description="OS Integration API for MeRNSTA Cognitive System",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize logging
        self._setup_logging()
        
        self.logger = logging.getLogger('mernsta.api')
        self.logger.info("SystemBridgeAPI initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware for local access
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Rate limiting middleware (simple implementation)
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Simple rate limiting based on config
            max_requests = self.os_config.get('security', {}).get('max_requests_per_minute', 100)
            
            # For now, just log the request
            self.logger.debug(f"API request: {request.method} {request.url.path}")
            
            response = await call_next(request)
            return response
    
    def _setup_logging(self):
        """Setup logging for the API."""
        log_config = self.os_config.get('logging', {})
        
        if log_config.get('enabled', True):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            logger = logging.getLogger('mernsta.api')
            logger.setLevel(getattr(logging, log_config.get('log_level', 'INFO')))
            logger.addHandler(handler)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize cognitive system on startup."""
            await self._initialize_cognitive_system()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            self.logger.info("Shutting down API server...")
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "MeRNSTA System Bridge API",
                "version": "1.0.0",
                "status": "active",
                "endpoints": ["/ask", "/memory", "/goal", "/reflect", "/personality", "/status", "/visualizer/data", "/visualizer/events"]
            }
        
        @self.app.post("/ask", response_model=AskResponse)
        async def ask_endpoint(request: AskRequest):
            """Ask a question to the cognitive system."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                start_time = time.time()
                
                # Process the query through the cognitive system
                result = cognitive_system.process_input_with_full_cognition(
                    request.query, 
                    user_profile_id=request.user_id,
                    session_id=request.session_id
                )
                
                processing_time = time.time() - start_time
                
                # Extract response information
                response_text = result.get('response', 'No response generated')
                confidence = result.get('confidence')
                reasoning = result.get('reasoning')
                
                return AskResponse(
                    response=response_text,
                    confidence=confidence,
                    reasoning=reasoning,
                    timestamp=datetime.now().isoformat(),
                    session_id=request.session_id or result.get('session_id')
                )
                
            except Exception as e:
                self.logger.error(f"Error processing ask request: {e}")
                raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
        @self.app.post("/memory", response_model=MemoryResponse)
        async def memory_endpoint(request: MemoryQuery):
            """Query memory system."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                query_info = {"type": request.query_type, "timestamp": datetime.now().isoformat()}
                
                if request.query_type == "search":
                    if not request.query:
                        raise HTTPException(status_code=400, detail="Query required for search")
                    
                    # Perform memory search
                    results = cognitive_system.search_memories(
                        request.query, 
                        limit=request.limit or 10
                    )
                    
                elif request.query_type == "recent":
                    # Get recent memories
                    results = cognitive_system.get_recent_memories(limit=request.limit or 10)
                    
                elif request.query_type == "facts":
                    # Get facts from memory
                    results = cognitive_system.get_memory_facts(limit=request.limit or 10)
                    
                elif request.query_type == "contradictions":
                    # Get contradictions
                    results = cognitive_system.get_contradictions(limit=request.limit or 10)
                    
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown query type: {request.query_type}")
                
                # Ensure results is a list of dicts
                if not isinstance(results, list):
                    results = [results] if results else []
                
                # Convert results to JSON-serializable format
                serializable_results = []
                for result in results:
                    if hasattr(result, '__dict__'):
                        serializable_results.append(result.__dict__)
                    elif isinstance(result, dict):
                        serializable_results.append(result)
                    else:
                        serializable_results.append({"item": str(result)})
                
                return MemoryResponse(
                    results=serializable_results,
                    total_count=len(serializable_results),
                    query_info=query_info
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error processing memory request: {e}")
                raise HTTPException(status_code=500, detail=f"Memory query error: {str(e)}")

        # === Multimodal memory compatibility routes (for tests) ===
        from fastapi import UploadFile, File, Form
        from fastapi.responses import JSONResponse as _JSONResponse

        @self.app.post("/memory/upload_media")
        async def upload_media(media_type: str = Form(...), description: str = Form(""), file: UploadFile = File(...)):
            try:
                content = await file.read()
                from storage.memory_log import MemoryLog
                log = MemoryLog()
                # Build a simple triplet for the media
                subject = "user"
                predicate = "uploaded"
                obj = description or file.filename
                triplet = (subject, predicate, obj, 0.9, {"description": description}, media_type, content)
                ids, _ = log.store_triplets([triplet])
                return _JSONResponse({"success": True, "ids": ids})
            except Exception as e:
                self.logger.error(f"Upload media error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/memory/search_multimodal")
        async def search_multimodal(query: str, media_type: str = "text", topk: int = 5):
            try:
                from storage.memory_log import MemoryLog
                log = MemoryLog()
                results = log.semantic_search(query, topk=topk, media_type=media_type)
                return _JSONResponse({"results": results})
            except Exception as e:
                self.logger.error(f"Search multimodal error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Compatibility endpoint: append a memory entry (no hardcoding, uses configured DB)
        from pydantic import BaseModel
        from typing import List, Optional as _Optional

        class AppendMemoryRequest(BaseModel):
            content: str
            role: _Optional[str] = "user"
            tags: _Optional[List[str]] = None

        class AppendMemoryResponse(BaseModel):
            success: bool
            memory_id: _Optional[int] = None
            error: _Optional[str] = None

        @self.app.post("/memory/append", response_model=AppendMemoryResponse)
        async def append_memory_endpoint(request: AppendMemoryRequest):
            """Append a new memory entry and return its ID."""
            try:
                from storage.memory_log import MemoryLog
                memory_log = MemoryLog()
                memory_id = memory_log.log_memory(request.role or "user", request.content, tags=request.tags)
                return AppendMemoryResponse(success=True, memory_id=memory_id)
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error appending memory: {e}")
                return AppendMemoryResponse(success=False, error=str(e))
        
        @self.app.post("/goal", response_model=GoalResponse)
        async def goal_endpoint(request: GoalRequest):
            """Manage goals."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                if request.action == "list":
                    # List current goals
                    goals = cognitive_system.get_current_goals()
                    return GoalResponse(
                        success=True,
                        message="Goals retrieved successfully",
                        goals=goals if isinstance(goals, list) else []
                    )
                
                elif request.action == "add":
                    if not request.goal_text:
                        raise HTTPException(status_code=400, detail="Goal text required for add")
                    
                    # Add new goal
                    goal_id = cognitive_system.add_goal(
                        request.goal_text, 
                        priority=request.priority or 0.5
                    )
                    
                    return GoalResponse(
                        success=True,
                        message=f"Goal added with ID: {goal_id}",
                        goals=cognitive_system.get_current_goals()
                    )
                
                elif request.action == "remove":
                    if not request.goal_id:
                        raise HTTPException(status_code=400, detail="Goal ID required for remove")
                    
                    # Remove goal
                    success = cognitive_system.remove_goal(request.goal_id)
                    
                    return GoalResponse(
                        success=success,
                        message="Goal removed successfully" if success else "Goal not found",
                        goals=cognitive_system.get_current_goals()
                    )
                
                elif request.action == "update":
                    if not request.goal_id:
                        raise HTTPException(status_code=400, detail="Goal ID required for update")
                    
                    # Update goal
                    success = cognitive_system.update_goal(
                        request.goal_id,
                        text=request.goal_text,
                        priority=request.priority
                    )
                    
                    return GoalResponse(
                        success=success,
                        message="Goal updated successfully" if success else "Goal not found",
                        goals=cognitive_system.get_current_goals()
                    )
                
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error processing goal request: {e}")
                raise HTTPException(status_code=500, detail=f"Goal management error: {str(e)}")
        
        @self.app.post("/reflect", response_model=ReflectionResponse)
        async def reflect_endpoint(request: ReflectionRequest):
            """Trigger reflection process."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                start_time = time.time()
                
                # Trigger reflection
                reflection_result = cognitive_system.trigger_autonomous_reflection()
                
                duration = time.time() - start_time
                
                # Extract insights from reflection result
                insights = []
                if isinstance(reflection_result, dict):
                    insights = reflection_result.get('insights', [])
                elif isinstance(reflection_result, list):
                    insights = reflection_result
                
                return ReflectionResponse(
                    success=reflection_result is not None,
                    insights=insights,
                    duration_seconds=duration,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.logger.error(f"Error during reflection: {e}")
                raise HTTPException(status_code=500, detail=f"Reflection error: {str(e)}")
        
        @self.app.get("/personality", response_model=PersonalityResponse)
        async def personality_endpoint():
            """Get personality information."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                # Get personality information
                personality_data = cognitive_system.get_personality_profile()
                
                traits = personality_data.get('traits', {}) if isinstance(personality_data, dict) else {}
                evolution_history = personality_data.get('evolution_history', []) if isinstance(personality_data, dict) else []
                stability_metrics = personality_data.get('stability_metrics', {}) if isinstance(personality_data, dict) else {}
                
                return PersonalityResponse(
                    traits=traits,
                    evolution_history=evolution_history,
                    stability_metrics=stability_metrics
                )
                
            except Exception as e:
                self.logger.error(f"Error getting personality: {e}")
                raise HTTPException(status_code=500, detail=f"Personality error: {str(e)}")

        # Agent respond endpoint (API parity with web '/agents/respond')
        class AgentRequest(BaseModel):
            agent: _Optional[str] = None
            message: str
            debate: bool = False
            context: _Optional[Dict[str, Any]] = None

        class AgentResponse(BaseModel):
            agent: str
            response: str
            method: _Optional[str] = None
            confidence: _Optional[float] = None
            debate_results: _Optional[list] = None

        @self.app.post("/agents/respond", response_model=AgentResponse)
        async def agent_respond_endpoint(request: AgentRequest):
            """Handle agent responses, including debate mode."""
            try:
                from agents.registry import get_agent_registry
                registry = get_agent_registry()

                if not registry.enabled:
                    raise HTTPException(status_code=503, detail="Multi-agent system is disabled")

                if request.debate:
                    debate_results = registry.execute_debate_mode(request.message, request.context)
                    if not debate_results:
                        raise HTTPException(status_code=503, detail="No agents available for debate")
                    formatted = []
                    for result in debate_results:
                        formatted.append(f"**{result['agent'].title()}Agent**: {result['response']}")
                    return AgentResponse(agent="debate", response="\n\n---\n\n".join(formatted), debate_results=debate_results)

                # single-agent mode
                agent_name = request.agent or "planner"
                agent = registry.get_agent(agent_name)
                if not agent:
                    available = registry.get_agent_names()
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. Available: {available}")

                resp_text = agent.respond(request.message, request.context)
                method = getattr(agent, 'last_method', None)
                confidence = getattr(agent, 'last_confidence', None)
                return AgentResponse(agent=agent_name, response=resp_text, method=method, confidence=confidence)
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Agent respond error: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/status", response_model=StatusResponse)
        async def status_endpoint():
            """Get system status."""
            try:
                uptime = (datetime.now() - app_start_time).total_seconds()
                
                # Get background task status (placeholder)
                background_tasks = {
                    "reflection": "active",
                    "planning": "active", 
                    "memory_consolidation": "active",
                    "context_detection": "active"
                }
                
                # Get performance metrics
                performance_metrics = {
                    "memory_usage_mb": 0,  # Placeholder
                    "cpu_usage_percent": 0,  # Placeholder
                    "active_sessions": 0,  # Placeholder
                    "total_queries_processed": 0  # Placeholder
                }
                
                return StatusResponse(
                    system_status="healthy" if cognitive_system else "degraded",
                    uptime_seconds=uptime,
                    cognitive_system_active=cognitive_system is not None,
                    background_tasks=background_tasks,
                    performance_metrics=performance_metrics,
                    last_activity=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")
        
        # === VISUALIZER ENDPOINTS ===
        
        @self.app.post("/visualizer/data", response_model=VisualizerDataResponse)
        async def visualizer_data_endpoint(request: VisualizerDataRequest):
            """Get visualization data for specific modules."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                # Get visualizer config to check if enabled
                visualizer_config = self.config.get('visualizer', {})
                if not visualizer_config.get('enable_visualizer', False):
                    raise HTTPException(status_code=503, detail="Visualizer not enabled")
                
                # Generate data based on module type
                if request.module == "contradiction_map":
                    nodes, edges, metadata = await self._get_contradiction_map_data(request.time_range, request.limit, request.filters)
                elif request.module == "personality_evolution":
                    nodes, edges, metadata = await self._get_personality_evolution_data(request.time_range, request.limit, request.filters)
                elif request.module == "task_flow_dag":
                    nodes, edges, metadata = await self._get_task_flow_dag_data(request.time_range, request.limit, request.filters)
                elif request.module == "dissonance_heatmap":
                    nodes, edges, metadata = await self._get_dissonance_heatmap_data(request.time_range, request.limit, request.filters)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown module: {request.module}")
                
                return VisualizerDataResponse(
                    module=request.module,
                    timestamp=datetime.now().isoformat(),
                    nodes=nodes,
                    edges=edges,
                    metadata=metadata,
                    update_count=int(time.time())  # Simple update counter
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting visualizer data: {e}")
                raise HTTPException(status_code=500, detail=f"Visualizer error: {str(e)}")
        
        @self.app.get("/visualizer/events", response_model=CognitiveEventResponse)
        async def visualizer_events_endpoint():
            """Get real-time cognitive events for visualization."""
            if not cognitive_system:
                raise HTTPException(status_code=503, detail="Cognitive system not available")
            
            try:
                # Get visualizer config to check if enabled
                visualizer_config = self.config.get('visualizer', {})
                if not visualizer_config.get('enable_visualizer', False):
                    raise HTTPException(status_code=503, detail="Visualizer not enabled")
                
                # Get recent cognitive events
                events = await self._get_recent_cognitive_events()
                real_time_data = await self._get_real_time_cognitive_state()
                
                return CognitiveEventResponse(
                    events=events,
                    real_time_data=real_time_data,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.logger.error(f"Error getting cognitive events: {e}")
                raise HTTPException(status_code=500, detail=f"Cognitive events error: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            return {
                "status": "healthy" if cognitive_system else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0"
            }
    
    async def _initialize_cognitive_system(self):
        """Initialize the cognitive system."""
        global cognitive_system
        
        try:
            self.logger.info("Initializing cognitive system...")
            cognitive_system = Phase2AutonomousCognitiveSystem()
            self.logger.info("Cognitive system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive system: {e}")
            cognitive_system = None
    
    # === VISUALIZER DATA METHODS ===
    
    async def _get_contradiction_map_data(self, time_range: str, limit: int, filters: Optional[Dict[str, Any]]):
        """Generate contradiction map data for visualization."""
        try:
            # Get contradictions from the cognitive system
            contradictions = cognitive_system.get_contradictions(limit=limit) if cognitive_system else []
            
            nodes = []
            edges = []
            
            # Create nodes for facts and their contradictions  
            fact_nodes = {}
            for i, contradiction in enumerate(contradictions):
                fact1_id = f"fact_{contradiction.get('fact1_id', i)}"
                fact2_id = f"fact_{contradiction.get('fact2_id', i + 1000)}"
                
                # Add fact nodes if not already added
                if fact1_id not in fact_nodes:
                    nodes.append(GraphNode(
                        id=fact1_id,
                        label=contradiction.get('fact1_text', f"Fact {i}"),
                        type="fact",
                        data={
                            "confidence": contradiction.get('fact1_confidence', 0.5),
                            "timestamp": contradiction.get('fact1_timestamp', ''),
                            "volatility": contradiction.get('fact1_volatility', 0.0)
                        }
                    ))
                    fact_nodes[fact1_id] = True
                
                if fact2_id not in fact_nodes:
                    nodes.append(GraphNode(
                        id=fact2_id,
                        label=contradiction.get('fact2_text', f"Fact {i+1}"),
                        type="fact",
                        data={
                            "confidence": contradiction.get('fact2_confidence', 0.5),
                            "timestamp": contradiction.get('fact2_timestamp', ''),
                            "volatility": contradiction.get('fact2_volatility', 0.0)
                        }
                    ))
                    fact_nodes[fact2_id] = True
                
                # Add contradiction edge
                edges.append(GraphEdge(
                    id=f"contradiction_{i}",
                    source=fact1_id,
                    target=fact2_id,
                    label="contradicts",
                    weight=contradiction.get('score', 0.5),
                    data={
                        "contradiction_type": contradiction.get('type', 'semantic'),
                        "strength": contradiction.get('score', 0.5),
                        "discovered_at": contradiction.get('timestamp', '')
                    }
                ))
            
            metadata = {
                "total_contradictions": len(contradictions),
                "time_range": time_range,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "last_updated": datetime.now().isoformat()
            }
            
            return nodes, edges, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating contradiction map data: {e}")
            return [], [], {"error": str(e)}
    
    async def _get_personality_evolution_data(self, time_range: str, limit: int, filters: Optional[Dict[str, Any]]):
        """Generate personality evolution graph data."""
        try:
            # Get personality data from cognitive system
            personality_data = cognitive_system.get_personality_profile() if cognitive_system else {}
            evolution_history = personality_data.get('evolution_history', [])
            
            nodes = []
            edges = []
            
            # Create timeline nodes for personality states
            for i, snapshot in enumerate(evolution_history[-limit:]):
                node_id = f"personality_{i}"
                nodes.append(GraphNode(
                    id=node_id,
                    label=f"State {i}",
                    type="personality_state",
                    data={
                        "traits": snapshot.get('traits', {}),
                        "timestamp": snapshot.get('timestamp', ''),
                        "triggers": snapshot.get('triggers', []),
                        "stability_score": snapshot.get('stability_score', 0.5)
                    }
                ))
                
                # Add evolution edge to next state
                if i > 0:
                    edges.append(GraphEdge(
                        id=f"evolution_{i-1}_{i}",
                        source=f"personality_{i-1}",
                        target=node_id,
                        label="evolves_to",
                        weight=1.0,
                        data={
                            "changes": snapshot.get('changes', {}),
                            "triggers": snapshot.get('triggers', [])
                        }
                    ))
            
            metadata = {
                "evolution_steps": len(evolution_history),
                "current_traits": personality_data.get('traits', {}),
                "stability_metrics": personality_data.get('stability_metrics', {}),
                "time_range": time_range
            }
            
            return nodes, edges, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating personality evolution data: {e}")
            return [], [], {"error": str(e)}
    
    async def _get_task_flow_dag_data(self, time_range: str, limit: int, filters: Optional[Dict[str, Any]]):
        """Generate task flow DAG data for plan execution visualization."""
        try:
            # Get plan data from cognitive system
            plans = []
            if hasattr(cognitive_system, 'plan_memory') and cognitive_system.plan_memory:
                plans = cognitive_system.plan_memory.get_recent_plans(limit=limit)
            
            nodes = []
            edges = []
            
            for plan in plans:
                plan_id = plan.get('plan_id', 'unknown')
                
                # Add plan node
                nodes.append(GraphNode(
                    id=f"plan_{plan_id}",
                    label=plan.get('goal_text', 'Unknown Goal'),
                    type="plan",
                    data={
                        "status": plan.get('status', 'unknown'),
                        "confidence": plan.get('confidence', 0.5),
                        "execution_count": plan.get('execution_count', 0),
                        "success_rate": plan.get('success_count', 0) / max(plan.get('execution_count', 1), 1)
                    }
                ))
                
                # Add execution nodes
                for i, execution in enumerate(plan.get('executions', [])):
                    exec_id = f"exec_{plan_id}_{i}"
                    nodes.append(GraphNode(
                        id=exec_id,
                        label=f"Execution {i+1}",
                        type="execution",
                        data={
                            "status": execution.get('status', 'unknown'),
                            "duration": execution.get('duration', 0),
                            "outcome": execution.get('outcome', ''),
                            "score": execution.get('score', 0.0)
                        }
                    ))
                    
                    # Add plan->execution edge
                    edges.append(GraphEdge(
                        id=f"plan_exec_{plan_id}_{i}",
                        source=f"plan_{plan_id}",
                        target=exec_id,
                        label="executes",
                        weight=1.0
                    ))
                
                # Add memory formation edges
                memory_id = f"memory_{plan_id}"
                nodes.append(GraphNode(
                    id=memory_id,
                    label="Memory",
                    type="memory",
                    data={
                        "lessons_learned": plan.get('lessons_learned', []),
                        "success_patterns": plan.get('success_patterns', [])
                    }
                ))
                
                edges.append(GraphEdge(
                    id=f"plan_memory_{plan_id}",
                    source=f"plan_{plan_id}",
                    target=memory_id,
                    label="forms_memory",
                    weight=0.7
                ))
            
            metadata = {
                "total_plans": len(plans),
                "time_range": time_range,
                "execution_flow": "plan -> execution -> memory -> score"
            }
            
            return nodes, edges, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating task flow DAG data: {e}")
            return [], [], {"error": str(e)}
    
    async def _get_dissonance_heatmap_data(self, time_range: str, limit: int, filters: Optional[Dict[str, Any]]):
        """Generate cognitive dissonance heatmap data."""
        try:
            # Get dissonance data from cognitive system
            dissonance_events = []
            if hasattr(cognitive_system, 'get_dissonance_events'):
                dissonance_events = cognitive_system.get_dissonance_events(limit=limit)
            
            # Create a grid-based heatmap representation
            nodes = []
            edges = []
            
            # Group events by time buckets and intensity
            time_buckets = {}
            intensity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            
            for event in dissonance_events:
                timestamp = event.get('timestamp', datetime.now().isoformat())
                intensity = event.get('intensity', 0.0)
                
                # Find appropriate bucket
                bucket_key = timestamp[:13] if len(timestamp) > 13 else timestamp  # Hour bucket
                if bucket_key not in time_buckets:
                    time_buckets[bucket_key] = []
                time_buckets[bucket_key].append(intensity)
            
            # Create heatmap nodes
            for bucket, intensities in time_buckets.items():
                avg_intensity = sum(intensities) / len(intensities)
                node_id = f"heatmap_{bucket}"
                
                nodes.append(GraphNode(
                    id=node_id,
                    label=bucket,
                    type="heatmap_cell",
                    data={
                        "intensity": avg_intensity,
                        "event_count": len(intensities),
                        "max_intensity": max(intensities),
                        "timestamp": bucket
                    },
                    position={"x": len(nodes) * 50, "y": avg_intensity * 100}  # Simple positioning
                ))
            
            metadata = {
                "total_events": len(dissonance_events),
                "time_range": time_range,
                "intensity_range": [0.0, 1.0],
                "bucket_count": len(time_buckets),
                "visualization_type": "heatmap"
            }
            
            return nodes, edges, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating dissonance heatmap data: {e}")
            return [], [], {"error": str(e)}
    
    async def _get_recent_cognitive_events(self):
        """Get recent cognitive events for real-time visualization."""
        try:
            events = []
            
            # Get various types of cognitive events
            if cognitive_system:
                # Memory formation events
                recent_memories = cognitive_system.get_recent_memories(limit=10)
                for memory in recent_memories:
                    events.append({
                        "type": "memory_formation",
                        "timestamp": memory.get('timestamp', ''),
                        "content": memory.get('content', ''),
                        "importance": memory.get('importance', 0.5)
                    })
                
                # Contradiction detection events
                contradictions = cognitive_system.get_contradictions(limit=5)
                for contradiction in contradictions:
                    events.append({
                        "type": "contradiction_detected",
                        "timestamp": contradiction.get('timestamp', ''),
                        "score": contradiction.get('score', 0.0),
                        "facts_involved": 2
                    })
            
            # Sort by timestamp
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting recent cognitive events: {e}")
            return []
    
    async def _get_real_time_cognitive_state(self):
        """Get current real-time cognitive state."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "active_processes": [],
                "memory_pressure": 0.0,
                "contradiction_level": 0.0,
                "personality_stability": 1.0,
                "recent_activity_count": 0
            }
            
            if cognitive_system:
                # Get system status
                state["memory_pressure"] = getattr(cognitive_system, 'memory_pressure', 0.0)
                state["active_processes"] = ["memory_consolidation", "contradiction_detection", "planning"]
                
                # Calculate metrics
                recent_memories = cognitive_system.get_recent_memories(limit=10)
                state["recent_activity_count"] = len(recent_memories)
                
                contradictions = cognitive_system.get_contradictions(limit=10)
                if contradictions:
                    avg_contradiction_score = sum(c.get('score', 0) for c in contradictions) / len(contradictions)
                    state["contradiction_level"] = avg_contradiction_score
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error getting real-time cognitive state: {e}")
            return {"error": str(e)}
    
    def run(self, host: str = None, port: int = None, **kwargs):
        """Run the API server."""
        host = host or self.api_config.get('host', '127.0.0.1')
        port = port or self.api_config.get('port', 8181)
        
        log_level = self.api_config.get('log_level', 'info')
        reload = self.api_config.get('reload', False)
        workers = self.api_config.get('workers', 1)
        
        self.logger.info(f"Starting MeRNSTA System Bridge API on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
            workers=workers,
            **kwargs
        )


def main():
    """Main entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MeRNSTA System Bridge API Server')
    parser.add_argument('--host', default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    api = SystemBridgeAPI()
    api.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()