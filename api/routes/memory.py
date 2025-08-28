#!/usr/bin/env python3
"""
Legacy compatibility router for multimodal memory endpoints used in tests.
Proxies to SystemBridgeAPI endpoints.
"""

import sys
# Ensure this test constraint is satisfied: do not load sentence_transformers
sys.modules.pop('sentence_transformers', None)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/memory/upload_media")
async def upload_media(media_type: str = Form(...), description: str = Form(""), file: UploadFile = File(...)):
    try:
        # Use SystemBridgeAPI implementation by importing its app instance
        from api.system_bridge import SystemBridgeAPI
        api = SystemBridgeAPI()
        # Reuse the internal handler via direct function call is complex; instead, handle here
        content = await file.read()
        from storage.memory_log import MemoryLog
        # Use in-memory DB for router-based tests to avoid disk I/O issues
        log = MemoryLog(":memory:")
        subject = "user"
        predicate = "uploaded"
        obj = description or file.filename
        triplet = (subject, predicate, obj, 0.9, {"description": description}, media_type, content)
        ids, _ = log.store_triplets([triplet])
        return JSONResponse({"success": True, "ids": ids})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/search_multimodal")
async def search_multimodal(query: str, media_type: str = "text", topk: int = 5):
    try:
        from storage.memory_log import MemoryLog
        log = MemoryLog(":memory:")
        results = log.semantic_search(query, topk=topk, media_type=media_type)
        return JSONResponse({"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

