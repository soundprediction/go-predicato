"""
GLiNER2 Local NER Service

A FastAPI service that loads the GLiNER2 model and serves entity extraction
and relation extraction over HTTP, compatible with the Predicato Go server's
gliner2 HTTP client (pkg/gliner2/http_client.go).

Usage:
    python -m predicato.gliner2_service

Environment variables:
    PORT            - Port to listen on (default: 8000)
    GLINER2_MODEL   - Model name to load (default: fastino/gliner2-base-v1)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger("gliner2_service")

# Global model reference, loaded at startup
_model = None


def _load_model():
    """Load the GLiNER2 model. Imports gliner2 lazily to allow the module
    to be importable even without the gliner2 dependency installed."""
    from gliner2 import GLiNER2

    model_name = os.environ.get("GLINER2_MODEL", "fastino/gliner2-base-v1")
    logger.info("Loading GLiNER2 model: %s", model_name)
    model = GLiNER2.from_pretrained(model_name)
    logger.info("GLiNER2 model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    yield
    _model = None


app = FastAPI(title="GLiNER2 Service", lifespan=lifespan)


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/gliner-2")
async def gliner2_endpoint(request: Request):
    """Accept raw JSON to avoid Pydantic conflicts with the 'schema' field name."""
    body = await request.json()
    task = body.get("task", "")
    text = body.get("text", "")
    schema = body.get("schema")
    threshold = body.get("threshold", 0.5)

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if task == "extract_entities":
        return _extract_entities(text, schema, threshold)
    elif task == "extract_relations":
        return _extract_relations(text, schema, threshold)
    elif task == "classify_text":
        raise HTTPException(status_code=501, detail="Text classification not yet implemented")
    elif task == "extract_json":
        raise HTTPException(status_code=501, detail="Structured extraction not yet implemented")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")


def _extract_entities(text: str, schema: Any, threshold: float) -> dict:
    """Run entity extraction and return in the format expected by the Go client."""
    if not schema:
        return {"result": {"entities": {}}}

    # Ensure schema is a list of strings
    entity_types = list(schema) if isinstance(schema, (list, tuple)) else [str(schema)]

    result = _model.extract_entities(
        text,
        entity_types,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
    )

    # result format from GLiNER2: {"entities": {"label": [{"text": ..., "confidence": ..., "start": ..., "end": ...}]}}
    # Go client expects: {"result": {"entities": {"label": [{"text": ..., "confidence": ..., "start": ..., "end": ...}]}}}
    entities = result.get("entities", {})

    # Normalize: ensure each entity has the expected fields
    formatted = {}
    for label, ents in entities.items():
        formatted_ents = []
        for e in ents:
            if isinstance(e, dict):
                formatted_ents.append({
                    "text": e.get("text", ""),
                    "confidence": e.get("confidence", 0.0),
                    "start": e.get("start", 0),
                    "end": e.get("end", 0),
                })
            elif isinstance(e, str):
                formatted_ents.append({"text": e, "confidence": 1.0})
            else:
                formatted_ents.append({"text": str(e), "confidence": 1.0})
        formatted[label] = formatted_ents

    return {"result": {"entities": formatted}}


def _extract_relations(text: str, schema: Any, threshold: float) -> dict:
    """Run relation extraction and return in the format expected by the Go client."""
    if not schema:
        return {"result": {"relation_extraction": {}}}

    relation_types = list(schema) if isinstance(schema, (list, tuple)) else [str(schema)]

    result = _model.extract_relations(
        text,
        relation_types,
        threshold=threshold,
    )

    # result format from GLiNER2: {"relation_extraction": {"type": [("head", "tail"), ...]}}
    # Go client expects: {"result": {"relation_extraction": {"type": [{"head": "...", "tail": "..."}]}}}
    raw = result.get("relation_extraction", {})

    formatted = {}
    for rel_type, tuples in raw.items():
        formatted_tuples = []
        for t in tuples:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                formatted_tuples.append({"head": str(t[0]), "tail": str(t[1])})
            elif isinstance(t, dict):
                formatted_tuples.append({
                    "head": str(t.get("head", "")),
                    "tail": str(t.get("tail", "")),
                })
        formatted[rel_type] = formatted_tuples

    return {"result": {"relation_extraction": formatted}}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=port)
