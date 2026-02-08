import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

try:
    from gliner2 import GLiNER2
except ImportError as e:
    # Fallback/mock for when gliner2 is not installed (prevents immediate crash if dependencies missing)
    print(f"Error importing GLiNER2: {e}")
    GLiNER2 = None
except Exception as e:
    print(f"Unexpected error importing GLiNER2: {e}")
    GLiNER2 = None

# Global model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if GLiNER2 is None:
        print("Warning: GLiNER2 library not found. Server functionality will be limited.")
        yield
        return

    # Load model (base version by default)
    # In production, this might come from env var
    model_name = os.getenv("GLINER_MODEL", "fastino/gliner2-multi-v1")
    print(f"Loading GLiNER2 model: {model_name}...")
    try:
        model = GLiNER2.from_pretrained(model_name)
        print("GLiNER2 model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    # Cleanup if needed (optional)
    model = None

app = FastAPI(lifespan=lifespan)

class ExtractRequest(BaseModel):
    task: str
    text: str
    # Rename field to avoid shadowing Pydantic's 'schema' method, but alias it to 'schema' for API compatibility
    extraction_schema: Union[List[str], Dict[str, Any], Any] = Field(
        alias="schema",
        description="Schema for extraction (list of labels or dict map)"
    )
    threshold: float = 0.5

@app.get("/health")
def health_check():
    status = "healthy" if model else "degraded"
    return {
        "status": status,
        "models": ["fastino/gliner2-multi-v1"],
        "timestamp": str(datetime.now())
    }

@app.post("/gliner-2")
def handle_request(request: ExtractRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized or GLiNER2 library missing")
    
    try:
        if request.task == "extract_entities":
            result = model.extract_entities(
                request.text,
                request.extraction_schema,
                threshold=request.threshold,
                include_confidence=True,
                include_spans=True
            )
            return {"result": result}

        elif request.task == "extract_relations":
            result = model.extract_relations(
                request.text,
                request.extraction_schema,
                threshold=request.threshold,
                include_confidence=True,
                include_spans=True
            )
            return {"result": result}

        elif request.task == "extract_json":
            result = model.extract_json(
                request.text,
                request.extraction_schema,
                threshold=request.threshold,
                include_confidence=True,
                include_spans=True
            )
            return {"result": result}

        elif request.task == "classify_text":
            result = model.classify_text(
                request.text,
                request.extraction_schema,
                threshold=request.threshold,
            )
            return {"result": result}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task: {request.task}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting GLiNER2 Python Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start()
