"""
FastAPI REST API for CQE

Production-ready API with:
- Health checks
- Embedding endpoints
- Query endpoints
- Metrics retrieval
- Async support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime

from cqe import CQEClient, __version__
from cqe.core.overlay import CQEOverlay


# Pydantic models for request/response validation
class EmbedRequest(BaseModel):
    """Request model for embedding"""
    content: str = Field(..., min_length=1, max_length=100000)
    domain: str = Field(default="text", pattern="^(text|code|scientific)$")
    optimize: bool = Field(default=True)


class EmbedResponse(BaseModel):
    """Response model for embedding"""
    overlay_id: str
    active_slots: int
    cartan_active: int
    phi_metrics: Dict[str, float]
    processing_time_ms: float


class QueryRequest(BaseModel):
    """Request model for similarity query"""
    overlay_id: str
    top_k: int = Field(default=10, ge=1, le=100)


class QueryResponse(BaseModel):
    """Response model for similarity query"""
    results: List[Dict[str, Any]]
    query_overlay_id: str


class TransformRequest(BaseModel):
    """Request model for transformation"""
    overlay_id: str
    operator: str = Field(..., pattern="^(rotation|midpoint|parity)$")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


# Initialize FastAPI app
app = FastAPI(
    title="CQE API",
    description="Cartan-Quadratic Equivalence Framework API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CQE client (singleton)
cqe_client: Optional[CQEClient] = None


@app.on_event("startup")
async def startup_event():
    """Initialize CQE client on startup"""
    global cqe_client
    cqe_client = CQEClient()
    print(f"CQE API v{__version__} started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("CQE API shutting down")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "CQE API",
        "version": __version__,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat(),
        components={
            "api": "operational",
            "cqe_client": "initialized" if cqe_client else "not_initialized",
            "e8_lattice": "ready",
            "morsr": "ready"
        }
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed_content(request: EmbedRequest):
    """
    Embed content into E8 space.

    Extracts features, projects to E8 lattice, applies MORSR optimization,
    and returns overlay with metrics.
    """
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    try:
        import time
        start_time = time.time()

        # Embed content
        overlay = cqe_client.embed(
            content=request.content,
            domain=request.domain,
            optimize=request.optimize
        )

        # Get metrics
        metrics = cqe_client.get_phi_metrics(overlay)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return EmbedResponse(
            overlay_id=overlay.hash_id,
            active_slots=len(overlay.active_slots),
            cartan_active=overlay.cartan_active,
            phi_metrics=metrics,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_similar(request: QueryRequest):
    """
    Query for similar overlays.

    Finds overlays in cache with similar Φ values and structural properties.
    """
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    try:
        # Get overlay from cache
        cache_stats = cqe_client.get_cache_stats()

        if request.overlay_id not in cache_stats['overlays']:
            raise HTTPException(status_code=404, detail="Overlay not found in cache")

        query_overlay = cqe_client._overlay_cache[request.overlay_id]

        # Find similar
        similar = cqe_client.find_similar(query_overlay, top_k=request.top_k)

        # Format results
        results = []
        for overlay, distance in similar:
            results.append({
                'overlay_id': overlay.hash_id,
                'distance': float(distance),
                'active_slots': len(overlay.active_slots),
                'cartan_active': overlay.cartan_active
            })

        return QueryResponse(
            results=results,
            query_overlay_id=request.overlay_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/transform")
async def transform_overlay(request: TransformRequest):
    """
    Apply operator transformation to overlay.

    Applies ALENA operator and returns transformed overlay with metrics.
    """
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    try:
        # Get overlay from cache
        cache_stats = cqe_client.get_cache_stats()

        if request.overlay_id not in cache_stats['overlays']:
            raise HTTPException(status_code=404, detail="Overlay not found in cache")

        overlay = cqe_client._overlay_cache[request.overlay_id]

        # Apply transformation
        transformed = cqe_client.apply_operator(request.operator, overlay)

        # Get metrics
        original_metrics = cqe_client.get_phi_metrics(overlay)
        transformed_metrics = cqe_client.get_phi_metrics(transformed)

        return {
            'original_overlay_id': overlay.hash_id,
            'transformed_overlay_id': transformed.hash_id,
            'operator': request.operator,
            'phi_delta': transformed_metrics['phi_total'] - original_metrics['phi_total'],
            'original_metrics': original_metrics,
            'transformed_metrics': transformed_metrics
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")


@app.get("/metrics/{overlay_id}")
async def get_overlay_metrics(overlay_id: str):
    """Get Φ metrics for specific overlay"""
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    cache_stats = cqe_client.get_cache_stats()

    if overlay_id not in cache_stats['overlays']:
        raise HTTPException(status_code=404, detail="Overlay not found")

    overlay = cqe_client._overlay_cache[overlay_id]
    metrics = cqe_client.get_phi_metrics(overlay)

    return {
        'overlay_id': overlay_id,
        'metrics': metrics,
        'active_slots': len(overlay.active_slots),
        'cartan_active': overlay.cartan_active,
        'provenance': overlay.provenance
    }


@app.get("/cache")
async def get_cache_info():
    """Get cache statistics"""
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    return cqe_client.get_cache_stats()


@app.get("/lattice")
async def get_lattice_info():
    """Get E8 lattice information"""
    if not cqe_client:
        raise HTTPException(status_code=503, detail="CQE client not initialized")

    return cqe_client.lattice.info()


def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "cqe.api.rest:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    serve()
