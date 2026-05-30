"""
FastAPI Medical Literature Q&A API
=====================================
REST API exposing the RAG pipeline for clinical literature retrieval
and answer synthesis.

Endpoints:
  POST /query              — Single question → grounded answer + citations
  POST /query/batch        — Batch questions (up to 20)
  GET  /search             — Raw retrieval (no LLM synthesis)
  GET  /health             — Health check + index stats
  POST /index/rebuild      — Rebuild FAISS index (admin)
  GET  /metrics            — Pipeline performance metrics

Authentication: API key via X-API-Key header (optional; disable for local dev)

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the mortality benefit of statins in heart failure?"}'
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App state (populated at startup)
# ---------------------------------------------------------------------------

_state: Dict = {
    "vector_store": None,
    "rag_engine": None,
    "corpus_df": None,
    "index_built": False,
    "n_docs": 0,
    "build_time_s": 0.0,
    "request_count": 0,
    "error_count": 0,
}


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """Build index on startup."""
    await _build_index()
    yield
    logger.info("API shutdown")


async def _build_index(n_docs: int = None):
    from corpus_builder import CorpusBuilder
    from vector_store import VectorStore
    from rag_engine import RAGEngine

    if n_docs is None:
        n_docs = int(os.getenv("CORPUS_SIZE", "50000"))

    index_dir = os.getenv("INDEX_DIR", "data/faiss_index")
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() == "true"

    t0 = time.time()

    # Load existing index if available
    if os.path.exists(os.path.join(index_dir, "index.faiss")) and not force_rebuild:
        logger.info("Loading existing FAISS index from %s", index_dir)
        vs = VectorStore.load(index_dir)
    else:
        logger.info("Building FAISS index (%d docs)...", n_docs)
        builder = CorpusBuilder(use_synthetic=True)
        corpus_df = builder.build(n_abstracts=n_docs, output_dir="data/")
        _state["corpus_df"] = corpus_df

        vs = VectorStore()
        vs.build(corpus_df, text_col="text")
        vs.save(index_dir)

    engine = RAGEngine(vector_store=vs)

    _state["vector_store"] = vs
    _state["rag_engine"] = engine
    _state["index_built"] = True
    _state["n_docs"] = vs._n_docs
    _state["build_time_s"] = round(time.time() - t0, 1)

    logger.info(
        "Index ready: %d docs, built in %.1fs",
        _state["n_docs"], _state["build_time_s"]
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator

    app = FastAPI(
        title="Medical Literature Q&A API",
        description=(
            "RAG-powered biomedical literature Q&A over 50K PubMed abstracts. "
            "Retrieves relevant studies and synthesizes grounded answers with citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------
    # Request / Response models
    # ---------------------------------------------------------------

    class QueryRequest(BaseModel):
        question: str = Field(
            ...,
            min_length=5,
            max_length=1000,
            description="Clinical or research question",
            example="What is the mortality benefit of beta-blockers in heart failure?",
        )
        top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
        min_score: float = Field(default=0.25, ge=0.0, le=1.0, description="Min similarity score")
        include_retrieved_docs: bool = Field(default=False, description="Include raw retrieved docs")

    class BatchQueryRequest(BaseModel):
        questions: List[str] = Field(..., min_items=1, max_items=20)
        top_k: int = Field(default=5, ge=1, le=20)

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=3, max_length=500)
        k: int = Field(default=10, ge=1, le=50)
        min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class RebuildRequest(BaseModel):
        n_docs: int = Field(default=50000, ge=100, le=200000)
        admin_key: str = Field(..., description="Admin key for index rebuild")

    # ---------------------------------------------------------------
    # Auth helper
    # ---------------------------------------------------------------

    API_KEY = os.getenv("API_KEY", "")  # Empty = auth disabled

    def check_auth(x_api_key: Optional[str]):
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ---------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------

    @app.get("/health")
    async def health():
        """Health check and index statistics."""
        return {
            "status": "healthy" if _state["index_built"] else "initializing",
            "index_built": _state["index_built"],
            "n_docs": _state["n_docs"],
            "index_build_time_s": _state["build_time_s"],
            "request_count": _state["request_count"],
            "error_count": _state["error_count"],
        }

    @app.post("/query")
    async def query(
        request: QueryRequest,
        x_api_key: Optional[str] = Header(default=None),
    ):
        """
        Answer a biomedical literature question using RAG.

        Returns a grounded answer synthesized from retrieved PubMed abstracts,
        with inline citations and source metadata.
        """
        check_auth(x_api_key)
        if not _state["index_built"]:
            raise HTTPException(status_code=503, detail="Index not ready")

        _state["request_count"] += 1
        engine = _state["rag_engine"]

        # Override engine settings per request
        engine.top_k = request.top_k
        engine.min_score = request.min_score

        try:
            response = engine.query(request.question)
        except Exception as e:
            _state["error_count"] += 1
            logger.error("Query failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        result = response.to_dict()
        if not request.include_retrieved_docs:
            result.pop("retrieved_docs", None)

        return JSONResponse(content=result)

    @app.post("/query/batch")
    async def batch_query(
        request: BatchQueryRequest,
        x_api_key: Optional[str] = Header(default=None),
    ):
        """Answer multiple questions in a single request (max 20)."""
        check_auth(x_api_key)
        if not _state["index_built"]:
            raise HTTPException(status_code=503, detail="Index not ready")

        _state["request_count"] += len(request.questions)
        engine = _state["rag_engine"]
        engine.top_k = request.top_k

        try:
            responses = engine.batch_query(request.questions)
        except Exception as e:
            _state["error_count"] += 1
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={
            "results": [r.to_dict() for r in responses],
            "n_queries": len(responses),
        })

    @app.get("/search")
    async def search(
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        x_api_key: Optional[str] = Header(default=None),
    ):
        """
        Raw vector similarity search — returns relevant abstracts without LLM synthesis.
        Useful for testing retrieval quality independently of generation.
        """
        check_auth(x_api_key)
        if not _state["index_built"]:
            raise HTTPException(status_code=503, detail="Index not ready")

        vs = _state["vector_store"]
        results = vs.search(query, k=min(k, 50), min_score=min_score)

        return JSONResponse(content={
            "query": query,
            "n_results": len(results),
            "results": [
                {
                    "rank": r.rank,
                    "pmid": r.pmid,
                    "title": r.title,
                    "topic": r.topic,
                    "score": r.score,
                    "snippet": r.text_snippet,
                }
                for r in results
            ],
        })

    @app.get("/metrics")
    async def metrics(x_api_key: Optional[str] = Header(default=None)):
        """Return pipeline performance metrics."""
        check_auth(x_api_key)
        return JSONResponse(content={
            "request_count": _state["request_count"],
            "error_count": _state["error_count"],
            "error_rate": (
                _state["error_count"] / _state["request_count"]
                if _state["request_count"] > 0 else 0.0
            ),
            "n_docs_indexed": _state["n_docs"],
        })

    @app.post("/index/rebuild")
    async def rebuild_index(
        request: RebuildRequest,
        background_tasks: BackgroundTasks,
    ):
        """Rebuild FAISS index with updated corpus (admin only)."""
        admin_key = os.getenv("ADMIN_KEY", "")
        if admin_key and request.admin_key != admin_key:
            raise HTTPException(status_code=403, detail="Invalid admin key")

        background_tasks.add_task(_build_index, n_docs=request.n_docs)
        return {"status": "rebuild_started", "n_docs": request.n_docs}

    # ---------------------------------------------------------------
    # 404 handler
    # ---------------------------------------------------------------

    @app.exception_handler(404)
    async def not_found(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found", "available_endpoints": [
                "POST /query", "POST /query/batch", "GET /search",
                "GET /health", "GET /metrics", "POST /index/rebuild",
            ]},
        )

except ImportError:
    logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
    app = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level="info",
    )
