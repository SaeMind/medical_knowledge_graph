"""
NER-Enriched Medical Literature API — Updated api.py
======================================================
Extends the base API with NER-enriched endpoints.
Drop-in replacement for the existing src/api.py in medical_knowledge_graph.

New endpoints:
  POST /query/entity        — Entity-filtered RAG query
  GET  /entities/search     — Pure entity search by disease/chemical/gene
  GET  /entities/cooccur    — Co-occurrence graph for a named entity
  GET  /entities/summary    — Corpus-level entity statistics
  GET  /entities/abstract   — All entities for a specific PMID

Existing endpoints unchanged:
  POST /query, POST /query/batch, GET /search, GET /health,
  GET /metrics, POST /index/rebuild
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

_state: Dict = {
    "vector_store": None,
    "rag_engine": None,
    "ner_pipeline": None,
    "entity_index": None,
    "ner_rag_engine": None,
    "corpus_df": None,
    "index_built": False,
    "ner_built": False,
    "n_docs": 0,
    "build_time_s": 0.0,
    "request_count": 0,
    "error_count": 0,
    "prediction_log": [],
}

N_DOCS = int(os.getenv("CORPUS_SIZE", "50000"))
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")
NER_INDEX_DIR = os.getenv("NER_INDEX_DIR", "data/entity_index")
MODEL_DIR = os.getenv("MODEL_DIR", "models/")


@asynccontextmanager
async def lifespan(app):
    await _build_all()
    yield


async def _build_all():
    from corpus_builder import CorpusBuilder
    from vector_store import VectorStore
    from rag_engine import RAGEngine
    from ner_pipeline import NERPipeline
    from entity_index import EntityIndex
    from ner_rag_engine import NEREnrichedRAGEngine
    import glob

    t0 = time.time()

    # --- Vector store ---
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        logger.info("Loading FAISS index...")
        vs = VectorStore.load(INDEX_DIR)
    else:
        logger.info("Building corpus + FAISS index (%d docs)...", N_DOCS)
        builder = CorpusBuilder(use_synthetic=True)
        corpus_df = builder.build(n_abstracts=N_DOCS, output_dir="data/")
        _state["corpus_df"] = corpus_df
        vs = VectorStore()
        vs.build(corpus_df, text_col="text")
        vs.save(INDEX_DIR)

    _state["vector_store"] = vs
    _state["rag_engine"] = RAGEngine(vector_store=vs)
    _state["n_docs"] = vs._n_docs

    # --- NER pipeline ---
    logger.info("Loading NER pipeline...")
    ner = NERPipeline(batch_size=64)
    ner.load_model()
    _state["ner_pipeline"] = ner

    # --- Entity index ---
    ner_index_path = os.path.join(NER_INDEX_DIR, "entity_index.pkl")
    if os.path.exists(ner_index_path):
        logger.info("Loading entity index...")
        entity_idx = EntityIndex.load(NER_INDEX_DIR)
    else:
        logger.info("Building entity index (NER enrichment)...")
        # Load or rebuild corpus
        corpus_path = "data/pubmed_corpus.parquet"
        if os.path.exists(corpus_path):
            import pandas as pd
            corpus_df = pd.read_parquet(corpus_path)
        elif _state["corpus_df"] is not None:
            corpus_df = _state["corpus_df"]
        else:
            from corpus_builder import CorpusBuilder
            builder = CorpusBuilder(use_synthetic=True)
            corpus_df = builder.build(n_abstracts=N_DOCS, output_dir="data/")

        enriched_df = ner.enrich_corpus(corpus_df, text_col="abstract")
        entity_idx = EntityIndex()
        entity_idx.build(enriched_df)
        entity_idx.save(NER_INDEX_DIR)

    _state["entity_index"] = entity_idx
    _state["ner_rag_engine"] = NEREnrichedRAGEngine(
        vector_store=vs,
        entity_index=entity_idx,
        ner_pipeline=ner,
    )
    _state["index_built"] = True
    _state["ner_built"] = True
    _state["build_time_s"] = round(time.time() - t0, 1)
    logger.info("All systems ready in %.1fs", _state["build_time_s"])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="NER-Enriched Medical Literature Q&A API",
        description=(
            "RAG + SciSpacy NER pipeline over 50K PubMed abstracts. "
            "Entity-filtered retrieval by disease, chemical, and gene."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # ---------------------------------------------------------------
    # Request models
    # ---------------------------------------------------------------

    class QueryRequest(BaseModel):
        question: str = Field(..., min_length=5, max_length=1000)
        top_k: int = Field(default=5, ge=1, le=20)
        min_score: float = Field(default=0.20, ge=0.0, le=1.0)
        include_retrieved_docs: bool = False

    class EntityQueryRequest(BaseModel):
        question: str = Field(..., min_length=5, max_length=1000)
        disease: Optional[str] = Field(default=None, example="heart failure")
        chemical: Optional[str] = Field(default=None, example="statin")
        gene: Optional[str] = Field(default=None, example="BRCA1")
        top_k: int = Field(default=5, ge=1, le=20)
        auto_entity_filter: bool = Field(
            default=True,
            description="Also extract entities from query automatically"
        )

    class BatchQueryRequest(BaseModel):
        questions: List[str] = Field(..., min_items=1, max_items=20)
        top_k: int = Field(default=5, ge=1, le=20)

    # ---------------------------------------------------------------
    # Existing endpoints (unchanged)
    # ---------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if _state["index_built"] else "initializing",
            "index_built": _state["index_built"],
            "ner_built": _state["ner_built"],
            "n_docs": _state["n_docs"],
            "build_time_s": _state["build_time_s"],
            "ner_model": _state["ner_pipeline"].model_info if _state["ner_pipeline"] else None,
            "request_count": _state["request_count"],
            "error_count": _state["error_count"],
        }

    @app.post("/query")
    async def query(request: QueryRequest):
        """Standard RAG query (no entity filter)."""
        if not _state["index_built"]:
            raise HTTPException(503, "Index not ready")
        _state["request_count"] += 1
        engine = _state["rag_engine"]
        engine.top_k = request.top_k
        engine.min_score = request.min_score
        try:
            response = engine.query(request.question)
        except Exception as e:
            _state["error_count"] += 1
            raise HTTPException(500, str(e))
        result = response.to_dict()
        if not request.include_retrieved_docs:
            result.pop("retrieved_docs", None)
        return JSONResponse(content=result)

    @app.post("/query/batch")
    async def batch_query(request: BatchQueryRequest):
        """Batch RAG query."""
        if not _state["index_built"]:
            raise HTTPException(503, "Index not ready")
        _state["request_count"] += len(request.questions)
        engine = _state["rag_engine"]
        engine.top_k = request.top_k
        responses = engine.batch_query(request.questions)
        return JSONResponse(content={
            "results": [r.to_dict() for r in responses],
            "n_queries": len(responses),
        })

    @app.get("/search")
    async def search(query: str, k: int = 10, min_score: float = 0.0):
        """Raw vector similarity search."""
        if not _state["index_built"]:
            raise HTTPException(503, "Index not ready")
        vs = _state["vector_store"]
        results = vs.search(query, k=min(k, 50), min_score=min_score)
        return JSONResponse(content={
            "query": query,
            "n_results": len(results),
            "results": [
                {"rank": r.rank, "pmid": r.pmid, "title": r.title,
                 "topic": r.topic, "score": r.score, "snippet": r.text_snippet}
                for r in results
            ],
        })

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content={
            "request_count": _state["request_count"],
            "error_count": _state["error_count"],
            "n_docs_indexed": _state["n_docs"],
        })

    # ---------------------------------------------------------------
    # New NER endpoints
    # ---------------------------------------------------------------

    @app.post("/query/entity")
    async def entity_query(request: EntityQueryRequest):
        """
        Entity-filtered RAG query.

        Extracts biomedical entities from the question, boosts results
        containing matching entities, and optionally hard-filters by
        explicitly provided disease/chemical/gene terms.

        Returns the standard RAG response plus:
          - query_entities: NER entities extracted from question
          - n_after_entity_filter: docs remaining after entity filter
          - entity_filter_applied: whether hard filter was used
        """
        if not _state["ner_built"]:
            raise HTTPException(503, "NER index not ready")
        _state["request_count"] += 1

        entity_filter = {}
        if request.disease:  entity_filter["disease"]  = request.disease
        if request.chemical: entity_filter["chemical"] = request.chemical
        if request.gene:     entity_filter["gene"]     = request.gene

        engine = _state["ner_rag_engine"]
        engine.top_k = request.top_k

        try:
            response = engine.query(
                question=request.question,
                entity_filter=entity_filter if entity_filter else None,
                auto_entity_filter=request.auto_entity_filter,
            )
        except Exception as e:
            _state["error_count"] += 1
            raise HTTPException(500, str(e))

        return JSONResponse(content=response.to_dict())

    @app.get("/entities/search")
    async def entity_search(
        disease: Optional[str] = None,
        chemical: Optional[str] = None,
        gene: Optional[str] = None,
        operator: str = "AND",
        limit: int = 20,
    ):
        """
        Pure entity search — returns matching PMIDs without vector search or LLM.
        Useful for browsing the corpus by entity type.

        Example: /entities/search?disease=heart+failure&chemical=statin
        """
        if not _state["ner_built"]:
            raise HTTPException(503, "Entity index not ready")
        if not any([disease, chemical, gene]):
            raise HTTPException(400, "Provide at least one of: disease, chemical, gene")

        idx = _state["entity_index"]
        pmids = idx.search(
            disease=disease, chemical=chemical, gene=gene, operator=operator
        )[:limit]

        return JSONResponse(content={
            "disease": disease,
            "chemical": chemical,
            "gene": gene,
            "operator": operator,
            "n_matches": len(pmids),
            "pmids": pmids,
        })

    @app.get("/entities/cooccur")
    async def entity_cooccurrence(entity: str, top_n: int = 10, min_count: int = 2):
        """
        Return top co-occurring entities for a given entity text.
        Useful for understanding entity relationships in the corpus.

        Example: /entities/cooccur?entity=metformin
        """
        if not _state["ner_built"]:
            raise HTTPException(503, "Entity index not ready")
        idx = _state["entity_index"]
        cooccur = idx.get_cooccurrences(entity, top_n=top_n, min_count=min_count)
        return JSONResponse(content={
            "entity": entity,
            "top_cooccurrences": [
                {"entity": e, "count": c} for e, c in cooccur
            ],
        })

    @app.get("/entities/summary")
    async def entity_summary():
        """Corpus-level entity statistics — top entities per type, counts."""
        if not _state["ner_built"]:
            raise HTTPException(503, "Entity index not ready")
        idx = _state["entity_index"]
        summary = idx.get_entity_summary()
        # Convert tuples to dicts for JSON
        for k in ["top_diseases", "top_chemicals", "top_genes"]:
            if isinstance(summary.get(k), list):
                summary[k] = [
                    {"entity": item[0], "count": item[1]}
                    if isinstance(item, (list, tuple)) else item
                    for item in summary[k]
                ]
        return JSONResponse(content=summary)

    @app.get("/entities/abstract")
    async def abstract_entities(pmid: str):
        """Return all NER entities extracted from a specific abstract."""
        if not _state["ner_built"]:
            raise HTTPException(503, "Entity index not ready")
        idx = _state["entity_index"]
        entities = idx.get_abstract_entities(pmid)
        if not entities:
            raise HTTPException(404, f"No entities found for PMID {pmid}")
        by_label: Dict[str, List] = {}
        for ent in entities:
            label = ent.get("label", "OTHER")
            by_label.setdefault(label, []).append(ent)
        return JSONResponse(content={
            "pmid": pmid,
            "n_entities": len(entities),
            "by_label": by_label,
        })

except ImportError as e:
    logger.error("FastAPI not installed: %s", e)
    app = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ner_api:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
