"""
NER-Enriched RAG Engine
========================
Extends the base RAG pipeline with entity-aware retrieval:

  1. Extract entities from the user's query (SciSpacy NER)
  2. Vector similarity search (as before)
  3. Entity-filtered hybrid reranking:
     - Boost abstracts that contain query entities
     - Filter by entity type when user specifies (e.g. "papers about statin")
  4. LLM synthesis with entity-aware context

New API endpoints added:
  POST /query/entity    — Entity-filtered RAG query
  GET  /entities/search — Pure entity search (no vector, no LLM)
  GET  /entities/cooccur — Co-occurrence graph for an entity
  GET  /entities/summary — Index-level entity statistics

Usage:
    from src.ner_rag_engine import NEREnrichedRAGEngine
    engine = NEREnrichedRAGEngine(vector_store, entity_index, ner_pipeline)
    response = engine.query(
        "What statins reduce mortality in heart failure?",
        entity_filter={"disease": "heart failure", "chemical": "statin"}
    )
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NERRAGResponse:
    """Extended RAG response with entity metadata."""
    query: str
    answer: str
    citations: List[Dict]
    retrieved_docs: List[Dict]
    n_retrieved: int
    n_after_entity_filter: int
    n_context_docs: int
    query_entities: List[Dict]          # Entities extracted from query
    mean_retrieval_score: float
    confidence: str
    latency_ms: int
    model_used: str
    fallback_used: bool
    entity_filter_applied: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "n_retrieved": self.n_retrieved,
            "n_after_entity_filter": self.n_after_entity_filter,
            "n_context_docs": self.n_context_docs,
            "query_entities": self.query_entities,
            "mean_retrieval_score": self.mean_retrieval_score,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "entity_filter_applied": self.entity_filter_applied,
        }


class NEREnrichedRAGEngine:
    """
    Entity-aware RAG engine combining vector similarity with NER filtering.

    Retrieval strategy:
      1. Vector search → top-K candidates
      2. Query NER → extract disease/chemical/gene from question
      3. Entity boost: candidates containing query entities ranked higher
      4. If entity_filter explicitly provided, hard-filter candidates
      5. LLM synthesis on reranked context
    """

    def __init__(
        self,
        vector_store,
        entity_index,
        ner_pipeline,
        top_k: int = 8,
        top_k_context: int = 5,
        min_score: float = 0.20,
        entity_boost: float = 0.15,   # Score boost for entity-matching docs
        llm_provider: str = "auto",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.vector_store = vector_store
        self.entity_index = entity_index
        self.ner_pipeline = ner_pipeline
        self.top_k = top_k
        self.top_k_context = top_k_context
        self.min_score = min_score
        self.entity_boost = entity_boost
        self.llm_provider = llm_provider
        self.model = model

    # ------------------------------------------------------------------
    # Primary query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        entity_filter: Optional[Dict[str, str]] = None,
        auto_entity_filter: bool = True,
    ) -> NERRAGResponse:
        """
        Entity-aware RAG query.

        Args:
            question:          Natural language clinical question.
            entity_filter:     Optional explicit filter, e.g.
                               {"disease": "heart failure", "chemical": "statin"}
            auto_entity_filter: If True, extract entities from question
                               automatically and use for boosting.

        Returns:
            NERRAGResponse with grounded answer and entity metadata.
        """
        t0 = time.time()

        # Step 1: Extract entities from query
        query_entities = []
        if auto_entity_filter or entity_filter:
            raw_entities = self.ner_pipeline.extract(question)
            query_entities = [e.to_dict() for e in raw_entities]

        # Step 2: Vector retrieval
        retrieved = self.vector_store.search(
            question, k=self.top_k, min_score=self.min_score
        )

        if not retrieved:
            return self._empty_response(question, query_entities, t0)

        # Step 3: Entity boost / filter
        entity_filter_applied = False
        n_after_filter = len(retrieved)

        if entity_filter:
            # Hard filter
            filtered = self.entity_index.filter_vector_results(
                retrieved,
                disease=entity_filter.get("disease"),
                chemical=entity_filter.get("chemical"),
                gene=entity_filter.get("gene"),
            )
            if filtered:  # Only apply if results remain
                retrieved = filtered
                entity_filter_applied = True
                n_after_filter = len(retrieved)
        elif auto_entity_filter and query_entities:
            # Soft boost: rerank by entity match
            retrieved = self._entity_boost_rerank(retrieved, query_entities)
            n_after_filter = len(retrieved)

        # Step 4: Select context docs
        context_docs = retrieved[:self.top_k_context]
        scores = [r.score for r in retrieved]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Step 5: Build context and generate answer
        context_str = self._build_entity_aware_context(context_docs, query_entities)
        answer, model_used, fallback = self._generate_answer(question, context_str)
        citations = self._extract_citations(answer, context_docs)
        confidence = self._assess_confidence(mean_score, len(retrieved), fallback)
        latency = int((time.time() - t0) * 1000)

        return NERRAGResponse(
            query=question,
            answer=answer,
            citations=citations,
            retrieved_docs=[
                {"rank": r.rank, "pmid": r.pmid, "title": r.title,
                 "score": r.score, "topic": r.topic}
                for r in retrieved
            ],
            n_retrieved=len(retrieved),
            n_after_entity_filter=n_after_filter,
            n_context_docs=len(context_docs),
            query_entities=query_entities,
            mean_retrieval_score=round(mean_score, 4),
            confidence=confidence,
            latency_ms=latency,
            model_used=model_used,
            fallback_used=fallback,
            entity_filter_applied=entity_filter_applied,
        )

    # ------------------------------------------------------------------
    # Entity-aware reranking
    # ------------------------------------------------------------------

    def _entity_boost_rerank(self, results, query_entities: List[Dict]):
        """Boost scores for results containing query entities."""
        if not query_entities:
            return results

        query_entity_texts = {
            e.get("normalized", e.get("text", "")).lower()
            for e in query_entities
        }

        boosted = []
        for result in results:
            boost = 0.0
            doc_entities = self.entity_index.get_abstract_entities(result.pmid)
            doc_entity_texts = {
                e.get("normalized", e.get("text", "")).lower()
                for e in doc_entities
            }
            overlap = query_entity_texts & doc_entity_texts
            if overlap:
                boost = self.entity_boost * len(overlap)
            # Create boosted copy
            from dataclasses import replace
            try:
                boosted_result = replace(result, score=min(1.0, result.score + boost))
            except Exception:
                boosted_result = result
            boosted.append((boosted_result, boost))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[0].score, reverse=True)
        reranked = [r for r, _ in boosted]
        for i, r in enumerate(reranked):
            r.rank = i + 1
        return reranked

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_entity_aware_context(
        self, docs, query_entities: List[Dict]
    ) -> str:
        """Build context string with entity annotations."""
        MAX_CHARS = 8_000
        parts = []
        total = 0
        entity_label_map = {
            e.get("normalized", e.get("text", "")).lower(): e.get("label", "")
            for e in query_entities
        }

        for doc in docs:
            doc_entities = self.entity_index.get_abstract_entities(doc.pmid)
            entity_summary = ""
            if doc_entities:
                diseases  = [e["text"] for e in doc_entities if e.get("label") == "DISEASE"][:3]
                chemicals = [e["text"] for e in doc_entities if e.get("label") == "CHEMICAL"][:3]
                genes     = [e["text"] for e in doc_entities if e.get("label") == "GENE"][:3]
                parts_summary = []
                if diseases:  parts_summary.append(f"Diseases: {', '.join(diseases)}")
                if chemicals: parts_summary.append(f"Chemicals: {', '.join(chemicals)}")
                if genes:     parts_summary.append(f"Genes: {', '.join(genes)}")
                entity_summary = " | ".join(parts_summary)

            entry = (
                f"[Document {doc.rank}]\n"
                f"PMID: {doc.pmid}\n"
                f"Title: {doc.title}\n"
                f"Topic: {doc.topic}\n"
                f"Entities: {entity_summary or 'N/A'}\n"
                f"Score: {doc.score:.3f}\n"
                f"Abstract: {doc.abstract[:600]}\n"
                f"---"
            )
            if total + len(entry) > MAX_CHARS:
                break
            parts.append(entry)
            total += len(entry)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM generation (same as base RAG engine)
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, context: str):
        import os
        provider = "extractive"
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"

        system = (
            "You are a clinical research assistant. Answer based ONLY on the provided "
            "abstracts. Cite claims with [PMID:XXXXX]. Flag uncertainty explicitly."
        )

        if provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic()
                msg = client.messages.create(
                    model=self.model, max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
                )
                return msg.content[0].text, self.model, False
            except Exception as e:
                logger.warning("Anthropic failed: %s", e)

        if provider == "openai":
            try:
                import openai
                client = openai.OpenAI()
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                    ],
                    max_tokens=1024,
                )
                return resp.choices[0].message.content, "gpt-4o-mini", False
            except Exception as e:
                logger.warning("OpenAI failed: %s", e)

        # Extractive fallback
        sentences = [
            l.strip() for l in context.split("\n")
            if len(l.strip()) > 60 and not l.startswith(("[", "PMID", "Title", "---", "Score", "Topic", "Entities"))
        ]
        q_words = set(question.lower().split())
        scored = sorted(
            [(len(q_words & set(s.lower().split())), s) for s in sentences],
            reverse=True,
        )
        top = [s for _, s in scored[:3]]
        answer = "Based on retrieved literature:\n\n" + "\n\n".join(top)
        answer += "\n\n[Note: Extractive summary — LLM unavailable.]"
        return answer, "extractive", True

    def _extract_citations(self, answer: str, docs) -> List[Dict]:
        import re
        cited = set(re.findall(r"PMID:(\d+)", answer))
        return [
            {"rank": d.rank, "pmid": d.pmid, "title": d.title, "score": d.score}
            for d in docs
            if d.pmid in cited or not cited
        ]

    @staticmethod
    def _assess_confidence(mean_score, n_retrieved, fallback):
        if fallback: return "low"
        if mean_score >= 0.55 and n_retrieved >= 4: return "high"
        if mean_score >= 0.35 and n_retrieved >= 2: return "medium"
        return "low"

    def _empty_response(self, question, query_entities, t0):
        return NERRAGResponse(
            query=question, answer="No relevant abstracts found.",
            citations=[], retrieved_docs=[],
            n_retrieved=0, n_after_entity_filter=0, n_context_docs=0,
            query_entities=query_entities, mean_retrieval_score=0.0,
            confidence="low", latency_ms=int((time.time()-t0)*1000),
            model_used="none", fallback_used=True, entity_filter_applied=False,
        )
