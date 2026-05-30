"""
RAG Engine — Retrieval-Augmented Generation
============================================
Combines vector retrieval with LLM-based answer synthesis for
biomedical literature Q&A.

Pipeline per query:
  1. Retrieve top-k relevant abstracts (vector similarity)
  2. Rerank by BM25 + cross-encoder (optional, if model available)
  3. Build structured context window (title + abstract snippets)
  4. Generate grounded answer via LLM (Anthropic / OpenAI / local)
  5. Return answer + citations + confidence score

Grounding strategy:
  - System prompt enforces citation-grounded responses
  - Answer must reference only retrieved abstracts
  - Uncertainty flagged when top-k similarity < threshold
  - Source attribution: PMID + title per cited claim

Usage:
    from src.rag_engine import RAGEngine
    engine = RAGEngine(vector_store=store)
    response = engine.query("What is the mortality benefit of beta-blockers in heart failure?")
    print(response.answer)
    print(response.citations)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

# Maximum context tokens to send to LLM
MAX_CONTEXT_CHARS = 8_000
# Minimum similarity score to include in context (cosine, [0,1])
MIN_RETRIEVAL_SCORE = 0.25
# Number of abstracts to retrieve per query
TOP_K_RETRIEVAL = 8
# Number of abstracts to include in LLM context after reranking
TOP_K_CONTEXT = 5


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    query: str
    answer: str
    citations: List[Dict]          # [{rank, pmid, title, score}]
    retrieved_docs: List[Dict]     # All retrieved docs (pre-LLM)
    n_retrieved: int
    n_context_docs: int
    retrieval_scores: List[float]
    mean_retrieval_score: float
    confidence: str                # "high", "medium", "low"
    latency_ms: int
    model_used: str
    fallback_used: bool            # True if LLM unavailable → extractive answer

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "n_retrieved": self.n_retrieved,
            "n_context_docs": self.n_context_docs,
            "mean_retrieval_score": self.mean_retrieval_score,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
        }


SYSTEM_PROMPT = """You are a clinical research assistant that answers questions 
about biomedical literature. You are given a question and a set of retrieved 
PubMed abstracts as context.

Instructions:
1. Answer the question based ONLY on the provided abstracts.
2. Cite each claim using [PMID:XXXXX] inline.
3. If the abstracts do not contain sufficient information, say so explicitly.
4. Be precise about study designs, sample sizes, and statistical findings.
5. Distinguish between findings from RCTs vs observational studies.
6. Do not fabricate citations or statistics not present in the context.
7. Limit your answer to 3-5 concise paragraphs.
"""


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for biomedical literature Q&A.

    Supports:
      - Anthropic Claude API (primary)
      - OpenAI API (secondary)
      - Extractive fallback (no API key required)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = TOP_K_RETRIEVAL,
        top_k_context: int = TOP_K_CONTEXT,
        min_score: float = MIN_RETRIEVAL_SCORE,
        llm_provider: str = "auto",   # "anthropic", "openai", "extractive", "auto"
        model: str = "claude-sonnet-4-20250514",
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.top_k_context = top_k_context
        self.min_score = min_score
        self.llm_provider = llm_provider
        self.model = model
        self._llm_client = None

    # ------------------------------------------------------------------
    # Primary query method
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """
        Answer a biomedical literature question using RAG.

        Args:
            question: Natural language clinical/research question.

        Returns:
            RAGResponse with grounded answer and citations.
        """
        t0 = time.time()

        # Stage 1: Retrieve
        retrieved = self.vector_store.search(
            question, k=self.top_k, min_score=self.min_score
        )

        if not retrieved:
            return self._empty_response(question, t0)

        scores = [r.score for r in retrieved]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Stage 2: Select context docs
        context_docs = retrieved[:self.top_k_context]

        # Stage 3: Generate answer
        context_str = self._build_context(context_docs)
        answer, model_used, fallback = self._generate_answer(question, context_str)

        # Stage 4: Extract citations from answer
        citations = self._extract_citations(answer, context_docs)

        # Stage 5: Assess confidence
        confidence = self._assess_confidence(mean_score, len(retrieved), fallback)

        latency = int((time.time() - t0) * 1000)

        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations,
            retrieved_docs=[
                {"rank": r.rank, "pmid": r.pmid, "title": r.title,
                 "score": r.score, "topic": r.topic}
                for r in retrieved
            ],
            n_retrieved=len(retrieved),
            n_context_docs=len(context_docs),
            retrieval_scores=scores,
            mean_retrieval_score=round(mean_score, 4),
            confidence=confidence,
            latency_ms=latency,
            model_used=model_used,
            fallback_used=fallback,
        )

    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Process a list of questions and return responses."""
        responses = []
        for i, q in enumerate(questions):
            logger.info("Query %d/%d: %s", i + 1, len(questions), q[:60])
            responses.append(self.query(q))
        return responses

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, docs: List[SearchResult]) -> str:
        """Build structured context string from retrieved documents."""
        parts = []
        total_chars = 0

        for doc in docs:
            entry = (
                f"[Document {doc.rank}]\n"
                f"PMID: {doc.pmid}\n"
                f"Title: {doc.title}\n"
                f"Topic: {doc.topic}\n"
                f"Relevance Score: {doc.score:.3f}\n"
                f"Abstract: {doc.abstract[:600]}\n"
                f"---"
            )
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM answer generation
    # ------------------------------------------------------------------

    def _generate_answer(
        self, question: str, context: str
    ) -> tuple:
        """
        Generate answer using available LLM.
        Returns (answer_str, model_name, is_fallback).
        """
        provider = self._detect_provider()

        if provider == "anthropic":
            try:
                return self._anthropic_generate(question, context)
            except Exception as e:
                logger.warning("Anthropic API failed: %s — falling back", e)

        if provider == "openai":
            try:
                return self._openai_generate(question, context)
            except Exception as e:
                logger.warning("OpenAI API failed: %s — falling back", e)

        # Extractive fallback
        return self._extractive_answer(question, context), "extractive", True

    def _detect_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "extractive"

    def _anthropic_generate(self, question: str, context: str) -> tuple:
        import anthropic
        client = anthropic.Anthropic()
        user_msg = (
            f"Context (retrieved PubMed abstracts):\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Please provide a grounded answer with inline citations."
        )
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text, self.model, False

    def _openai_generate(self, question: str, context: str) -> tuple:
        import openai
        client = openai.OpenAI()
        user_msg = (
            f"Context:\n\n{context}\n\n"
            f"Question: {question}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content, "gpt-4o-mini", False

    @staticmethod
    def _extractive_answer(question: str, context: str) -> str:
        """
        Extractive fallback: returns top 3 most relevant sentences
        from context when no LLM is available.
        """
        sentences = []
        for line in context.split("\n"):
            line = line.strip()
            if (len(line) > 60
                    and not line.startswith("[Document")
                    and not line.startswith("PMID:")
                    and not line.startswith("Title:")
                    and not line.startswith("---")
                    and not line.startswith("Relevance")):
                sentences.append(line)

        # Simple keyword overlap scoring
        q_words = set(question.lower().split())
        scored = []
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(q_words & s_words)
            scored.append((overlap, sent))

        top = sorted(scored, reverse=True)[:3]
        if not top:
            return "Insufficient information found in retrieved abstracts to answer this question."

        answer = "Based on retrieved literature:\n\n"
        answer += "\n\n".join(s for _, s in top)
        answer += "\n\n[Note: This is an extractive summary — LLM synthesis unavailable.]"
        return answer

    # ------------------------------------------------------------------
    # Citation extraction
    # ------------------------------------------------------------------

    def _extract_citations(
        self,
        answer: str,
        context_docs: List[SearchResult],
    ) -> List[Dict]:
        """Extract PMIDs cited in the answer text."""
        import re
        cited_pmids = set(re.findall(r"PMID:(\d+)", answer))
        citations = []
        for doc in context_docs:
            if doc.pmid in cited_pmids or len(cited_pmids) == 0:
                citations.append({
                    "rank": doc.rank,
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "score": doc.score,
                    "topic": doc.topic,
                })
        return citations

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_confidence(
        mean_score: float,
        n_retrieved: int,
        fallback: bool,
    ) -> str:
        if fallback:
            return "low"
        if mean_score >= 0.55 and n_retrieved >= 4:
            return "high"
        if mean_score >= 0.35 and n_retrieved >= 2:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_dataset: List[Dict],
    ) -> Dict:
        """
        Evaluate RAG pipeline on a QA dataset.

        eval_dataset: list of {question, expected_topics, expected_pmids}
        Returns: precision, recall, mean_score, mean_latency
        """
        precisions, recalls, scores, latencies = [], [], [], []

        for item in eval_dataset:
            response = self.query(item["question"])
            retrieved_pmids = {d["pmid"] for d in response.retrieved_docs}
            expected_pmids = set(item.get("expected_pmids", []))

            if expected_pmids:
                tp = len(retrieved_pmids & expected_pmids)
                precision = tp / len(retrieved_pmids) if retrieved_pmids else 0.0
                recall = tp / len(expected_pmids) if expected_pmids else 0.0
                precisions.append(precision)
                recalls.append(recall)

            scores.append(response.mean_retrieval_score)
            latencies.append(response.latency_ms)

        return {
            "n_queries": len(eval_dataset),
            "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "mean_retrieval_score": round(sum(scores) / len(scores), 4),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p90_latency_ms": int(sorted(latencies)[int(len(latencies) * 0.9)]),
        }

    def _empty_response(self, question: str, t0: float) -> RAGResponse:
        return RAGResponse(
            query=question,
            answer="No relevant abstracts found for this query.",
            citations=[],
            retrieved_docs=[],
            n_retrieved=0,
            n_context_docs=0,
            retrieval_scores=[],
            mean_retrieval_score=0.0,
            confidence="low",
            latency_ms=int((time.time() - t0) * 1000),
            model_used="none",
            fallback_used=True,
        )
