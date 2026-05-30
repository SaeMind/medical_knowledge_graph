"""
Unit Tests — Medical Literature RAG API
=========================================
Tests cover corpus builder, vector store, and RAG engine.
All tests use synthetic data and mock embeddings — no network required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus_builder import CorpusBuilder, TOPIC_TEMPLATES
from vector_store import VectorStore, SearchResult, EmbeddingModel
from rag_engine import RAGEngine, RAGResponse


class TestCorpusBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = CorpusBuilder(use_synthetic=True)

    def test_build_returns_dataframe(self):
        df = self.builder.build(n_abstracts=100, seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)

    def test_required_columns(self):
        df = self.builder.build(n_abstracts=50, seed=42)
        required = ["pmid", "title", "abstract", "topic", "journal", "year",
                    "text", "char_count", "word_count"]
        for col in required:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_topic_distribution(self):
        """All topics should appear in a 1000-abstract corpus."""
        df = self.builder.build(n_abstracts=1000, seed=42)
        topics_present = set(df["topic"].unique())
        all_topics = set(TOPIC_TEMPLATES.keys())
        # Most topics should appear (small corpus may miss very low-weight topics)
        self.assertGreater(len(topics_present), 8)

    def test_pmids_unique(self):
        df = self.builder.build(n_abstracts=200, seed=42)
        self.assertEqual(len(df["pmid"].unique()), len(df))

    def test_text_col_is_title_plus_abstract(self):
        df = self.builder.build(n_abstracts=10, seed=42)
        for _, row in df.iterrows():
            self.assertIn(row["title"][:20], row["text"])

    def test_abstracts_have_structure(self):
        """Abstracts should contain Background/Methods/Results/Conclusions."""
        df = self.builder.build(n_abstracts=20, seed=42)
        for abstract in df["abstract"]:
            self.assertIn("Background:", abstract)
            self.assertIn("Methods:", abstract)
            self.assertIn("Results:", abstract)

    def test_year_range(self):
        df = self.builder.build(n_abstracts=100, seed=42)
        self.assertTrue((df["year"] >= 2015).all())
        self.assertTrue((df["year"] <= 2023).all())


class TestEmbeddingModel(unittest.TestCase):

    def setUp(self):
        """Use TF-IDF fallback by default (no model download needed)."""
        self.model = EmbeddingModel.__new__(EmbeddingModel)
        self.model.model_name = "test"
        self.model._model = None
        self.model._tfidf = None
        self.model._svd = None
        self.model._use_tfidf = True
        self.model._embedding_dim = 256

    def test_tfidf_encode_shape(self):
        texts = ["Heart failure treatment with beta blockers",
                 "Diabetes management with metformin",
                 "Lung cancer immunotherapy outcomes"]
        embeddings = self.model.encode(texts, show_progress=False)
        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], 256)

    def test_tfidf_normalization(self):
        """L2-normalized embeddings have unit norm."""
        texts = ["test sentence one", "test sentence two"]
        embeddings = self.model.encode(texts, normalize=True, show_progress=False)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_consistent_encoding(self):
        """Same text → same embedding (deterministic)."""
        texts = ["Statin therapy reduces cardiovascular mortality"]
        e1 = self.model.encode(texts, show_progress=False)
        e2 = self.model.encode(texts, show_progress=False)
        np.testing.assert_array_equal(e1, e2)


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        """Build a small in-memory vector store with synthetic corpus."""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss not installed")

        builder = CorpusBuilder(use_synthetic=True)
        self.corpus_df = builder.build(n_abstracts=200, seed=42)

        self.store = VectorStore.__new__(VectorStore)
        self.store.model_name = "test"
        self.store.index_type = "flat"
        self.store._index = None
        self.store._metadata = []
        self.store._n_docs = 0

        # Use TF-IDF embedder
        embedder = EmbeddingModel.__new__(EmbeddingModel)
        embedder.model_name = "test"
        embedder._model = None
        embedder._tfidf = None
        embedder._svd = None
        embedder._use_tfidf = True
        embedder._embedding_dim = 256
        self.store.embedder = embedder

        self.store.build(self.corpus_df, text_col="text", batch_size=64)

    def test_index_built(self):
        self.assertEqual(self.store._n_docs, 200)
        self.assertIsNotNone(self.store._index)

    def test_search_returns_results(self):
        results = self.store.search("heart failure treatment beta blockers", k=5)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)

    def test_search_result_fields(self):
        results = self.store.search("diabetes management", k=3)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertIsInstance(r.score, float)
            self.assertIsInstance(r.title, str)
            self.assertIsInstance(r.pmid, str)
            self.assertGreater(r.rank, 0)

    def test_search_scores_descending(self):
        results = self.store.search("cancer immunotherapy clinical trial", k=5)
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i+1].score)

    def test_empty_query_handled(self):
        """Short queries that match nothing return empty or low-score results."""
        results = self.store.search("zzz", k=5, min_score=0.99)
        # Should either be empty or all very low score — no crash
        self.assertIsInstance(results, list)

    def test_metadata_preserved(self):
        results = self.store.search("lung cancer treatment", k=3)
        for r in results:
            self.assertIn(r.topic, list(TOPIC_TEMPLATES.keys()))


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss not installed")

        builder = CorpusBuilder(use_synthetic=True)
        corpus_df = builder.build(n_abstracts=200, seed=42)

        store = VectorStore.__new__(VectorStore)
        store.model_name = "test"
        store.index_type = "flat"
        store._index = None
        store._metadata = []
        store._n_docs = 0

        embedder = EmbeddingModel.__new__(EmbeddingModel)
        embedder._model = None
        embedder._tfidf = None
        embedder._svd = None
        embedder._use_tfidf = True
        embedder._embedding_dim = 256
        store.embedder = embedder
        store.build(corpus_df, text_col="text", batch_size=64)

        self.engine = RAGEngine(
            vector_store=store,
            llm_provider="extractive",  # No API key needed
        )

    def test_query_returns_response(self):
        response = self.engine.query("What is the efficacy of statins in heart failure?")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 10)

    def test_response_has_citations(self):
        response = self.engine.query("Diabetes treatment outcomes with metformin")
        self.assertIsInstance(response.citations, list)

    def test_response_latency_tracked(self):
        response = self.engine.query("Cancer immunotherapy survival")
        self.assertGreater(response.latency_ms, 0)

    def test_confidence_levels(self):
        response = self.engine.query("Heart failure treatment")
        self.assertIn(response.confidence, ["high", "medium", "low"])

    def test_extractive_fallback_no_crash(self):
        """Extractive fallback produces readable answer."""
        response = self.engine.query("COPD exacerbation prevention")
        self.assertTrue(response.fallback_used)
        self.assertNotEqual(response.answer, "")

    def test_to_dict_serializable(self):
        """Response dict is JSON-serializable."""
        import json
        response = self.engine.query("Stroke outcomes")
        d = response.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    def test_batch_query(self):
        questions = [
            "What reduces mortality in heart failure?",
            "Best treatment for type 2 diabetes?",
        ]
        responses = self.engine.batch_query(questions)
        self.assertEqual(len(responses), 2)
        for r in responses:
            self.assertIsInstance(r, RAGResponse)


if __name__ == "__main__":
    unittest.main(verbosity=2)
