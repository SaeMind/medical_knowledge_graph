"""
Vector Store — Embedding + FAISS Index
========================================
Converts biomedical abstracts to dense vector embeddings and builds
a FAISS index for approximate nearest-neighbor retrieval.

Embedding strategy (priority order):
  1. SentenceTransformers: 'pritamdeka/S-PubMedBert-MS-MARCO'
     — domain-specific biomedical model fine-tuned for retrieval
  2. SentenceTransformers: 'all-MiniLM-L6-v2' (general, fast)
  3. TF-IDF + SVD fallback (no GPU / no model download required)

FAISS index type:
  - < 10K docs:   IndexFlatL2 (exact, no training required)
  - 10K–1M docs:  IndexIVFFlat (approximate, faster at scale)
  - Similarity:   cosine (via inner product on normalized vectors)

Usage:
    from src.vector_store import VectorStore
    store = VectorStore()
    store.build(corpus_df, text_col="text")
    store.save("data/faiss_index/")

    store2 = VectorStore.load("data/faiss_index/")
    results = store2.search("What is the efficacy of statins in heart failure?", k=5)
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 / S-PubMedBert dimension
CHUNK_SIZE = 512      # Max tokens per chunk


@dataclass
class SearchResult:
    """Single retrieval result."""
    rank: int
    pmid: str
    title: str
    abstract: str
    topic: str
    score: float          # Similarity score (higher = more similar)
    text_snippet: str     # First 300 chars of abstract


class EmbeddingModel:
    """
    Wraps sentence-transformers with graceful fallback to TF-IDF.
    """

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self._svd = None
        self._use_tfidf = False
        self._embedding_dim = EMBEDDING_DIM

    def load(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding dim: %d", self._embedding_dim)
        except (ImportError, Exception) as e:
            logger.warning("SentenceTransformer unavailable (%s) — trying MiniLM fallback", e)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._embedding_dim = 384
                logger.info("Loaded fallback: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.warning("SentenceTransformer unavailable (%s) — using TF-IDF+SVD", e2)
                self._use_tfidf = True
                self._embedding_dim = 256

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to dense vectors.

        Args:
            texts:          List of strings to encode.
            batch_size:     Batch size for transformer inference.
            normalize:      L2-normalize for cosine similarity via dot product.
            show_progress:  Show tqdm progress bar.

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if self._model is None and not self._use_tfidf:
            self.load()

        if self._use_tfidf:
            return self._tfidf_encode(texts, normalize)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def _tfidf_encode(self, texts: List[str], normalize: bool) -> np.ndarray:
        """TF-IDF + TruncatedSVD fallback embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize as sk_normalize

        if self._tfidf is None:
            logger.info("Fitting TF-IDF vectorizer on %d texts...", len(texts))
            self._tfidf = TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )
            self._svd = TruncatedSVD(n_components=256, random_state=42)
            tfidf_matrix = self._tfidf.fit_transform(texts)
            embeddings = self._svd.fit_transform(tfidf_matrix)
        else:
            tfidf_matrix = self._tfidf.transform(texts)
            embeddings = self._svd.transform(tfidf_matrix)

        embeddings = embeddings.astype(np.float32)
        if normalize:
            embeddings = sk_normalize(embeddings, norm="l2")
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class VectorStore:
    """
    FAISS-backed vector store for biomedical abstract retrieval.

    Stores embeddings + metadata in a self-contained directory:
        index.faiss    — FAISS index
        metadata.pkl   — per-document metadata (title, abstract, etc.)
        config.json    — index configuration
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        index_type: str = "auto",  # "flat", "ivf", or "auto"
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.embedder = EmbeddingModel(model_name)
        self._index = None
        self._metadata: List[dict] = []
        self._n_docs = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        corpus_df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 64,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Build FAISS index from corpus DataFrame.

        Args:
            corpus_df:  DataFrame with text_col and metadata columns.
            text_col:   Column containing text to embed.
            batch_size: Encoding batch size.
        """
        import faiss

        logger.info("Building vector store for %s documents...", f"{len(corpus_df):,}")

        # Load model
        self.embedder.load()
        dim = self.embedder.embedding_dim

        # Encode all texts
        texts = corpus_df[text_col].fillna("").tolist()
        logger.info("Encoding %s texts (dim=%d)...", f"{len(texts):,}", dim)
        embeddings = self.embedder.encode(
            texts, batch_size=batch_size, normalize=True, show_progress=True
        )

        # Build FAISS index
        n = len(embeddings)
        if self.index_type == "flat" or (self.index_type == "auto" and n < 10_000):
            logger.info("Building IndexFlatIP (exact, n=%d)", n)
            self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine on L2-normalized
        else:
            # IVF for larger corpora
            n_clusters = min(int(np.sqrt(n)), 256)
            logger.info("Building IndexIVFFlat (n=%d, clusters=%d)", n, n_clusters)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training IVF index...")
            self._index.train(embeddings)
            self._index.nprobe = min(32, n_clusters)

        self._index.add(embeddings)
        self._n_docs = n

        # Store metadata
        meta_cols = [c for c in corpus_df.columns if c != text_col]
        self._metadata = corpus_df[meta_cols].to_dict(orient="records")

        logger.info("Vector store built: %d vectors, dim=%d", self._n_docs, dim)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query:     Natural language query string.
            k:         Number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if self._index is None:
            raise RuntimeError("Vector store not built. Call build() or load() first.")

        query_emb = self.embedder.encode(
            [query], batch_size=1, normalize=True, show_progress=False
        )
        scores, indices = self._index.search(query_emb, k * 2)  # Over-retrieve then filter

        results = []
        seen_pmids = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if float(score) < min_score:
                continue

            meta = self._metadata[idx]
            pmid = str(meta.get("pmid", idx))
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            abstract = str(meta.get("abstract", ""))
            results.append(SearchResult(
                rank=len(results) + 1,
                pmid=pmid,
                title=str(meta.get("title", "")),
                abstract=abstract,
                topic=str(meta.get("topic", "")),
                score=round(float(score), 4),
                text_snippet=abstract[:300] + "..." if len(abstract) > 300 else abstract,
            ))

            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Persist / Load
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save index + metadata to directory."""
        import faiss
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self._metadata, f)

        config = {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "n_docs": self._n_docs,
            "embedding_dim": self.embedder.embedding_dim,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Vector store saved to %s (%d docs)", directory, self._n_docs)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load a saved vector store."""
        import faiss

        config_path = os.path.join(directory, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        store = cls(
            model_name=config["model_name"],
            index_type=config["index_type"],
        )
        store.embedder.load()
        store._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            store._metadata = pickle.load(f)
        store._n_docs = config["n_docs"]

        logger.info("Vector store loaded: %d docs from %s", store._n_docs, directory)
        return store
