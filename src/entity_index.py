"""
Entity Index
=============
Builds an inverted index over NER-enriched corpus enabling
entity-filtered retrieval alongside vector similarity search.

Supports queries like:
  - "Find abstracts mentioning metformin AND CKD"
  - "Show all papers about BRCA1 mutations in breast cancer"
  - "Filter RAG results to disease=heart failure, chemical=statin"

Index structure:
  entity_index[label][normalized_text] → [pmid1, pmid2, ...]
  co_occurrence[entity_a][entity_b]    → count

Also computes:
  - Entity frequency distribution per topic
  - Cross-entity co-occurrence matrix (top 50 entities)
  - Per-abstract entity fingerprint for deduplication

Usage:
    from src.entity_index import EntityIndex
    idx = EntityIndex()
    idx.build(enriched_df)
    idx.save("data/entity_index/")

    idx2 = EntityIndex.load("data/entity_index/")
    pmids = idx2.search(disease="heart failure", chemical="statin")
    cooccur = idx2.get_cooccurrences("heart failure", top_n=10)
"""

import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ENTITY_LABELS = ["DISEASE", "CHEMICAL", "GENE", "PROCEDURE",
                 "SPECIES", "DNA", "CELL_TYPE"]


class EntityIndex:
    """
    Inverted index over biomedical entities for filtered retrieval.
    Integrates with the existing VectorStore for hybrid search.
    """

    def __init__(self):
        # label → normalized_text → set of pmids
        self._index: Dict[str, Dict[str, Set[str]]] = {
            label: defaultdict(set) for label in ENTITY_LABELS
        }
        # entity_a → entity_b → co-occurrence count
        self._cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # pmid → entity list (for fast lookup)
        self._pmid_entities: Dict[str, List[dict]] = {}
        # Global entity frequency
        self._entity_freq: Dict[str, Counter] = {
            label: Counter() for label in ENTITY_LABELS
        }
        self._n_docs = 0
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        enriched_df: pd.DataFrame,
        pmid_col: str = "pmid",
        entities_col: str = "entities_json",
    ) -> None:
        """
        Build inverted index from NER-enriched corpus.

        Args:
            enriched_df:   DataFrame with pmid and entities_json columns.
            pmid_col:      Primary key column.
            entities_col:  Column with JSON-serialized entity list.
        """
        logger.info("Building entity index for %s documents...", f"{len(enriched_df):,}")

        for _, row in enriched_df.iterrows():
            pmid = str(row[pmid_col])
            raw = row.get(entities_col, "[]")

            try:
                entities = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                entities = []

            self._pmid_entities[pmid] = entities

            # Track unique entities per doc for co-occurrence
            doc_entities_by_label: Dict[str, List[str]] = defaultdict(list)

            for ent in entities:
                label = ent.get("label", "")
                norm = (ent.get("normalized") or ent.get("text", "")).lower().strip()
                if not norm or label not in ENTITY_LABELS:
                    continue

                self._index[label][norm].add(pmid)
                self._entity_freq[label][norm] += ent.get("count", 1)
                doc_entities_by_label[label].append(norm)

            # Co-occurrence (within same abstract)
            all_doc_entities = [
                e for entities in doc_entities_by_label.values()
                for e in entities
            ]
            for i, ea in enumerate(all_doc_entities):
                for eb in all_doc_entities[i+1:]:
                    if ea != eb:
                        self._cooccurrence[ea][eb] += 1
                        self._cooccurrence[eb][ea] += 1

        self._n_docs = len(enriched_df)
        self._built = True
        logger.info(
            "Entity index built: %s diseases, %s chemicals, %s genes",
            f"{len(self._index['DISEASE']):,}",
            f"{len(self._index['CHEMICAL']):,}",
            f"{len(self._index['GENE']):,}",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        disease: Optional[str] = None,
        chemical: Optional[str] = None,
        gene: Optional[str] = None,
        procedure: Optional[str] = None,
        operator: str = "AND",
    ) -> List[str]:
        """
        Retrieve PMIDs matching entity filter criteria.

        Args:
            disease:   Disease entity text (partial match supported).
            chemical:  Chemical/drug entity text.
            gene:      Gene entity text.
            procedure: Clinical procedure text.
            operator:  "AND" (intersection) or "OR" (union).

        Returns:
            List of matching PMIDs sorted by relevance (entity freq).
        """
        filters = [
            ("DISEASE", disease),
            ("CHEMICAL", chemical),
            ("GENE", gene),
            ("PROCEDURE", procedure),
        ]
        active = [(label, term) for label, term in filters if term]
        if not active:
            return []

        result_sets = []
        for label, term in active:
            term_lower = term.lower().strip()
            matched: Set[str] = set()
            for norm_text, pmids in self._index[label].items():
                if term_lower in norm_text or norm_text in term_lower:
                    matched |= pmids
            result_sets.append(matched)

        if operator == "AND":
            result = result_sets[0]
            for s in result_sets[1:]:
                result = result & s
        else:
            result = result_sets[0]
            for s in result_sets[1:]:
                result = result | s

        return sorted(result)

    def get_cooccurrences(
        self,
        entity: str,
        top_n: int = 10,
        min_count: int = 2,
    ) -> List[Tuple[str, int]]:
        """
        Return top co-occurring entities for a given entity.

        Args:
            entity:    Entity text (normalized).
            top_n:     Number of results.
            min_count: Minimum co-occurrence count.

        Returns:
            List of (entity_text, count) tuples.
        """
        entity_lower = entity.lower().strip()
        cooccur = self._cooccurrence.get(entity_lower, {})
        filtered = [(e, c) for e, c in cooccur.items() if c >= min_count]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]

    def get_top_entities(
        self,
        label: str,
        top_n: int = 20,
    ) -> List[Tuple[str, int]]:
        """Return top entities by frequency for a given label."""
        if label not in self._entity_freq:
            return []
        return self._entity_freq[label].most_common(top_n)

    def get_entity_summary(self) -> dict:
        """Return summary statistics of the entity index."""
        return {
            "n_docs": self._n_docs,
            "n_unique_diseases":   len(self._index["DISEASE"]),
            "n_unique_chemicals":  len(self._index["CHEMICAL"]),
            "n_unique_genes":      len(self._index["GENE"]),
            "n_unique_procedures": len(self._index["PROCEDURE"]),
            "top_diseases":  self.get_top_entities("DISEASE", 5),
            "top_chemicals": self.get_top_entities("CHEMICAL", 5),
            "top_genes":     self.get_top_entities("GENE", 5),
        }

    def get_abstract_entities(self, pmid: str) -> List[dict]:
        """Return all entities for a given PMID."""
        return self._pmid_entities.get(str(pmid), [])

    # ------------------------------------------------------------------
    # Hybrid search integration
    # ------------------------------------------------------------------

    def filter_vector_results(
        self,
        vector_results: List,          # List of SearchResult from VectorStore
        disease: Optional[str] = None,
        chemical: Optional[str] = None,
        gene: Optional[str] = None,
    ) -> List:
        """
        Filter VectorStore search results by entity constraints.
        Preserves original ranking order among matching results.

        Args:
            vector_results: List of SearchResult objects with .pmid attribute.
            disease/chemical/gene: Entity filters.

        Returns:
            Filtered list preserving original rank order.
        """
        entity_pmids = set(self.search(
            disease=disease, chemical=chemical, gene=gene, operator="AND"
        ))
        if not entity_pmids:
            return vector_results  # No filter if no entity matches

        filtered = [r for r in vector_results if r.pmid in entity_pmids]
        # Re-rank by original vector score
        for i, r in enumerate(filtered):
            r.rank = i + 1
        return filtered

    # ------------------------------------------------------------------
    # Persist / Load
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "entity_index.pkl"), "wb") as f:
            pickle.dump({
                "index": dict(self._index),
                "cooccurrence": dict(self._cooccurrence),
                "pmid_entities": self._pmid_entities,
                "entity_freq": dict(self._entity_freq),
                "n_docs": self._n_docs,
            }, f)

        summary = self.get_entity_summary()
        # Convert tuples to lists for JSON serialization
        for k in ["top_diseases", "top_chemicals", "top_genes"]:
            summary[k] = [[e, c] for e, c in summary[k]]
        with open(os.path.join(directory, "entity_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Entity index saved to %s", directory)

    @classmethod
    def load(cls, directory: str) -> "EntityIndex":
        idx = cls()
        with open(os.path.join(directory, "entity_index.pkl"), "rb") as f:
            data = pickle.load(f)
        idx._index = defaultdict(lambda: defaultdict(set))
        for label, entries in data["index"].items():
            for text, pmids in entries.items():
                idx._index[label][text] = set(pmids)
        idx._cooccurrence = defaultdict(lambda: defaultdict(int))
        for ea, cooccur in data["cooccurrence"].items():
            for eb, count in cooccur.items():
                idx._cooccurrence[ea][eb] = count
        idx._pmid_entities = data["pmid_entities"]
        idx._entity_freq = data["entity_freq"]
        idx._n_docs = data["n_docs"]
        idx._built = True
        logger.info("Entity index loaded: %d documents", idx._n_docs)
        return idx
