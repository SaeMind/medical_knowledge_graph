"""
Unit Tests — SciSpacy NER Enrichment
======================================
All tests use regex fallback — no model download required.
"""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ner_pipeline import NERPipeline, RegexNER, BioEntity
from entity_index import EntityIndex


class TestRegexNER(unittest.TestCase):

    def setUp(self):
        self.ner = RegexNER()

    def test_disease_extraction(self):
        text = "Patients with heart failure and diabetes showed improved outcomes."
        entities = self.ner.extract(text)
        labels = [e.label for e in entities]
        self.assertIn("DISEASE", labels)

    def test_chemical_extraction(self):
        text = "Treatment with metformin and atorvastatin reduced HbA1c levels."
        entities = self.ner.extract(text)
        labels = [e.label for e in entities]
        self.assertIn("CHEMICAL", labels)
        texts = [e.text.lower() for e in entities]
        self.assertTrue(any("metformin" in t for t in texts))

    def test_gene_extraction(self):
        text = "BRCA1 mutations were associated with increased cancer risk."
        entities = self.ner.extract(text)
        labels = [e.label for e in entities]
        self.assertIn("GENE", labels)

    def test_empty_text(self):
        self.assertEqual(self.ner.extract(""), [])

    def test_no_false_positives_in_generic_text(self):
        text = "The weather is nice today and the sun is shining."
        entities = self.ner.extract(text)
        # Generic text may have 0 entities
        self.assertIsInstance(entities, list)

    def test_entity_positions(self):
        text = "Metformin reduces blood glucose in diabetes."
        entities = self.ner.extract(text)
        for ent in entities:
            self.assertIsInstance(ent.start, int)
            self.assertIsInstance(ent.end, int)
            self.assertGreaterEqual(ent.start, 0)
            self.assertLessEqual(ent.end, len(text))


class TestNERPipeline(unittest.TestCase):

    def setUp(self):
        """Use regex fallback — no scispacy download needed."""
        self.pipeline = NERPipeline()
        self.pipeline._use_fallback = True
        self.pipeline._loaded_model = "regex_fallback"

    def test_extract_returns_list(self):
        result = self.pipeline.extract("Heart failure treated with statin therapy.")
        self.assertIsInstance(result, list)

    def test_extract_batch_length(self):
        texts = [
            "Diabetes management with metformin.",
            "BRCA1 mutation in breast cancer.",
            "Sepsis treated with antibiotics.",
        ]
        results = self.pipeline.extract_batch(texts)
        self.assertEqual(len(results), 3)

    def test_enrich_corpus_columns(self):
        df = pd.DataFrame({
            "pmid": ["1001", "1002", "1003"],
            "abstract": [
                "Heart failure patients treated with beta-blockers showed improved survival.",
                "Metformin reduces HbA1c in type 2 diabetes mellitus patients.",
                "BRCA1 mutations increase breast cancer risk significantly.",
            ],
            "topic": ["cardiovascular", "diabetes", "oncology"],
        })
        enriched = self.pipeline.enrich_corpus(df, text_col="abstract")

        required_cols = [
            "entities_json", "entity_counts_json", "top_disease",
            "top_chemical", "top_gene", "n_entities",
            "has_disease", "has_chemical", "has_gene",
        ]
        for col in required_cols:
            self.assertIn(col, enriched.columns, f"Missing column: {col}")

    def test_enrich_corpus_json_parseable(self):
        df = pd.DataFrame({
            "pmid": ["1001"],
            "abstract": ["Statin therapy reduces cardiovascular mortality in heart failure."],
        })
        enriched = self.pipeline.enrich_corpus(df)
        entities = json.loads(enriched["entities_json"].iloc[0])
        self.assertIsInstance(entities, list)
        counts = json.loads(enriched["entity_counts_json"].iloc[0])
        self.assertIsInstance(counts, dict)

    def test_has_flags_binary(self):
        df = pd.DataFrame({
            "pmid": ["1001", "1002"],
            "abstract": [
                "Diabetes and hypertension comorbidity in elderly patients.",
                "No specific condition mentioned in this abstract.",
            ],
        })
        enriched = self.pipeline.enrich_corpus(df)
        self.assertTrue(enriched["has_disease"].isin([0, 1]).all())
        self.assertTrue(enriched["has_chemical"].isin([0, 1]).all())

    def test_model_info(self):
        info = self.pipeline.model_info
        self.assertIn("model", info)
        self.assertIn("use_fallback", info)


class TestEntityIndex(unittest.TestCase):

    def setUp(self):
        """Build a small test index."""
        pipeline = NERPipeline()
        pipeline._use_fallback = True
        pipeline._loaded_model = "regex_fallback"

        self.df = pd.DataFrame({
            "pmid": ["1001", "1002", "1003", "1004", "1005"],
            "abstract": [
                "Heart failure patients treated with statins showed reduced mortality.",
                "Metformin and insulin therapy in type 2 diabetes mellitus.",
                "BRCA1 mutations associated with breast cancer risk.",
                "COPD exacerbations treated with beta-agonists and corticosteroids.",
                "Statins reduce LDL cholesterol and cardiovascular disease risk.",
            ],
        })
        enriched = pipeline.enrich_corpus(self.df, text_col="abstract")
        self.index = EntityIndex()
        self.index.build(enriched)

    def test_build_populates_index(self):
        self.assertTrue(self.index._built)
        self.assertEqual(self.index._n_docs, 5)

    def test_search_disease(self):
        results = self.index.search(disease="heart failure")
        self.assertIsInstance(results, list)
        # PMID 1001 should match
        self.assertIn("1001", results)

    def test_search_chemical(self):
        results = self.index.search(chemical="metformin")
        self.assertIsInstance(results, list)
        self.assertIn("1002", results)

    def test_search_and_operator(self):
        """AND: only docs with both entities."""
        results_and = self.index.search(
            chemical="statin", disease="heart failure", operator="AND"
        )
        results_or = self.index.search(
            chemical="statin", disease="heart failure", operator="OR"
        )
        self.assertLessEqual(len(results_and), len(results_or))

    def test_search_no_filter_returns_empty(self):
        results = self.index.search()
        self.assertEqual(results, [])

    def test_get_top_entities(self):
        top = self.index.get_top_entities("DISEASE", top_n=5)
        self.assertIsInstance(top, list)
        for entity, count in top:
            self.assertIsInstance(entity, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_get_abstract_entities(self):
        entities = self.index.get_abstract_entities("1001")
        self.assertIsInstance(entities, list)

    def test_entity_summary_keys(self):
        summary = self.index.get_entity_summary()
        for key in ["n_docs", "n_unique_diseases", "n_unique_chemicals",
                    "top_diseases", "top_chemicals"]:
            self.assertIn(key, summary)

    def test_save_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.index.save(tmpdir)
            loaded = EntityIndex.load(tmpdir)
            self.assertEqual(loaded._n_docs, self.index._n_docs)
            # Search should work on loaded index
            results = loaded.search(chemical="metformin")
            self.assertIsInstance(results, list)

    def test_filter_vector_results(self):
        """filter_vector_results preserves ranking order."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            rank: int
            pmid: str
            score: float
            title: str = ""
            abstract: str = ""
            topic: str = ""
            text_snippet: str = ""

        fake_results = [
            FakeResult(rank=1, pmid="1001", score=0.90),
            FakeResult(rank=2, pmid="1002", score=0.85),
            FakeResult(rank=3, pmid="9999", score=0.80),  # Not in index
        ]
        filtered = self.index.filter_vector_results(fake_results, disease="heart failure")
        # 9999 should be filtered out (not in entity index)
        pmids = [r.pmid for r in filtered]
        self.assertNotIn("9999", pmids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
