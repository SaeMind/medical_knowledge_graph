"""
SciSpacy Biomedical NER Pipeline
==================================
Extracts named biomedical entities from PubMed abstracts using SciSpacy
models and enriches the corpus with structured entity metadata.

Models (priority order, falls back down the list):
  1. en_ner_bc5cdr_md   — diseases + chemicals (BC5CDR corpus)
  2. en_ner_bionlp13cg_md — cancer genetics entities
  3. en_core_sci_md     — general biomedical entities
  4. Regex fallback     — pattern-based extraction (no model download needed)

Entity types extracted:
  DISEASE    — disease/disorder names (UMLS: T047, T048, T191)
  CHEMICAL   — drugs, chemicals, compounds (UMLS: T116, T121, T195)
  GENE       — genes and gene products (UMLS: T116, T123, T126)
  SPECIES    — organisms (UMLS: T001, T004, T005, T007, T008)
  DNA        — DNA sequences and regions
  CELL_TYPE  — cell types and tissues
  PROCEDURE  — clinical procedures (regex fallback)

Output per abstract:
  entities: [
    {text, label, start, end, kb_id (UMLS CUI if linked), count}
  ]
  entity_counts: {DISEASE: n, CHEMICAL: n, ...}
  top_disease: str | None
  top_chemical: str | None

Usage:
    from src.ner_pipeline import NERPipeline
    pipeline = NERPipeline()
    pipeline.load_model()
    entities = pipeline.extract(abstract_text)

    # Batch processing
    enriched_df = pipeline.enrich_corpus(corpus_df, text_col="abstract")
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

@dataclass
class BioEntity:
    text: str
    label: str          # DISEASE, CHEMICAL, GENE, etc.
    start: int          # Character offset in source text
    end: int
    kb_id: str = ""     # UMLS CUI (if entity linker available)
    normalized: str = ""  # Normalized form
    count: int = 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "kb_id": self.kb_id,
            "normalized": self.normalized or self.text,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Regex fallback patterns (no model required)
# ---------------------------------------------------------------------------

REGEX_PATTERNS = {
    "DISEASE": [
        r"\b(?:cancer|carcinoma|tumor|tumour|lymphoma|leukemia|melanoma|sarcoma)\b",
        r"\b(?:diabetes|hypertension|heart failure|myocardial infarction|stroke)\b",
        r"\b(?:pneumonia|sepsis|COPD|asthma|fibrosis|cirrhosis)\b",
        r"\b(?:Alzheimer|Parkinson|multiple sclerosis|epilepsy|dementia)\b",
        r"\b(?:COVID-19|HIV|infection|syndrome|disorder|disease)\b",
        r"\b[A-Z][a-z]+ (?:disease|syndrome|disorder|cancer)\b",
    ],
    "CHEMICAL": [
        r"\b(?:metformin|insulin|statin|aspirin|warfarin|heparin)\b",
        r"\b(?:atorvastatin|rosuvastatin|lisinopril|losartan|metoprolol)\b",
        r"\b(?:pembrolizumab|nivolumab|bevacizumab|trastuzumab|rituximab)\b",
        r"\b(?:dexamethasone|prednisone|methylprednisolone|hydrocortisone)\b",
        r"\b(?:chemotherapy|immunotherapy|radiotherapy|antibiotics?)\b",
        r"\b[a-z]+(?:mab|nib|zib|vir|ine|ide|ate|one)\b",
    ],
    "GENE": [
        r"\b(?:BRCA[12]|TP53|EGFR|KRAS|BRAF|ALK|ROS1|PD-L1|HER2)\b",
        r"\b(?:IL-[0-9]+|TNF-?α?|IFN-?[αβγ]?|TGF-?β?)\b",
        r"\b(?:mTOR|PI3K|AKT|MEK|ERK|NF-κB|VEGF)\b",
        r"\b[A-Z]{2,6}[0-9]?\b(?= gene| mutation| expression| pathway)",
    ],
    "PROCEDURE": [
        r"\b(?:surgery|resection|transplant|dialysis|chemotherapy|biopsy)\b",
        r"\b(?:MRI|CT scan|PET scan|echocardiogram|endoscopy|colonoscopy)\b",
        r"\b(?:catheterization|angioplasty|stenting|bypass|arthroplasty)\b",
        r"\b(?:randomized controlled trial|RCT|meta-analysis|cohort study)\b",
    ],
}


class RegexNER:
    """Pattern-based NER fallback when scispacy models unavailable."""

    def __init__(self):
        self._patterns = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in REGEX_PATTERNS.items()
        }

    def extract(self, text: str) -> List[BioEntity]:
        entities = []
        seen = set()
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group(0)
                    key = (entity_text.lower(), label)
                    if key not in seen:
                        seen.add(key)
                        entities.append(BioEntity(
                            text=entity_text,
                            label=label,
                            start=match.start(),
                            end=match.end(),
                        ))
        return entities


# ---------------------------------------------------------------------------
# SciSpacy NER Pipeline
# ---------------------------------------------------------------------------

SCISPACY_MODELS = [
    "en_ner_bc5cdr_md",      # BC5CDR: disease + chemical (best for our use case)
    "en_ner_bionlp13cg_md",  # BioNLP13CG: cancer genetics
    "en_core_sci_md",        # General biomedical
    "en_core_sci_sm",        # Small general (last resort)
]


class NERPipeline:
    """
    SciSpacy-based biomedical NER pipeline with entity linking.

    Falls back to regex-based extraction if SciSpacy is unavailable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_entity_linker: bool = False,  # UMLS linking (heavy, optional)
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.use_entity_linker = use_entity_linker
        self.batch_size = batch_size
        self._nlp = None
        self._regex_ner = RegexNER()
        self._use_fallback = False
        self._loaded_model = None

    def load_model(self) -> str:
        """
        Load SciSpacy model. Tries models in priority order.
        Falls back to regex if none available.

        Returns:
            Name of loaded model or "regex_fallback"
        """
        try:
            import spacy
        except ImportError:
            logger.warning("spacy not installed — using regex fallback")
            self._use_fallback = True
            return "regex_fallback"

        # Try requested model first, then priority list
        models_to_try = (
            [self.model_name] if self.model_name else []
        ) + SCISPACY_MODELS

        for model in models_to_try:
            try:
                import spacy
                self._nlp = spacy.load(model)
                self._loaded_model = model
                logger.info("Loaded SciSpacy model: %s", model)

                # Optional: add UMLS entity linker
                if self.use_entity_linker:
                    try:
                        self._nlp.add_pipe(
                            "scispacy_linker",
                            config={
                                "resolve_abbreviations": True,
                                "linker_name": "umls",
                                "max_entities_per_mention": 1,
                            }
                        )
                        logger.info("UMLS entity linker loaded")
                    except Exception as e:
                        logger.warning("Entity linker unavailable: %s", e)

                return model

            except (OSError, Exception) as e:
                logger.debug("Model %s unavailable: %s", model, e)
                continue

        logger.warning(
            "No SciSpacy model available. Install with:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz\n"
            "Using regex fallback."
        )
        self._use_fallback = True
        return "regex_fallback"

    def extract(self, text: str) -> List[BioEntity]:
        """
        Extract biomedical entities from a single text.

        Args:
            text: Raw text (abstract, sentence, etc.)

        Returns:
            List of BioEntity objects.
        """
        if not text or not text.strip():
            return []

        if self._use_fallback:
            return self._regex_ner.extract(text)

        if self._nlp is None:
            self.load_model()

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            kb_id = ""
            if hasattr(ent, "kb_ents") and ent.kb_ents:
                kb_id = ent.kb_ents[0][0]  # Top UMLS CUI

            entities.append(BioEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                kb_id=kb_id,
                normalized=ent.text.lower().strip(),
            ))

        return entities

    def extract_batch(
        self,
        texts: List[str],
        progress_interval: int = 1000,
    ) -> List[List[BioEntity]]:
        """
        Extract entities from a list of texts using spacy pipe for efficiency.
        """
        if self._use_fallback:
            return [self._regex_ner.extract(t) for t in texts]

        if self._nlp is None:
            self.load_model()

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            docs = list(self._nlp.pipe(batch))
            for doc in docs:
                batch_entities = []
                for ent in doc.ents:
                    kb_id = ""
                    if hasattr(ent, "kb_ents") and ent.kb_ents:
                        kb_id = ent.kb_ents[0][0]
                    batch_entities.append(BioEntity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        kb_id=kb_id,
                        normalized=ent.text.lower().strip(),
                    ))
                results.append(batch_entities)

            if i > 0 and (i // self.batch_size) % (progress_interval // self.batch_size + 1) == 0:
                logger.info("  NER: %d / %d texts processed", i, len(texts))

        return results

    # ------------------------------------------------------------------
    # Corpus enrichment
    # ------------------------------------------------------------------

    def enrich_corpus(
        self,
        corpus_df: pd.DataFrame,
        text_col: str = "abstract",
        pmid_col: str = "pmid",
        progress_interval: int = 5000,
    ) -> pd.DataFrame:
        """
        Enrich corpus DataFrame with NER entity columns.

        Adds columns:
          entities_json      — JSON list of all entities
          entity_counts_json — JSON dict of label → count
          top_disease        — Most frequent disease entity
          top_chemical       — Most frequent chemical entity
          top_gene           — Most frequent gene entity
          n_entities         — Total entity count
          has_disease        — Binary flag
          has_chemical       — Binary flag
          has_gene           — Binary flag

        Args:
            corpus_df:         Input DataFrame.
            text_col:          Column to run NER on.
            pmid_col:          ID column (for logging).
            progress_interval: Log every N records.

        Returns:
            corpus_df with NER columns added.
        """
        logger.info(
            "Running NER enrichment on %s abstracts (model: %s)...",
            f"{len(corpus_df):,}",
            self._loaded_model or "not loaded",
        )

        if self._nlp is None and not self._use_fallback:
            self.load_model()

        texts = corpus_df[text_col].fillna("").tolist()
        all_entity_lists = self.extract_batch(texts, progress_interval)

        # Build output columns
        entities_json = []
        entity_counts_json = []
        top_diseases = []
        top_chemicals = []
        top_genes = []
        n_entities_list = []
        has_disease = []
        has_chemical = []
        has_gene = []

        for entity_list in all_entity_lists:
            # Deduplicate by (text_lower, label), keep count
            counter: Dict[Tuple[str, str], BioEntity] = {}
            for ent in entity_list:
                key = (ent.normalized or ent.text.lower(), ent.label)
                if key in counter:
                    counter[key].count += 1
                else:
                    counter[key] = BioEntity(
                        text=ent.text, label=ent.label,
                        start=ent.start, end=ent.end,
                        kb_id=ent.kb_id, normalized=ent.normalized,
                        count=1,
                    )

            deduped = list(counter.values())
            label_counts: Dict[str, int] = {}
            for ent in deduped:
                label_counts[ent.label] = label_counts.get(ent.label, 0) + ent.count

            # Top entity per type
            def top_by_label(label: str) -> Optional[str]:
                candidates = [e for e in deduped if e.label == label]
                if not candidates:
                    return None
                return max(candidates, key=lambda e: e.count).text

            entities_json.append(json.dumps([e.to_dict() for e in deduped]))
            entity_counts_json.append(json.dumps(label_counts))
            top_diseases.append(top_by_label("DISEASE"))
            top_chemicals.append(top_by_label("CHEMICAL"))
            top_genes.append(top_by_label("GENE"))
            n_entities_list.append(len(deduped))
            has_disease.append(int("DISEASE" in label_counts))
            has_chemical.append(int("CHEMICAL" in label_counts))
            has_gene.append(int("GENE" in label_counts))

        enriched = corpus_df.copy()
        enriched["entities_json"]      = entities_json
        enriched["entity_counts_json"] = entity_counts_json
        enriched["top_disease"]        = top_diseases
        enriched["top_chemical"]       = top_chemicals
        enriched["top_gene"]           = top_genes
        enriched["n_entities"]         = n_entities_list
        enriched["has_disease"]        = has_disease
        enriched["has_chemical"]       = has_chemical
        enriched["has_gene"]           = has_gene

        logger.info(
            "NER complete. Abstracts with ≥1 disease: %s (%.1f%%), "
            "chemical: %s (%.1f%%), gene: %s (%.1f%%)",
            f"{sum(has_disease):,}", sum(has_disease) / len(has_disease) * 100,
            f"{sum(has_chemical):,}", sum(has_chemical) / len(has_chemical) * 100,
            f"{sum(has_gene):,}", sum(has_gene) / len(has_gene) * 100,
        )
        return enriched

    @property
    def model_info(self) -> dict:
        return {
            "model": self._loaded_model or "regex_fallback",
            "use_fallback": self._use_fallback,
            "use_entity_linker": self.use_entity_linker,
            "batch_size": self.batch_size,
        }
