# Medical Knowledge Graph & Clinical Question Answering System

A production-grade medical knowledge graph built from 47,628 PubMed articles, featuring automated entity extraction, graph-based knowledge representation, and RAG-powered clinical question answering.

---

## Project Overview

**Domain:** Biomedical Informatics | Clinical Data Science
**Techniques:** Knowledge Graph Engineering, NLP, Information Retrieval, Graph Databases
**Technologies:** Neo4j, Python, PubMed API, NetworkX

---

## Key Achievements

- **Data Scale:** 47,628 peer-reviewed medical articles from PubMed
- **Knowledge Graph:** 57,515 nodes, 120,983 relationships
- **Entity Coverage:** 31 diseases, 40 drugs, 15,791 genes
- **Q&A Accuracy:** 100% on 10-question clinical test set (see Evaluation section)
- **Relationship Extraction:** 156 drug-disease, 4,858 gene-disease, 842 drug-gene connections

---

## Architecture

```
┌─────────────────┐
│  PubMed API     │
│  47K+ Articles  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Entity Extraction Engine       │
│  - Diseases (31 unique)         │
│  - Drugs (40 unique)            │
│  - Genes (15,791 unique)        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Neo4j Knowledge Graph          │
│  - 57K nodes                    │
│  - 121K relationships           │
│  - Co-occurrence analysis       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  RAG-based Q&A System           │
│  - Graph retrieval              │
│  - Context-aware answers        │
│  - Evidence linking             │
└─────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- Neo4j Desktop or Neo4j Server
- 8GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/SaeMind/medical_knowledge_graph.git
cd medical_knowledge_graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Neo4j Setup

**Option 1: Neo4j Desktop**

1. Download from https://neo4j.com/download/
2. Create new database
3. Start database and note credentials

**Option 2: Docker**

```bash
docker run --name neo4j-medical \
    -p7474:7474 -p7687:7687 \
    -e NEO4J_AUTH=neo4j/medical123 \
    neo4j:latest
```

---

## Usage

### 1. Data Collection

```bash
# Collect 47K+ articles from PubMed
python collect_pubmed_data.py
# Output: data/pubmed_articles_all.json
```

### 2. Entity Extraction

```bash
# Extract medical entities (diseases, drugs, genes)
python extract_entities.py
# Output: data/pubmed_entities_extracted.json
```

### 3. Build Knowledge Graph

```bash
# Update credentials in build_knowledge_graph.py, then:
python build_knowledge_graph.py
# Creates: 57K nodes, 121K relationships in Neo4j
```

### 4. Query the Graph

```bash
python query_knowledge_graph.py
python visualize_graph.py
```

### 5. Clinical Q&A

```bash
python clinical_qa_system.py   # Run Q&A examples
python interactive_qa.py       # Interactive Q&A session
python evaluate_qa_system.py   # Evaluate system accuracy
```

---

## Sample Queries

### Find Treatments for a Disease

```cypher
MATCH (dr:Drug)-[r:TREATS]->(d:Disease {name: 'diabetes'})
RETURN dr.name, r.evidence_count
ORDER BY r.evidence_count DESC
```

### Find Associated Genes

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(d:Disease {name: 'cancer'})
RETURN g.name, r.evidence_count
ORDER BY r.evidence_count DESC
LIMIT 20
```

### Disease Similarity Network

```cypher
MATCH (d1:Disease {name: 'diabetes'})
MATCH (d1)<-[:TREATS]-(dr:Drug)-[:TREATS]->(d2:Disease)
WHERE d1 <> d2
WITH d2, count(DISTINCT dr) as shared_drugs
ORDER BY shared_drugs DESC
RETURN d2.name, shared_drugs
```

---

## Project Structure

```
medical_knowledge_graph/
├── data/
│   ├── pubmed_articles_all.json          # Raw articles
│   ├── pubmed_entities_extracted.json    # Extracted entities
│   └── diabetes_articles.csv            # Category-specific data
├── visualizations/
│   ├── diabetes_network.png
│   ├── cancer_network.png
│   └── hypertension_network.png
├── collect_pubmed_data.py                # Data collection
├── extract_entities.py                   # Entity extraction
├── build_knowledge_graph.py              # Graph construction
├── query_knowledge_graph.py              # Example queries
├── visualize_graph.py                    # Visualization
├── clinical_qa_system.py                 # Q&A system
├── interactive_qa.py                     # Interactive interface
├── evaluate_qa_system.py                 # Evaluation metrics
├── PROJECT_SUMMARY.json                  # Structured project metadata
├── requirements.txt
└── README.md
```

---

## Key Results

### Entity Extraction Performance

- **Diseases:** 44,176 mentions across 31 unique entities
- **Drugs:** 6,854 mentions across 40 unique entities
- **Genes:** 82,632 mentions across 15,791 unique entities

### Knowledge Graph Statistics

| Relationship Type | Count |
|---|---|
| MENTIONS_DISEASE | 37,189 |
| MENTIONS_DRUG | 5,987 |
| MENTIONS_GENE | 71,951 |
| TREATS (drug → disease) | 156 |
| ASSOCIATED_WITH (gene → disease) | 4,858 |
| TARGETS (drug → gene) | 842 |
| **Total Relationships** | **120,983** |

### Q&A System Evaluation

- **Test Set:** 10 curated clinical questions spanning drug-disease and gene-disease domains
- **Retrieval Accuracy:** 100% on test set
- **Average Response Time:** <2 seconds
- **Evidence Quality:** Responses linked to PubMed articles with PMIDs

> **Evaluation Note:** The 100% accuracy figure reflects performance on the 10-question evaluation set used for this portfolio demonstration. Production deployment would require a larger, blinded evaluation corpus for statistical validity.

---

## Performance Metrics

| Metric | Value |
|---|---|
| Articles Processed | 47,628 |
| Processing Time | ~45 min |
| Graph Construction Time | ~8 min |
| Storage Size (Neo4j) | ~850 MB |
| Query Response Time | <2 sec |

---

## Clinical Use Cases

1. **Drug Discovery:** Identify candidate drugs for diseases based on gene associations
2. **Treatment Planning:** Find evidence-based treatment options for conditions
3. **Literature Review:** Rapidly identify relevant research on drug-disease relationships
4. **Hypothesis Generation:** Discover novel connections through graph traversal

---

## Technical Highlights

### Scalable Data Pipeline
- Automated PubMed API integration with rate limiting
- Batch processing for 47K+ articles
- Resilient to API failures with retry logic

### Advanced Entity Recognition
- Pattern-based extraction for medical terminology
- Disease suffix detection (-itis, -osis, -emia)
- Drug naming convention recognition (-mab, -nib, -statin)
- Gene symbol extraction with filtering

### Graph-Based Knowledge Representation
- Co-occurrence analysis for relationship extraction
- Evidence-weighted edges (min threshold filtering)
- Multi-hop relationship discovery
- Network centrality analysis

### RAG Architecture
- Graph-based retrieval (not just vector similarity)
- Entity-aware query processing
- Context enrichment from knowledge graph
- Citation linking to original sources

---

## Future Enhancements

- Integrate BioBERT for improved entity extraction
- Add UMLS ontology mapping
- Implement graph neural networks for link prediction
- Deploy as web API with FastAPI
- Add real-time PubMed updates
- Expand to full-text articles (PMC)

---

## Citations & Data Sources

- **PubMed/MEDLINE:** National Library of Medicine
- **Data Access:** NCBI E-utilities API
- **Categories:** Diabetes, Cancer, Cardiovascular, Infectious Disease, Neurology

---

## Skills Demonstrated

- Biomedical NLP & Information Extraction
- Knowledge Graph Engineering (Neo4j)
- Graph Database Query Optimization (Cypher)
- Medical Ontologies & Terminologies
- RAG System Architecture
- Clinical Data Science
- API Integration & Data Pipelines
- Network Analysis & Visualization

---

## Contact

**Andrew Lee**
Clinical Data Science | Biomedical Informatics

- [LinkedIn](https://www.linkedin.com/in/agllee)
- [Portfolio](https://andrew-gihbeom-lee.figma.site/)
- [Email](mailto:gihbeom@gmail.com)
- [GitHub](https://github.com/SaeMind)

---

## License

MIT License — See LICENSE file for details

---

*Built as part of a portfolio demonstrating expertise in clinical informatics, knowledge graphs, and AI-powered medical applications.*
