# query_knowledge_graph.py
from neo4j import GraphDatabase
import json

class KnowledgeGraphQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def find_treatments_for_disease(self, disease_name):
        """Find drugs that treat a specific disease"""
        with self.driver.session() as session:
            query = """
            MATCH (dr:Drug)-[r:TREATS]->(d:Disease {name: $disease})
            RETURN dr.name as drug, r.evidence_count as evidence
            ORDER BY r.evidence_count DESC
            """
            result = session.run(query, disease=disease_name.lower())
            return [dict(record) for record in result]
    
    def find_associated_genes(self, disease_name):
        """Find genes associated with a disease"""
        with self.driver.session() as session:
            query = """
            MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(d:Disease {name: $disease})
            RETURN g.name as gene, r.evidence_count as evidence
            ORDER BY r.evidence_count DESC
            LIMIT 20
            """
            result = session.run(query, disease=disease_name.lower())
            return [dict(record) for record in result]
    
    def find_drug_targets(self, drug_name):
        """Find genes targeted by a drug"""
        with self.driver.session() as session:
            query = """
            MATCH (dr:Drug {name: $drug})-[r:TARGETS]->(g:Gene)
            RETURN g.name as gene, r.evidence_count as evidence
            ORDER BY r.evidence_count DESC
            LIMIT 20
            """
            result = session.run(query, drug=drug_name.lower())
            return [dict(record) for record in result]
    
    def find_disease_network(self, disease_name):
        """Get full network around a disease (drugs, genes, articles)"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease {name: $disease})
            OPTIONAL MATCH (dr:Drug)-[t:TREATS]->(d)
            OPTIONAL MATCH (g:Gene)-[a:ASSOCIATED_WITH]->(d)
            OPTIONAL MATCH (art:Article)-[m:MENTIONS_DISEASE]->(d)
            RETURN d, 
                   collect(DISTINCT {drug: dr.name, evidence: t.evidence_count}) as treatments,
                   collect(DISTINCT {gene: g.name, evidence: a.evidence_count})[0..10] as genes,
                   count(DISTINCT art) as article_count
            """
            result = session.run(query, disease=disease_name.lower())
            return dict(result.single())
    
    def find_similar_diseases(self, disease_name, limit=5):
        """Find diseases with similar drug/gene profiles"""
        with self.driver.session() as session:
            query = """
            MATCH (d1:Disease {name: $disease})
            MATCH (d1)<-[:TREATS]-(dr:Drug)-[:TREATS]->(d2:Disease)
            WHERE d1 <> d2
            WITH d2, count(DISTINCT dr) as shared_drugs
            ORDER BY shared_drugs DESC
            LIMIT $limit
            RETURN d2.name as disease, shared_drugs
            """
            result = session.run(query, disease=disease_name.lower(), limit=limit)
            return [dict(record) for record in result]
    
    def get_top_diseases(self, limit=10):
        """Get most mentioned diseases"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease)
            RETURN d.name as disease, d.mention_count as mentions
            ORDER BY d.mention_count DESC
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
    
    def get_top_drugs(self, limit=10):
        """Get most mentioned drugs"""
        with self.driver.session() as session:
            query = """
            MATCH (dr:Drug)
            RETURN dr.name as drug, dr.mention_count as mentions
            ORDER BY dr.mention_count DESC
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
    
    def search_drug_disease_path(self, drug_name, disease_name):
        """Find shortest path between drug and disease through genes"""
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (dr:Drug {name: $drug})-[*..3]-(d:Disease {name: $disease})
            )
            RETURN path
            """
            result = session.run(query, drug=drug_name.lower(), disease=disease_name.lower())
            records = list(result)
            if records:
                return "Path found"
            return "No path found"


def run_example_queries():
    """Run example queries and display results"""
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "MedicalLiterature"
    
    kg = KnowledgeGraphQuery(URI, USER, PASSWORD)
    
    print("="*60)
    print("MEDICAL KNOWLEDGE GRAPH - SAMPLE QUERIES")
    print("="*60)
    
    # Query 1: Top diseases
    print("\n1. TOP 10 MOST MENTIONED DISEASES:")
    print("-" * 60)
    diseases = kg.get_top_diseases(10)
    for d in diseases:
        print(f"  {d['disease']:30s}: {d['mentions']:6,} mentions")
    
    # Query 2: Top drugs
    print("\n2. TOP 10 MOST MENTIONED DRUGS:")
    print("-" * 60)
    drugs = kg.get_top_drugs(10)
    for d in drugs:
        print(f"  {d['drug']:30s}: {d['mentions']:6,} mentions")
    
    # Query 3: Treatments for diabetes
    print("\n3. TREATMENTS FOR DIABETES:")
    print("-" * 60)
    treatments = kg.find_treatments_for_disease('diabetes')
    if treatments:
        for t in treatments:
            print(f"  {t['drug']:30s}: {t['evidence']:3d} articles")
    else:
        print("  No treatments found with >= 5 co-occurrences")
    
    # Query 4: Genes associated with cancer
    print("\n4. TOP GENES ASSOCIATED WITH CANCER:")
    print("-" * 60)
    genes = kg.find_associated_genes('cancer')
    for g in genes[:10]:
        print(f"  {g['gene']:30s}: {g['evidence']:3d} articles")
    
    # Query 5: Targets of insulin
    print("\n5. GENES TARGETED BY INSULIN:")
    print("-" * 60)
    targets = kg.find_drug_targets('insulin')
    if targets:
        for t in targets[:10]:
            print(f"  {t['gene']:30s}: {t['evidence']:3d} articles")
    else:
        print("  No targets found with >= 3 co-occurrences")
    
    # Query 6: Disease network for diabetes
    print("\n6. DIABETES DISEASE NETWORK:")
    print("-" * 60)
    network = kg.find_disease_network('diabetes')
    print(f"  Articles mentioning diabetes: {network['article_count']}")
    print(f"  Treatments found: {len([t for t in network['treatments'] if t['drug']])}")
    print(f"  Associated genes: {len([g for g in network['genes'] if g['gene']])}")
    
    # Query 7: Similar diseases to diabetes
    print("\n7. DISEASES SIMILAR TO DIABETES (shared drugs):")
    print("-" * 60)
    similar = kg.find_similar_diseases('diabetes', 5)
    for s in similar:
        print(f"  {s['disease']:30s}: {s['shared_drugs']:2d} shared drugs")
    
    kg.close()
    print("\n" + "="*60)
    print("✓ Query examples complete!")
    print("="*60)


if __name__ == "__main__":
    run_example_queries()