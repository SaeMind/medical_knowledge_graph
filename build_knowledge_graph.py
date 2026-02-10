# build_knowledge_graph.py
import json
from neo4j import GraphDatabase
import pandas as pd
from collections import defaultdict
import time

class MedicalKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing data"""
        with self.driver.session() as session:
            # Drop constraints first
            print("Dropping constraints...")
            try:
                constraints = session.run("SHOW CONSTRAINTS")
                for constraint in constraints:
                    constraint_name = constraint.get('name')
                    if constraint_name:
                        session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
            except:
                pass
            
            # Delete all data
            print("Deleting all data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"✓ Database cleared (nodes remaining: {count})")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT article_pmid IF NOT EXISTS FOR (a:Article) REQUIRE a.pmid IS UNIQUE",
                "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
                "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE",
                "CREATE CONSTRAINT gene_name IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint warning: {e}")
            
            print("✓ Constraints created")
    
    def create_articles(self, articles_batch):
        """Create Article nodes in batch using MERGE"""
        with self.driver.session() as session:
            query = """
            UNWIND $articles AS article
            MERGE (a:Article {pmid: article.pmid})
            ON CREATE SET 
                a.title = article.title,
                a.abstract = article.abstract,
                a.category = article.category,
                a.year = article.year,
                a.disease_count = article.disease_count,
                a.drug_count = article.drug_count,
                a.gene_count = article.gene_count
            ON MATCH SET
                a.title = article.title,
                a.abstract = article.abstract,
                a.category = article.category,
                a.year = article.year,
                a.disease_count = article.disease_count,
                a.drug_count = article.drug_count,
                a.gene_count = article.gene_count
            """
            session.run(query, articles=articles_batch)
    
    def create_entities(self, entity_type, entities_batch):
        """Create entity nodes in batch"""
        with self.driver.session() as session:
            query = f"""
            UNWIND $entities AS entity
            MERGE (e:{entity_type} {{name: entity.name}})
            ON CREATE SET e.mention_count = entity.count
            ON MATCH SET e.mention_count = e.mention_count + entity.count
            """
            session.run(query, entities=entities_batch)
    
    def create_relationships(self, rel_type, relationships_batch):
        """Create relationships in batch"""
        with self.driver.session() as session:
            if rel_type == "MENTIONS_DISEASE":
                query = """
                UNWIND $rels AS rel
                MATCH (a:Article {pmid: rel.pmid})
                MATCH (d:Disease {name: rel.entity})
                MERGE (a)-[:MENTIONS_DISEASE]->(d)
                """
            elif rel_type == "MENTIONS_DRUG":
                query = """
                UNWIND $rels AS rel
                MATCH (a:Article {pmid: rel.pmid})
                MATCH (dr:Drug {name: rel.entity})
                MERGE (a)-[:MENTIONS_DRUG]->(dr)
                """
            elif rel_type == "MENTIONS_GENE":
                query = """
                UNWIND $rels AS rel
                MATCH (a:Article {pmid: rel.pmid})
                MATCH (g:Gene {name: rel.entity})
                MERGE (a)-[:MENTIONS_GENE]->(g)
                """
            
            session.run(query, rels=relationships_batch)
    
    def create_cooccurrence_relationships(self):
        """Create relationships between entities that co-occur in articles"""
        with self.driver.session() as session:
            # Drug-Disease relationships
            print("Creating TREATS relationships (Drug->Disease)...")
            query_treats = """
            MATCH (a:Article)-[:MENTIONS_DRUG]->(dr:Drug)
            MATCH (a)-[:MENTIONS_DISEASE]->(d:Disease)
            WITH dr, d, count(a) as cooccurrence
            WHERE cooccurrence >= 5
            MERGE (dr)-[r:TREATS]->(d)
            SET r.evidence_count = cooccurrence
            """
            session.run(query_treats)
            
            # Gene-Disease relationships
            print("Creating ASSOCIATED_WITH relationships (Gene->Disease)...")
            query_assoc = """
            MATCH (a:Article)-[:MENTIONS_GENE]->(g:Gene)
            MATCH (a)-[:MENTIONS_DISEASE]->(d:Disease)
            WITH g, d, count(a) as cooccurrence
            WHERE cooccurrence >= 3
            MERGE (g)-[r:ASSOCIATED_WITH]->(d)
            SET r.evidence_count = cooccurrence
            """
            session.run(query_assoc)
            
            # Drug-Gene relationships
            print("Creating TARGETS relationships (Drug->Gene)...")
            query_targets = """
            MATCH (a:Article)-[:MENTIONS_DRUG]->(dr:Drug)
            MATCH (a)-[:MENTIONS_GENE]->(g:Gene)
            WITH dr, g, count(a) as cooccurrence
            WHERE cooccurrence >= 3
            MERGE (dr)-[r:TARGETS]->(g)
            SET r.evidence_count = cooccurrence
            """
            session.run(query_targets)
    
    def get_statistics(self):
        """Get graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Node counts
            result = session.run("MATCH (a:Article) RETURN count(a) as count")
            stats['articles'] = result.single()['count']
            
            result = session.run("MATCH (d:Disease) RETURN count(d) as count")
            stats['diseases'] = result.single()['count']
            
            result = session.run("MATCH (dr:Drug) RETURN count(dr) as count")
            stats['drugs'] = result.single()['count']
            
            result = session.run("MATCH (g:Gene) RETURN count(g) as count")
            stats['genes'] = result.single()['count']
            
            # Relationship counts
            result = session.run("MATCH ()-[r:MENTIONS_DISEASE]->() RETURN count(r) as count")
            stats['mentions_disease'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:MENTIONS_DRUG]->() RETURN count(r) as count")
            stats['mentions_drug'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:MENTIONS_GENE]->() RETURN count(r) as count")
            stats['mentions_gene'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:TREATS]->() RETURN count(r) as count")
            stats['treats'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:ASSOCIATED_WITH]->() RETURN count(r) as count")
            stats['associated_with'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:TARGETS]->() RETURN count(r) as count")
            stats['targets'] = result.single()['count']
            
            return stats


def build_graph(data_file, uri, user, password):
    """Build knowledge graph from extracted entities"""
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r') as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles\n")
    
    # Initialize graph
    kg = MedicalKnowledgeGraph(uri, user, password)
    
    # Clear existing data
    print("Clearing existing data...")
    kg.clear_database()
    
    # Create constraints
    print("Creating constraints...")
    kg.create_constraints()
    
    # Step 1: Create Article nodes
    print(f"\nStep 1: Creating Article nodes...")
    batch_size = 1000
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        kg.create_articles(batch)
        print(f"  Created {min(i+batch_size, len(articles))}/{len(articles)} articles")
    
    # Step 2: Aggregate and create entity nodes
    print(f"\nStep 2: Creating entity nodes...")
    
    disease_counts = defaultdict(int)
    drug_counts = defaultdict(int)
    gene_counts = defaultdict(int)
    
    for article in articles:
        for disease in article['diseases']:
            disease_counts[disease] += 1
        for drug in article['drugs']:
            drug_counts[drug] += 1
        for gene in article['genes']:
            gene_counts[gene] += 1
    
    # Create Disease nodes
    disease_batch = [{'name': k, 'count': v} for k, v in disease_counts.items()]
    kg.create_entities('Disease', disease_batch)
    print(f"  Created {len(disease_batch)} Disease nodes")
    
    # Create Drug nodes
    drug_batch = [{'name': k, 'count': v} for k, v in drug_counts.items()]
    kg.create_entities('Drug', drug_batch)
    print(f"  Created {len(drug_batch)} Drug nodes")
    
    # Create Gene nodes
    gene_batch = [{'name': k, 'count': v} for k, v in gene_counts.items()]
    kg.create_entities('Gene', gene_batch)
    print(f"  Created {len(gene_batch)} Gene nodes")
    
    # Step 3: Create Article->Entity relationships
    print(f"\nStep 3: Creating Article->Entity relationships...")
    
    # Disease relationships
    disease_rels = []
    for article in articles:
        for disease in article['diseases']:
            disease_rels.append({'pmid': article['pmid'], 'entity': disease})
    
    for i in range(0, len(disease_rels), batch_size):
        batch = disease_rels[i:i+batch_size]
        kg.create_relationships('MENTIONS_DISEASE', batch)
    print(f"  Created {len(disease_rels)} MENTIONS_DISEASE relationships")
    
    # Drug relationships
    drug_rels = []
    for article in articles:
        for drug in article['drugs']:
            drug_rels.append({'pmid': article['pmid'], 'entity': drug})
    
    for i in range(0, len(drug_rels), batch_size):
        batch = drug_rels[i:i+batch_size]
        kg.create_relationships('MENTIONS_DRUG', batch)
    print(f"  Created {len(drug_rels)} MENTIONS_DRUG relationships")
    
    # Gene relationships
    gene_rels = []
    for article in articles:
        for gene in article['genes']:
            gene_rels.append({'pmid': article['pmid'], 'entity': gene})
    
    for i in range(0, len(gene_rels), batch_size):
        batch = gene_rels[i:i+batch_size]
        kg.create_relationships('MENTIONS_GENE', batch)
    print(f"  Created {len(gene_rels)} MENTIONS_GENE relationships")
    
    # Step 4: Create co-occurrence relationships
    print(f"\nStep 4: Creating co-occurrence relationships...")
    kg.create_cooccurrence_relationships()
    
    # Get statistics
    print(f"\n{'='*60}")
    print("KNOWLEDGE GRAPH STATISTICS")
    print(f"{'='*60}")
    stats = kg.get_statistics()
    
    print(f"\nNodes:")
    print(f"  Articles: {stats['articles']:,}")
    print(f"  Diseases: {stats['diseases']:,}")
    print(f"  Drugs:    {stats['drugs']:,}")
    print(f"  Genes:    {stats['genes']:,}")
    print(f"  TOTAL:    {sum([stats['articles'], stats['diseases'], stats['drugs'], stats['genes']]):,}")
    
    print(f"\nRelationships:")
    print(f"  MENTIONS_DISEASE:  {stats['mentions_disease']:,}")
    print(f"  MENTIONS_DRUG:     {stats['mentions_drug']:,}")
    print(f"  MENTIONS_GENE:     {stats['mentions_gene']:,}")
    print(f"  TREATS:            {stats['treats']:,}")
    print(f"  ASSOCIATED_WITH:   {stats['associated_with']:,}")
    print(f"  TARGETS:           {stats['targets']:,}")
    print(f"  TOTAL:             {sum([stats['mentions_disease'], stats['mentions_drug'], stats['mentions_gene'], stats['treats'], stats['associated_with'], stats['targets']]):,}")
    
    kg.close()
    print(f"\n✓ Knowledge graph construction complete!")


if __name__ == "__main__":
    # Neo4j connection settings
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "MedicalLiterature"
    
    data_file = 'data/pubmed_entities_extracted.json'
    
    build_graph(data_file, URI, USER, PASSWORD)