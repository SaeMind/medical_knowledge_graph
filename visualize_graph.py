# visualize_graph.py
from neo4j import GraphDatabase
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class GraphVisualizer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_disease_subgraph(self, disease_name, max_nodes=50):
        """Extract subgraph around a disease"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease {name: $disease})
            OPTIONAL MATCH (d)<-[t:TREATS]-(dr:Drug)
            OPTIONAL MATCH (d)<-[a:ASSOCIATED_WITH]-(g:Gene)
            WITH d, collect(DISTINCT {type: 'Drug', name: dr.name, evidence: t.evidence_count}) as drugs,
                    collect(DISTINCT {type: 'Gene', name: g.name, evidence: a.evidence_count}) as genes
            RETURN d.name as disease, drugs, genes
            """
            result = session.run(query, disease=disease_name.lower())
            return dict(result.single())
    
    def visualize_disease_network(self, disease_name, output_file='disease_network.png'):
        """Create network visualization for a disease"""
        data = self.get_disease_subgraph(disease_name)
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add disease node
        G.add_node(data['disease'], node_type='disease')
        
        # Add drug nodes and edges
        for drug in data['drugs']:
            if drug['name']:
                G.add_node(drug['name'], node_type='drug')
                G.add_edge(data['disease'], drug['name'], 
                          weight=drug['evidence'], edge_type='treats')
        
        # Add gene nodes and edges (top 20)
        genes_sorted = sorted([g for g in data['genes'] if g['name']], 
                            key=lambda x: x['evidence'], reverse=True)[:20]
        for gene in genes_sorted:
            G.add_node(gene['name'], node_type='gene')
            G.add_edge(data['disease'], gene['name'], 
                      weight=gene['evidence'], edge_type='associated')
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Separate nodes by type
        disease_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'disease']
        drug_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']
        gene_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'gene']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, 
                              node_color='red', node_size=3000, label='Disease')
        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, 
                              node_color='green', node_size=1500, label='Drugs')
        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, 
                              node_color='blue', node_size=800, label='Genes')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        plt.title(f"Knowledge Graph: {disease_name.upper()} Network", fontsize=16, fontweight='bold')
        plt.legend(scatterpoints=1, fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_file}")
        plt.close()
    
    def generate_network_statistics(self):
        """Generate graph-level statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Graph density
            query_nodes = "MATCH (n) RETURN count(n) as count"
            query_edges = "MATCH ()-[r]->() RETURN count(r) as count"
            
            nodes = session.run(query_nodes).single()['count']
            edges = session.run(query_edges).single()['count']
            
            stats['nodes'] = nodes
            stats['edges'] = edges
            stats['density'] = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
            
            # Most connected diseases
            query_connected = """
            MATCH (d:Disease)
            OPTIONAL MATCH (d)<-[r]-()
            WITH d, count(r) as connections
            RETURN d.name as disease, connections
            ORDER BY connections DESC
            LIMIT 10
            """
            result = session.run(query_connected)
            stats['most_connected_diseases'] = [dict(r) for r in result]
            
            # Most connected drugs
            query_drugs = """
            MATCH (dr:Drug)
            OPTIONAL MATCH (dr)-[r]->()
            WITH dr, count(r) as connections
            RETURN dr.name as drug, connections
            ORDER BY connections DESC
            LIMIT 10
            """
            result = session.run(query_drugs)
            stats['most_connected_drugs'] = [dict(r) for r in result]
            
            return stats


def main():
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "MedicalLiterature"
    
    viz = GraphVisualizer(URI, USER, PASSWORD)
    
    print("="*60)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("="*60)
    
    # Generate visualizations for top diseases
    diseases_to_visualize = ['diabetes', 'cancer', 'hypertension']
    
    for disease in diseases_to_visualize:
        print(f"\nGenerating visualization for: {disease}")
        viz.visualize_disease_network(disease, f'visualizations/{disease}_network.png')
    
    # Generate statistics
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    stats = viz.generate_network_statistics()
    
    print(f"\nGraph Metrics:")
    print(f"  Total nodes: {stats['nodes']:,}")
    print(f"  Total edges: {stats['edges']:,}")
    print(f"  Density: {stats['density']:.6f}")
    
    print(f"\nMost Connected Diseases:")
    for d in stats['most_connected_diseases']:
        print(f"  {d['disease']:30s}: {d['connections']:4d} connections")
    
    print(f"\nMost Connected Drugs:")
    for d in stats['most_connected_drugs']:
        print(f"  {d['drug']:30s}: {d['connections']:4d} connections")
    
    viz.close()
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)
    main()