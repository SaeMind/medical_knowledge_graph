# clinical_qa_system.py
from neo4j import GraphDatabase
import json
import numpy as np
from typing import List, Dict
import re

class ClinicalQASystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Load article data for retrieval
        with open('data/pubmed_entities_extracted.json', 'r') as f:
            self.articles = json.load(f)
        
        # Create article index by PMID
        self.article_index = {a['pmid']: a for a in self.articles}
    
    def close(self):
        self.driver.close()
    
    def extract_entities_from_question(self, question: str) -> Dict:
        """Extract medical entities from question"""
        question_lower = question.lower()
        
        # Common disease keywords
        diseases = ['diabetes', 'cancer', 'hypertension', 'stroke', 'alzheimer', 
                   'infection', 'tumor', 'obesity', 'heart failure', 'parkinson']
        
        # Common drug keywords
        drugs = ['insulin', 'metformin', 'aspirin', 'chemotherapy', 'immunotherapy',
                'statin', 'antibiotic', 'antiviral']
        
        found_diseases = [d for d in diseases if d in question_lower]
        found_drugs = [d for d in drugs if d in question_lower]
        
        return {
            'diseases': found_diseases,
            'drugs': found_drugs
        }
    
    def retrieve_relevant_articles(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant articles from knowledge graph"""
        entities = self.extract_entities_from_question(question)
        
        with self.driver.session() as session:
            # Build query based on detected entities
            if entities['diseases'] and entities['drugs']:
                # Question involves both disease and drug
                disease = entities['diseases'][0]
                drug = entities['drugs'][0]
                
                query = """
                MATCH (a:Article)-[:MENTIONS_DISEASE]->(d:Disease {name: $disease})
                MATCH (a)-[:MENTIONS_DRUG]->(dr:Drug {name: $drug})
                RETURN a.pmid as pmid, a.title as title, a.abstract as abstract
                LIMIT $top_k
                """
                result = session.run(query, disease=disease, drug=drug, top_k=top_k)
                
            elif entities['diseases']:
                # Question about disease only
                disease = entities['diseases'][0]
                
                query = """
                MATCH (a:Article)-[:MENTIONS_DISEASE]->(d:Disease {name: $disease})
                RETURN a.pmid as pmid, a.title as title, a.abstract as abstract
                ORDER BY a.year DESC
                LIMIT $top_k
                """
                result = session.run(query, disease=disease, top_k=top_k)
                
            elif entities['drugs']:
                # Question about drug only
                drug = entities['drugs'][0]
                
                query = """
                MATCH (a:Article)-[:MENTIONS_DRUG]->(dr:Drug {name: $drug})
                RETURN a.pmid as pmid, a.title as title, a.abstract as abstract
                ORDER BY a.year DESC
                LIMIT $top_k
                """
                result = session.run(query, drug=drug, top_k=top_k)
            else:
                # No specific entities detected - return recent articles
                return []
            
            articles = [dict(record) for record in result]
            return articles
    
    def generate_answer(self, question: str, articles: List[Dict]) -> str:
        """Generate answer based on retrieved articles"""
        if not articles:
            return "I don't have enough information to answer this question based on the available literature."
        
        # Simple answer generation (would use LLM in production)
        entities = self.extract_entities_from_question(question)
        
        answer_parts = []
        answer_parts.append(f"Based on {len(articles)} relevant scientific articles:\n")
        
        # Summarize findings
        if entities['diseases'] and entities['drugs']:
            disease = entities['diseases'][0]
            drug = entities['drugs'][0]
            answer_parts.append(f"\nRegarding {drug} for {disease}:")
            answer_parts.append(f"- Found {len(articles)} articles discussing this combination")
            
        elif entities['diseases']:
            disease = entities['diseases'][0]
            answer_parts.append(f"\nRegarding {disease}:")
            
        elif entities['drugs']:
            drug = entities['drugs'][0]
            answer_parts.append(f"\nRegarding {drug}:")
        
        # Add article references
        answer_parts.append("\n\nKey findings from recent literature:")
        for i, article in enumerate(articles[:3], 1):
            title_short = article['title'][:100] + "..." if len(article['title']) > 100 else article['title']
            answer_parts.append(f"{i}. {title_short} (PMID: {article['pmid']})")
        
        return "\n".join(answer_parts)
    
    def answer_question(self, question: str) -> Dict:
        """Main Q&A pipeline"""
        # Step 1: Retrieve relevant articles
        articles = self.retrieve_relevant_articles(question, top_k=5)
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, articles)
        
        # Step 3: Get additional context from graph
        entities = self.extract_entities_from_question(question)
        graph_context = self.get_graph_context(entities)
        
        return {
            'question': question,
            'answer': answer,
            'supporting_articles': articles,
            'graph_context': graph_context
        }
    
    def get_graph_context(self, entities: Dict) -> Dict:
        """Get additional context from knowledge graph"""
        context = {}
        
        with self.driver.session() as session:
            if entities['diseases']:
                disease = entities['diseases'][0]
                
                # Get treatment count
                query = """
                MATCH (dr:Drug)-[r:TREATS]->(d:Disease {name: $disease})
                RETURN count(dr) as treatment_count
                """
                result = session.run(query, disease=disease)
                record = result.single()
                context['treatments_available'] = record['treatment_count'] if record else 0
                
                # Get associated genes count
                query = """
                MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(d:Disease {name: $disease})
                RETURN count(g) as gene_count
                """
                result = session.run(query, disease=disease)
                record = result.single()
                context['associated_genes'] = record['gene_count'] if record else 0
            
            if entities['drugs']:
                drug = entities['drugs'][0]
                
                # Get diseases treated
                query = """
                MATCH (dr:Drug {name: $drug})-[r:TREATS]->(d:Disease)
                RETURN count(d) as disease_count
                """
                result = session.run(query, drug=drug)
                record = result.single()
                context['diseases_treated'] = record['disease_count'] if record else 0
        
        return context


def run_qa_examples():
    """Run example clinical questions"""
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "MedicalLiterature"
    
    qa = ClinicalQASystem(URI, USER, PASSWORD)
    
    print("="*80)
    print("CLINICAL QUESTION ANSWERING SYSTEM - EXAMPLES")
    print("="*80)
    
    # Example questions
    questions = [
        "What is the role of insulin in diabetes treatment?",
        "How does metformin work for diabetes?",
        "What are the treatment options for hypertension?",
        "What genes are associated with cancer?",
        "Is chemotherapy effective for cancer treatment?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print(f"{'='*80}")
        
        result = qa.answer_question(question)
        
        print(f"\n{result['answer']}")
        
        if result['graph_context']:
            print(f"\nKnowledge Graph Context:")
            for key, value in result['graph_context'].items():
                print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nRetrieved {len(result['supporting_articles'])} supporting articles")
    
    qa.close()
    print(f"\n{'='*80}")
    print("✓ Q&A examples complete!")
    print("="*80)


if __name__ == "__main__":
    run_qa_examples()