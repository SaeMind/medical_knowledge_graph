# collect_pubmed_data.py
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import json
from datetime import datetime
import os

class PubMedCollector:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.rate_limit_delay = 0.34  # 3 requests/second max
        
    def search_ids(self, query, max_results=10000):
        """Search PubMed and return list of PMIDs"""
        search_url = f"{self.base_url}esearch.fcgi"
        
        all_ids = []
        retstart = 0
        retmax = 10000  # API max per request
        
        while retstart < max_results:
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': min(retmax, max_results - retstart),
                'retstart': retstart,
                'retmode': 'json'
            }
            
            print(f"Fetching IDs {retstart} to {retstart + retmax}...")
            response = requests.get(search_url, params=params)
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
            
            data = response.json()
            id_list = data['esearchresult']['idlist']
            
            if not id_list:
                break
            
            all_ids.extend(id_list)
            retstart += len(id_list)
            
            print(f"Total IDs collected: {len(all_ids)}")
            
            if len(id_list) < retmax:
                break
            
            time.sleep(self.rate_limit_delay)
        
        return all_ids
    
    def fetch_articles_batch(self, pmid_list):
        """Fetch article details for a batch of PMIDs"""
        fetch_url = f"{self.base_url}efetch.fcgi"
        
        # Join PMIDs
        id_string = ','.join(pmid_list)
        
        params = {
            'db': 'pubmed',
            'id': id_string,
            'retmode': 'xml'
        }
        
        response = requests.get(fetch_url, params=params)
        
        if response.status_code != 200:
            return []
        
        # Parse XML
        root = ET.fromstring(response.content)
        articles = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                pmid_elem = article.find('.//PMID')
                title_elem = article.find('.//ArticleTitle')
                abstract_elem = article.find('.//AbstractText')
                journal_elem = article.find('.//Journal/Title')
                year_elem = article.find('.//PubDate/Year')
                
                # Extract MeSH terms
                mesh_terms = []
                for mesh in article.findall('.//MeshHeading/DescriptorName'):
                    mesh_terms.append(mesh.text)
                
                # Extract keywords
                keywords = []
                for kw in article.findall('.//Keyword'):
                    keywords.append(kw.text)
                
                article_data = {
                    'pmid': pmid_elem.text if pmid_elem is not None else None,
                    'title': title_elem.text if title_elem is not None else None,
                    'abstract': abstract_elem.text if abstract_elem is not None else None,
                    'journal': journal_elem.text if journal_elem is not None else None,
                    'year': year_elem.text if year_elem is not None else None,
                    'mesh_terms': mesh_terms,
                    'keywords': keywords
                }
                
                # Only include if has abstract
                if article_data['abstract']:
                    articles.append(article_data)
                    
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def collect_articles(self, query, max_articles=10000, batch_size=200):
        """Collect articles for a specific query"""
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Target: {max_articles} articles")
        print(f"{'='*60}\n")
        
        # Step 1: Get PMIDs
        print("Step 1: Searching for PMIDs...")
        pmids = self.search_ids(query, max_results=max_articles)
        print(f"✓ Found {len(pmids)} PMIDs\n")
        
        # Step 2: Fetch articles in batches
        print("Step 2: Fetching article details...")
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            print(f"Batch {i//batch_size + 1}: Fetching PMIDs {i} to {i+len(batch)}...")
            
            articles = self.fetch_articles_batch(batch)
            all_articles.extend(articles)
            
            print(f"  → Retrieved {len(articles)} articles with abstracts")
            print(f"  → Total collected: {len(all_articles)}")
            
            time.sleep(self.rate_limit_delay)
        
        return all_articles


def main():
    collector = PubMedCollector()
    
    # Define queries for different medical domains
    queries = {
        'diabetes': '(diabetes mellitus OR type 2 diabetes OR type 1 diabetes) AND hasabstract[text]',
        'cardiovascular': '(cardiovascular disease OR coronary artery disease OR heart failure OR hypertension) AND hasabstract[text]',
        'cancer': '(cancer OR carcinoma OR tumor OR neoplasm OR oncology) AND hasabstract[text]',
        'infectious_disease': '(infection OR infectious disease OR pathogen OR antimicrobial) AND hasabstract[text]',
        'neurology': '(neurology OR neurological disease OR alzheimer OR parkinson OR stroke) AND hasabstract[text]'
    }
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    all_data = []
    category_counts = {}
    
    # Collect articles for each category
    for category, query in queries.items():
        print(f"\n{'#'*60}")
        print(f"# Collecting: {category.upper()}")
        print(f"{'#'*60}")
        
        articles = collector.collect_articles(query, max_articles=25000, batch_size=200)
        
        # Add category label
        for article in articles:
            article['category'] = category
        
        all_data.extend(articles)
        category_counts[category] = len(articles)
        
        # Save intermediate results
        df_temp = pd.DataFrame(articles)
        df_temp.to_csv(f'data/{category}_articles.csv', index=False)
        print(f"\n✓ Saved {len(articles)} articles to data/{category}_articles.csv")
    
    # Save combined dataset
    df_all = pd.DataFrame(all_data)
    df_all.to_csv('data/pubmed_articles_all.csv', index=False)
    
    # Save as JSON for easier processing
    with open('data/pubmed_articles_all.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles collected: {len(all_data)}")
    print(f"\nBy category:")
    for category, count in category_counts.items():
        print(f"  {category:20s}: {count:6d} articles")
    
    print(f"\nFiles saved:")
    print(f"  - data/pubmed_articles_all.csv")
    print(f"  - data/pubmed_articles_all.json")
    for category in queries.keys():
        print(f"  - data/{category}_articles.csv")
    
    print(f"\n✓ Data collection complete!")

if __name__ == "__main__":
    main()