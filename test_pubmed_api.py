# test_pubmed_api.py
from Bio import Entrez, Medline
import time

# Set email for NCBI compliance
Entrez.email = "gihbeom@gmail.com"  # Replace with your email

def test_pubmed_search(query, max_results=10):
    """Test PubMed search functionality"""
    print(f"Searching PubMed for: {query}")
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    id_list = record["IdList"]
    print(f"Found {len(id_list)} articles")
    
    if not id_list:
        return []
    
    # Fetch article details
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    
    articles = []
    for record in records:
        article = {
            'pmid': record.get('PMID', 'N/A'),
            'title': record.get('TI', 'N/A'),
            'abstract': record.get('AB', 'N/A'),
            'authors': record.get('AU', []),
            'journal': record.get('JT', 'N/A'),
            'year': record.get('DP', 'N/A')
        }
        articles.append(article)
    
    handle.close()
    return articles

# Test execution
if __name__ == "__main__":
    # Test query: diabetes and metformin
    test_query = "diabetes AND metformin"
    results = test_pubmed_search(test_query, max_results=5)
    
    print("\n=== RESULTS ===")
    for i, article in enumerate(results, 1):
        print(f"\n{i}. PMID: {article['pmid']}")
        print(f"   Title: {article['title'][:100]}...")
        print(f"   Journal: {article['journal']}")
        print(f"   Year: {article['year']}")
        print(f"   Abstract length: {len(article['abstract'])} chars")
    
    print(f"\n✓ API test successful: {len(results)} articles retrieved")