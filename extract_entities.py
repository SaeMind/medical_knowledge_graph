# extract_entities.py
import pandas as pd
import json
import re
from collections import defaultdict
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class MedicalEntityExtractor:
    def __init__(self):
        # Load medical term dictionaries
        self.disease_patterns = self._load_disease_patterns()
        self.drug_patterns = self._load_drug_patterns()
        self.gene_patterns = self._load_gene_patterns()
        
    def _load_disease_patterns(self):
        """Common disease name patterns"""
        diseases = [
            'diabetes', 'hypertension', 'cancer', 'carcinoma', 'tumor', 'neoplasm',
            'infection', 'stroke', 'alzheimer', 'parkinson', 'heart failure',
            'coronary artery disease', 'myocardial infarction', 'pneumonia',
            'sepsis', 'cirrhosis', 'nephropathy', 'neuropathy', 'retinopathy',
            'obesity', 'atherosclerosis', 'fibrillation', 'ischemia', 'leukemia',
            'lymphoma', 'melanoma', 'sarcoma', 'glioma', 'hepatitis', 'tuberculosis'
        ]
        return diseases
    
    def _load_drug_patterns(self):
        """Common drug name patterns"""
        drugs = [
            'metformin', 'insulin', 'aspirin', 'warfarin', 'heparin', 'statin',
            'atorvastatin', 'simvastatin', 'lisinopril', 'amlodipine', 'losartan',
            'levothyroxine', 'omeprazole', 'albuterol', 'gabapentin', 'prednisone',
            'ibuprofen', 'acetaminophen', 'amoxicillin', 'azithromycin', 'ciprofloxacin',
            'doxycycline', 'chemotherapy', 'immunotherapy', 'antibiotic', 'antiviral',
            'beta-blocker', 'ace inhibitor', 'diuretic', 'anticoagulant', 'corticosteroid'
        ]
        return drugs
    
    def _load_gene_patterns(self):
        """Common gene name patterns"""
        genes = [
            'TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA', 'APC', 'PTEN',
            'RB1', 'VHL', 'CDKN2A', 'SMAD4', 'MLH1', 'MSH2', 'ATM', 'HER2',
            'BCR-ABL', 'JAK2', 'FLT3', 'NPM1', 'DNMT3A', 'IDH1', 'IDH2'
        ]
        return genes
    
    def extract_diseases(self, text):
        """Extract disease mentions from text"""
        text_lower = text.lower()
        found_diseases = set()
        
        for disease in self.disease_patterns:
            if disease in text_lower:
                found_diseases.add(disease)
        
        # Pattern matching for disease suffixes
        disease_suffixes = ['-itis', '-osis', '-emia', '-pathy', '-oma']
        words = text.split()
        
        for word in words:
            word_clean = re.sub(r'[^\w-]', '', word.lower())
            for suffix in disease_suffixes:
                if word_clean.endswith(suffix) and len(word_clean) > len(suffix) + 2:
                    found_diseases.add(word_clean)
        
        return list(found_diseases)
    
    def extract_drugs(self, text):
        """Extract drug mentions from text"""
        text_lower = text.lower()
        found_drugs = set()
        
        for drug in self.drug_patterns:
            if drug in text_lower:
                found_drugs.add(drug)
        
        # Pattern for drug suffixes
        drug_suffixes = ['-mab', '-nib', '-ine', '-ol', '-pril', '-sartan', '-statin', '-cillin']
        words = text.split()
        
        for word in words:
            word_clean = re.sub(r'[^\w-]', '', word.lower())
            for suffix in drug_suffixes:
                if word_clean.endswith(suffix) and len(word_clean) > len(suffix) + 2:
                    found_drugs.add(word_clean)
        
        return list(found_drugs)
    
    def extract_genes(self, text):
        """Extract gene mentions from text"""
        found_genes = set()
        
        # Known genes
        for gene in self.gene_patterns:
            if re.search(r'\b' + gene + r'\b', text):
                found_genes.add(gene)
        
        # Pattern: 2-6 uppercase letters possibly with numbers
        gene_pattern = r'\b[A-Z]{2,6}[0-9]{0,2}\b'
        potential_genes = re.findall(gene_pattern, text)
        
        # Filter out common abbreviations
        exclude = {'DNA', 'RNA', 'FDA', 'CDC', 'WHO', 'USA', 'UK', 'EU', 'BMI', 'HIV', 'AIDS'}
        
        for gene in potential_genes:
            if gene not in exclude and len(gene) >= 3:
                found_genes.add(gene)
        
        return list(found_genes)
    
    def extract_all_entities(self, text):
        """Extract all entity types from text"""
        if not text or text == 'N/A':
            return {
                'diseases': [],
                'drugs': [],
                'genes': []
            }
        
        return {
            'diseases': self.extract_diseases(text),
            'drugs': self.extract_drugs(text),
            'genes': self.extract_genes(text)
        }


def process_articles(input_file, output_file):
    """Process all articles and extract entities"""
    print(f"Loading articles from {input_file}...")
    df = pd.read_json(input_file)
    print(f"Loaded {len(df)} articles\n")
    
    extractor = MedicalEntityExtractor()
    
    print("Extracting entities...")
    all_entities = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(df)} articles...")
        
        # Combine title and abstract
        text = f"{row['title']} {row['abstract']}"
        
        # Extract entities
        entities = extractor.extract_all_entities(text)
        
        article_data = {
            'pmid': row['pmid'],
            'title': row['title'],
            'abstract': row['abstract'],
            'category': row['category'],
            'year': row['year'],
            'diseases': entities['diseases'],
            'drugs': entities['drugs'],
            'genes': entities['genes'],
            'disease_count': len(entities['diseases']),
            'drug_count': len(entities['drugs']),
            'gene_count': len(entities['genes'])
        }
        
        all_entities.append(article_data)
    
    print(f"\n✓ Processed {len(all_entities)} articles")
    
    # Save results
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_entities, f, indent=2)
    
    # Also save as CSV for analysis
    df_entities = pd.DataFrame(all_entities)
    csv_file = output_file.replace('.json', '.csv')
    df_entities.to_csv(csv_file, index=False)
    print(f"✓ Saved to {output_file}")
    print(f"✓ Saved to {csv_file}")
    
    # Generate statistics
    print_statistics(all_entities)
    
    return all_entities


def print_statistics(entities):
    """Print entity extraction statistics"""
    print(f"\n{'='*60}")
    print("ENTITY EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    total_diseases = sum(len(e['diseases']) for e in entities)
    total_drugs = sum(len(e['drugs']) for e in entities)
    total_genes = sum(len(e['genes']) for e in entities)
    
    print(f"Total unique mentions:")
    print(f"  Diseases: {total_diseases:,}")
    print(f"  Drugs:    {total_drugs:,}")
    print(f"  Genes:    {total_genes:,}")
    
    # Count unique entities
    all_diseases = set()
    all_drugs = set()
    all_genes = set()
    
    for e in entities:
        all_diseases.update(e['diseases'])
        all_drugs.update(e['drugs'])
        all_genes.update(e['genes'])
    
    print(f"\nUnique entities:")
    print(f"  Diseases: {len(all_diseases):,}")
    print(f"  Drugs:    {len(all_drugs):,}")
    print(f"  Genes:    {len(all_genes):,}")
    
    # Most common entities
    disease_counts = defaultdict(int)
    drug_counts = defaultdict(int)
    gene_counts = defaultdict(int)
    
    for e in entities:
        for d in e['diseases']:
            disease_counts[d] += 1
        for dr in e['drugs']:
            drug_counts[dr] += 1
        for g in e['genes']:
            gene_counts[g] += 1
    
    print(f"\nTop 10 diseases:")
    for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {disease:30s}: {count:5d} mentions")
    
    print(f"\nTop 10 drugs:")
    for drug, count in sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {drug:30s}: {count:5d} mentions")
    
    print(f"\nTop 10 genes:")
    for gene, count in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {gene:30s}: {count:5d} mentions")


if __name__ == "__main__":
    input_file = 'data/pubmed_articles_all.json'
    output_file = 'data/pubmed_entities_extracted.json'
    
    entities = process_articles(input_file, output_file)
    
    print(f"\n✓ Entity extraction complete!")