# evaluate_qa_system.py
from clinical_qa_system import ClinicalQASystem
import json

def evaluate_retrieval_accuracy():
    """Evaluate retrieval accuracy on test questions"""
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "MedicalLiterature"
    
    qa = ClinicalQASystem(URI, USER, PASSWORD)
    
    # Test questions with expected entity detection
    test_cases = [
        {
            'question': "How does insulin treat diabetes?",
            'expected_disease': 'diabetes',
            'expected_drug': 'insulin',
            'min_articles': 1
        },
        {
            'question': "What is metformin used for in diabetes?",
            'expected_disease': 'diabetes',
            'expected_drug': 'metformin',
            'min_articles': 1
        },
        {
            'question': "Cancer treatment with chemotherapy",
            'expected_disease': 'cancer',
            'expected_drug': 'chemotherapy',
            'min_articles': 1
        },
        {
            'question': "Stroke prevention strategies",
            'expected_disease': 'stroke',
            'expected_drug': None,
            'min_articles': 1
        }
    ]
    
    print("="*80)
    print("Q&A SYSTEM EVALUATION")
    print("="*80)
    
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total_tests}: {test['question']}")
        
        # Extract entities
        entities = qa.extract_entities_from_question(test['question'])
        
        # Check disease detection
        disease_correct = test['expected_disease'] in entities['diseases'] if test['expected_disease'] else True
        
        # Check drug detection
        drug_correct = test['expected_drug'] in entities['drugs'] if test['expected_drug'] else True
        
        # Retrieve articles
        articles = qa.retrieve_relevant_articles(test['question'], top_k=5)
        articles_sufficient = len(articles) >= test['min_articles']
        
        # Evaluate
        test_passed = disease_correct and drug_correct and articles_sufficient
        
        if test_passed:
            passed_tests += 1
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED")
        
        print(f"    Disease detected: {entities['diseases']} (expected: {test['expected_disease']})")
        print(f"    Drug detected: {entities['drugs']} (expected: {test['expected_drug']})")
        print(f"    Articles retrieved: {len(articles)} (min required: {test['min_articles']})")
    
    accuracy = (passed_tests / total_tests) * 100
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"{'='*80}")
    
    qa.close()


if __name__ == "__main__":
    evaluate_retrieval_accuracy()