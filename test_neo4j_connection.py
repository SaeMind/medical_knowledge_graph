# test_neo4j_connection.py
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "MedicalLiterature"

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 'Connection successful!' as message")
        print(result.single()['message'])
    driver.close()
    print("✓ Neo4j connection verified")
except Exception as e:
    print(f"✗ Connection failed: {e}")