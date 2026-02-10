# clear_neo4j.py
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "MedicalLiterature"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session() as session:
    # Drop all constraints first
    print("Dropping constraints...")
    constraints = session.run("SHOW CONSTRAINTS")
    for constraint in constraints:
        constraint_name = constraint.get('name')
        if constraint_name:
            session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
    
    # Delete all nodes and relationships
    print("Deleting all nodes and relationships...")
    session.run("MATCH (n) DETACH DELETE n")
    
    # Verify empty
    result = session.run("MATCH (n) RETURN count(n) as count")
    count = result.single()['count']
    print(f"Nodes remaining: {count}")

driver.close()
print("✓ Database cleared successfully")