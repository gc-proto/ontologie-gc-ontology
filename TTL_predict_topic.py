# Script 2: Processing User Query and OpenAI Query

import requests
from rdflib import Graph, Namespace
import spacy

# Load models
ttl_model_path = "./model/ttl_model"
node_model_path = "./model/node_model"
nlp_ttl = spacy.load(ttl_model_path)
nlp_node = spacy.load(node_model_path)

# Service map
service_map = {
    'https://test.canada.ca/ontologie-gc-ontology/services/canada-child-benefit.ttl': 'ccb',
    'https://test.canada.ca/ontologie-gc-ontology/services/employment-insurance.ttl': 'ei',
    # ... additional mappings ...
}

# Function to predict model
def predict_model(nlp, user_query):
    doc = nlp(user_query)
    predicted_category = max(doc.cats, key=doc.cats.get)
    return predicted_category

# Function to extract information from TTL
def extract_info_from_ttl(ttl_url, service_name, node):
    response = requests.get(ttl_url)
    if response.status_code != 200:
        return f"Error fetching the TTL file: {response.status_code}"
    g = Graph()
    g.parse(ttl_url, format="ttl")
    GC = Namespace("https://test.canada.ca/ontologie-gc-ontology/gc-ontology.ttl#")
    SVC = Namespace("https://test.canada.ca/ontologie-gc-ontology/services/")
    node_uri = GC[node]    
    service_uri = SVC[service_name].n3(g.namespace_manager)
    query = f"""
    SELECT ?value
    WHERE {{
        {service_uri} {node_uri.n3()} ?value .
    }}
    """
    results = g.query(query)
    info = " ".join([str(value) for value in results])
    return info

# User query
user_query = input("Enter your query: ")

# Predict TTL and Node
predicted_ttl = predict_model(nlp_ttl, user_query)
predicted_node = predict_model(nlp_node, user_query)

# Extract info from TTL
ttl_url = predicted_ttl
node = predicted_node

# Query OpenAI
print('TTL file: ' + ttl_url)
print('Property used: ' + node)
