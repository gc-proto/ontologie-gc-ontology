# Script to extract info from TTL file and augment a query

import requests
from rdflib import Graph, Namespace, Literal
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
    # Convert results to strings and clean them
    cleaned_results = []
    for result in results:
        # Each result is a tuple, with the literal value as the first element
        value = result[0]
        # Check if the value is a Literal and extract its string representation
        if isinstance(value, Literal):
            str_value = str(value)
        else:
            str_value = str(value)
        cleaned_results.append(str_value)
    
    # Join the cleaned string values, keeping newline characters as they are
    info = "\n".join(cleaned_results)  # This keeps newline characters

    return info


def get_augmented_query(user_query):
    predicted_ttl = predict_model(nlp_ttl, user_query)
    predicted_node = predict_model(nlp_node, user_query)
    ttl_url = predicted_ttl
    node = predicted_node
    service_name = service_map.get(ttl_url)
    info = extract_info_from_ttl(ttl_url, service_name, node)
    augmented_content = f". Use this information retrieved from a knowledge graph to improve your answer: {node} : {info}"
    combined_query = user_query + " " + augmented_content
    return combined_query, ttl_url, node, info

# User query
user_query = input("Enter your query: ")

combined_query, ttl_url, node, info = get_augmented_query(user_query)

# Print result
print('TTL file: ' + ttl_url)
print('Property used: ' + node)
print('Augmented query: ' + combined_query)

