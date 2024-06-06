# Script to call the OpenAI API

import ollama

# Predict TTL and Node
predicted_ttl = predict_model(nlp_ttl, user_query)
predicted_node = predict_model(nlp_node, user_query)

# Extract info from TTL
ttl_url = predicted_ttl
node = predicted_node
service_name = service_map.get(ttl_url)
info = extract_info_from_ttl(ttl_url, service_name, node)

# Prepare combined query
augmented_content = ". (Use this information retrieved from a knowledge graph to improve your answer: " + node + " :" + info
combined_query = user_query + " " + augmented_content


# Query OpenAI
print('TTL file: ' + ttl_url)
print('Property used: ' + node)
print('Content retrieved from TTL): ' + info)

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': combined_query,
  },
])
print(response['message']['content'])
