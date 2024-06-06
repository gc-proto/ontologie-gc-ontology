import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import spacy
import ollama

# Load SentenceTransformer model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Load spaCy model for NER if needed (kept from the original code, though not used here)
nlp = spacy.load("en_core_web_sm")

def fetch_and_chunk_by_sections(url):
    """
    Fetches and chunks content from the given URL by <h2> sections.
    
    Parameters:
    url (str): The URL to fetch content from.
    
    Returns:
    list: A list of content sections.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching the web page: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract meaningful content based on <h2> sections
    sections = []
    current_section = ''
    for tag in soup.find_all(['h2', 'p', 'ul', 'li']):
        if tag.name == 'h2' and current_section:
            sections.append(remove_duplicates(current_section.strip()))
            current_section = tag.get_text(separator=' ', strip=True) + '\n'
        else:
            current_section += tag.get_text(separator=' ', strip=True) + ' '
    
    if current_section:
        sections.append(remove_duplicates(current_section.strip()))
    
    return sections

def remove_duplicates(text):
    """
    Removes duplicate lines from the text.
    
    Parameters:
    text (str): The text to process.
    
    Returns:
    str: The text with duplicate lines removed.
    """
    lines = text.split('. ')
    seen = set()
    filtered_lines = []
    for line in lines:
        if line not in seen:
            filtered_lines.append(line)
            seen.add(line)
    return '. '.join(filtered_lines)

def find_relevant_chunk(sections, user_query):
    """
    Finds the most relevant section of content based on the user's query.
    
    Parameters:
    sections (list): A list of content sections.
    user_query (str): The user's query.
    
    Returns:
    str: The most relevant section of content.
    """
    # Create embeddings for all sections and the user query
    section_embeddings = model.encode(sections, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, section_embeddings)
    
    # Find the index of the most relevant section
    most_relevant_section_idx = cosine_scores.argmax().item()
    
    return sections[most_relevant_section_idx]

def read_modelfile(filename):
    """
    Reads the Modelfile.txt and returns the parameters and system message.
    
    Parameters:
    filename (str): The path to the Modelfile.txt
    
    Returns:
    dict: A dictionary with parameters and system message
    """
    params = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('PARAMETER'):
                key, value = line.split()[1:]
                params[key] = float(value) if '.' in value else int(value)
            elif line.startswith('SYSTEM'):
                params['system_message'] = ' '.join(line.split()[1:])
    return params


# User query
user_query = input("Enter your query: ")

# URL to fetch content from
url = 'https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada/prepare-arrival.html'

# Fetch and chunk content from the URL by sections
sections = fetch_and_chunk_by_sections(url)
print(f"Fetched {len(sections)} sections of content.")

# Find the most relevant section
relevant_section = find_relevant_chunk(sections, user_query)
print("Most relevant section:")
print(relevant_section)

# Prepare combined query
augmented_query = f"{user_query} (Use this additional information to improve your answer if relevant: {relevant_section})"
print("Augmented query:")
print(augmented_query)

# Read parameters and system message from Modelfile.txt
modelfile_path = './Modelfile.txt'
params = read_modelfile(modelfile_path)

# Prepare the messages with the system message
messages = [
    {'role': 'system', 'content': params.get('system_message', '')},
    {'role': 'user', 'content': augmented_query}
]

# Query with the specified parameters
stream = ollama.chat(
    model='ttl_llama3',
    messages=messages,
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

print(' For more information, go to: ' + url)
