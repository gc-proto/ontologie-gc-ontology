# Script 1: Training and Saving Models

import random
import spacy
from spacy.util import minibatch
from spacy.training import Example
import csv
from sklearn.model_selection import train_test_split

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized_text

# Initialize lists for datasets
data_ttl = []
data_node = []

# Load and preprocess data from CSV
with open('TTL_training_data.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        query = preprocess_text(row['Query'])
        data_ttl.append((query, {'cats': {row['TTL']: True}}))
        data_node.append((query, {'cats': {row['Node']: True}}))

# Split data for training and testing
train_data_ttl, test_data_ttl = train_test_split(data_ttl, test_size=0.2)
train_data_node, test_data_node = train_test_split(data_node, test_size=0.2)

# Function to train a model
def train_model(training_data):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    for _, annotations in training_data:
        for label in annotations['cats']:
            textcat.add_label(label)
    optimizer = nlp.begin_training()
    for i in range(10):
        random.shuffle(training_data)
        batches = minibatch(training_data, size=8)
        for batch in batches:
            examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
            nlp.update(examples, sgd=optimizer)
    return nlp

# Train and save models
nlp_ttl = train_model(train_data_ttl)
nlp_node = train_model(train_data_node)

model_path_ttl = "./model/ttl_model"
model_path_node = "./model/node_model"

# Save the models
nlp_ttl.to_disk(model_path_ttl)
nlp_node.to_disk(model_path_node)
