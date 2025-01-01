import torch
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.spatial.distance import cosine

# Initialize the Hugging Face model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Function to get embeddings from Hugging Face's DistilBERT
def get_embedding(text):
    # Tokenize the input text and get token IDs
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Forward pass through the model to get the hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings of the [CLS] token (first token)
    cls_embedding = outputs.last_hidden_state[0][0].numpy()  # Embedding of the [CLS] token
    return cls_embedding


# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  # Cosine similarity ranges from -1 to 1


# Example words
words = ['apple', 'banana']

# Get embeddings for the words
embedding_1 = get_embedding(words[0])
embedding_2 = get_embedding(words[1])

# Compute cosine similarity
similarity = cosine_similarity(embedding_1, embedding_2)

# Print results
print(f"Embedding for '{words[0]}': {embedding_1[:10]}...")  # Showing first 10 components of the embedding
print(f"Embedding for '{words[1]}': {embedding_2[:10]}...")  # Showing first 10 components of the embedding
print(f"Cosine similarity between '{words[0]}' and '{words[1]}': {similarity:.4f}")
