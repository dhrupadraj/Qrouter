from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load a lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

def extract_features(email_text):
    text = clean_text(email_text)
    embedding = model.encode(text)

    # Convert to numpy array and return
    return np.array(embedding)
