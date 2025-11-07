import re
import numpy as np

# Basic keyword-based vectorizer
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_features(email_text):
    text = clean_text(email_text)
    keywords = ["login", "password", "refund", "billing", "feature", "bug"]
    vec = np.zeros(len(keywords))
    for i, key in enumerate(keywords):
        if key in text:
            vec[i] = 1
    return vec
