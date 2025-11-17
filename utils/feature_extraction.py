import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import USE_SENTENCE_TRANSFORMERS, EMBEDDING_MODEL, SEED

# Lazy imports for sentence-transformers
_sentence_model = None
_tfidf_vectorizer = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _load_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
    return _sentence_model

def fit_tfidf(corpus):
    global _tfidf_vectorizer
    _tfidf_vectorizer = TfidfVectorizer(max_features=512)
    _tfidf_vectorizer.fit(corpus)
    return _tfidf_vectorizer

def transform_tfidf(text):
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        raise RuntimeError("TF-IDF vectorizer not fitted. Call fit_tfidf(corpus) first.")
    v = _tfidf_vectorizer.transform([text]).toarray()[0]
    return v.astype(np.float32)

def extract_features(text, fit_tfidf_if_needed=False):
    """
    Return a 1D numpy float32 vector representing the input text.
    Uses sentence-transformers when available and configured,
    otherwise uses TF-IDF fallback.
    """
    text = clean_text(text)
    if USE_SENTENCE_TRANSFORMERS:
        try:
            model = _load_sentence_model()
            emb = model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float32)
        except Exception:
            # fall back to TF-IDF (will fit if asked)
            if fit_tfidf_if_needed:
                raise RuntimeError("Sentence embedding failed and TF-IDF fallback is not fit.")
            raise

    # TF-IDF path
    if _tfidf_vectorizer is None:
        if fit_tfidf_if_needed:
            raise RuntimeError("TF-IDF vectorizer not fitted. Call fit_tfidf(corpus) before training.")
        raise RuntimeError("TF-IDF vectorizer not set. Call fit_tfidf(corpus).")
    return transform_tfidf(text)
