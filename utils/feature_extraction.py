import re
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def extract_features(email_text):
    text = clean_text(email_text)

    # STATE 0 – Login issues
    if any(k in text for k in ["login", "password", "account", "credentials"]):
        return 0

    # STATE 1 – Payment or billing issues
    if any(k in text for k in ["billing", "payment", "refund", "invoice"]):
        return 1

    # STATE 2 – Bugs or errors
    if any(k in text for k in ["bug", "crash", "error", "issue", "broken"]):
        return 2

    # STATE 3 – Feature requests
    if any(k in text for k in ["feature", "request", "improvement", "enhancement"]):
        return 3

    # STATE 4 – Unknown / fallback
    return 4
