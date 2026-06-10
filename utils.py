"""
utils.py — Text cleaning and prediction utilities
"""

import re
import string
import joblib
import os

# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, digits, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punct
    text = re.sub(r"\d+", "", text)                        # remove digits
    text = re.sub(r"\s+", " ", text).strip()               # normalise spaces
    return text


# ── Prediction ───────────────────────────────────────────────────────────────

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def load_artifacts():
    """Load saved model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python model.py` first to train and save it."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_category(text: str, model=None, vectorizer=None) -> str:
    """
    Predict the category of a classified ad.

    Args:
        text: Raw ad text (title + description)
        model: Optional pre-loaded model (avoids disk I/O on repeated calls)
        vectorizer: Optional pre-loaded vectorizer

    Returns:
        Predicted category string: 'cars' | 'property' | 'electronics' | 'jobs'
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction


def predict_with_confidence(text: str, model=None, vectorizer=None) -> dict:
    """
    Returns predicted category AND confidence scores for all classes.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    classes = model.classes_

    scores = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probas)}
    return {"prediction": prediction, "confidence": scores}
