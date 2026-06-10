"""
model.py — Train, evaluate, and save the NLP classifier
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import generate_dataset
from utils import clean_text


# ── 1. Load / Generate Dataset ───────────────────────────────────────────────

def load_data():
    df = generate_dataset()
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"📦 Dataset: {len(df)} samples | {df['label'].nunique()} categories")
    print(df["label"].value_counts().to_string())
    print()
    return df


# ── 2. TF-IDF Vectorisation ──────────────────────────────────────────────────

def build_vectorizer(X_train):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log scaling
        min_df=1,
    )
    vectorizer.fit(X_train)
    return vectorizer


# ── 3. Train & Evaluate ──────────────────────────────────────────────────────

def train_and_evaluate(df):
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)

    print("=" * 55)
    print("  LOGISTIC REGRESSION")
    print("=" * 55)
    print(f"  Accuracy : {lr_acc:.4f}  ({lr_acc*100:.1f}%)")
    print()
    print(classification_report(y_test, lr_preds))

    print("Confusion Matrix (LR):")
    cm = confusion_matrix(y_test, lr_preds, labels=lr_model.classes_)
    _print_confusion_matrix(cm, lr_model.classes_)
    print()

    # ── Naive Bayes (comparison) ─────────────────────────────────────────────
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_preds)

    print("=" * 55)
    print("  NAIVE BAYES (comparison)")
    print("=" * 55)
    print(f"  Accuracy : {nb_acc:.4f}  ({nb_acc*100:.1f}%)")
    print()
    print(classification_report(y_test, nb_preds))

    # ── Winner Summary ───────────────────────────────────────────────────────
    winner = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    print(f"✅ Best model: {winner}  (LR={lr_acc*100:.1f}% | NB={nb_acc*100:.1f}%)")
    print()

    # ── Top TF-IDF Features per Class ───────────────────────────────────────
    print_top_features(lr_model, vectorizer, n=8)

    # ── Sample Predictions Table ─────────────────────────────────────────────
    sample_predictions(lr_model, vectorizer)

    return lr_model, vectorizer


# ── 4. Helper: Pretty Confusion Matrix ──────────────────────────────────────

def _print_confusion_matrix(cm, labels):
    col_w = 13
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(labels)))
        print(row)


# ── 5. Top TF-IDF Features ───────────────────────────────────────────────────

def print_top_features(model, vectorizer, n=8):
    print("=" * 55)
    print("  TOP TF-IDF FEATURES PER CATEGORY")
    print("=" * 55)
    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, class_label in enumerate(model.classes_):
        top_idx = np.argsort(model.coef_[i])[-n:][::-1]
        top_words = feature_names[top_idx]
        print(f"  {class_label.upper():15s}: {', '.join(top_words)}")
    print()


# ── 6. Sample Predictions Table ──────────────────────────────────────────────

SAMPLE_ADS = [
    ("Toyota Camry 2020 low mileage GCC specs urgent sale", "cars"),
    ("2 bedroom apartment Dubai Marina sea view furnished", "property"),
    ("iPhone 15 Pro Max 256GB excellent condition full box", "electronics"),
    ("Software engineer Python React Dubai startup hiring", "jobs"),
    ("Nissan Patrol V8 single owner service history available", "cars"),
    ("Studio for rent near Deira metro bills included", "property"),
    ("MacBook Pro M2 chip 16GB RAM original charger box", "electronics"),
    ("Sales executive real estate commission training provided", "jobs"),
]

def sample_predictions(model, vectorizer):
    from utils import clean_text
    print("=" * 55)
    print("  SAMPLE PREDICTIONS")
    print("=" * 55)
    print(f"  {'Ad (truncated)':<40} {'Actual':<12} {'Predicted':<12} {'✓'}")
    print("  " + "-" * 68)
    for ad, actual in SAMPLE_ADS:
        cleaned = clean_text(ad)
        pred = model.predict(vectorizer.transform([cleaned]))[0]
        check = "✅" if pred == actual else "❌"
        print(f"  {ad[:38]:<40} {actual:<12} {pred:<12} {check}")
    print()


# ── 7. Save Artifacts ────────────────────────────────────────────────────────

def save_artifacts(model, vectorizer):
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("💾 Saved: model.joblib + vectorizer.joblib")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Dubizzle Smart Ad Classifier — Training\n")
    df = load_data()
    model, vectorizer = train_and_evaluate(df)
    save_artifacts(model, vectorizer)
    print("\n✅ Training complete. Run `streamlit run app.py` to launch the UI.\n")
