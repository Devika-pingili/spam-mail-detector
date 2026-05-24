# -*- coding: utf-8 -*-
"""
Train TF-IDF + Multinomial Naive Bayes spam classifier.
Loads dataset/spam.csv, evaluates metrics, saves model.pkl and vectorizer.pkl.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load SMS spam CSV and normalize columns to label + message."""
    df = pd.read_csv(path, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    elif "label" not in df.columns or "message" not in df.columns:
        raise ValueError(
            "Dataset must have columns 'label' and 'message' (or 'v1' and 'v2')."
        )

    df = df.dropna(subset=["message"])
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_map = {"spam": 1, "ham": 0}
    df["target"] = df["label"].map(label_map)
    invalid = df["target"].isna()
    if invalid.any():
        df = df[~invalid].copy()

    return df.reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and empty messages after preprocessing."""
    df = df.copy()
    df["clean_message"] = df["message"].apply(preprocess_text)
    df = df[df["clean_message"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_message", "target"])
    return df.reset_index(drop=True)


def print_metrics(y_true, y_pred) -> dict:
    """Print and return evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print("\n" + "=" * 50)
    print("  SPAM DETECTION MODEL - EVALUATION METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("=" * 50)
    print("\nDetailed Classification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Ham (Not Spam)", "Spam"],
            digits=4,
        )
    )
    return metrics


def train_and_save(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Full training pipeline: load, clean, vectorize, train, evaluate, save."""
    print("Loading dataset...")
    df = load_dataset(dataset_path)
    print(
        f"  Loaded {len(df)} messages "
        f"({df['target'].sum()} spam, {len(df) - df['target'].sum()} ham)"
    )

    print("Cleaning and preprocessing...")
    df = clean_dataset(df)
    print(f"  After cleaning: {len(df)} messages")

    X = df["clean_message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = print_metrics(y_test, y_pred)

    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saving vectorizer to {vectorizer_path}...")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    metrics_path = BASE_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    print("\nTraining complete. Artifacts saved successfully.")
    return metrics


if __name__ == "__main__":
    train_and_save()
