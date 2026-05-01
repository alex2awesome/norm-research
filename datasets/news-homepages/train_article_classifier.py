#!/usr/bin/env python3
"""
Train an article-vs-furniture classifier on Ben Welsh's storysniffer labeled data.

Uses URL path + anchor text features with LogisticRegression.
Outputs a sklearn pipeline that can bulk-predict on DataFrames.

Usage:
    python train_article_classifier.py

    # Then in your pipeline:
    import joblib
    clf = joblib.load("article_classifier.joblib")
    predictions = clf.predict(df[["url_path", "text"]])  # array of True/False
"""
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def load_data(path="storysniffer_labeled.csv"):
    """Load Ben Welsh's labeled data."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_raw = row.get("is_story", "").strip().upper()
            if label_raw not in ("TRUE", "FALSE"):
                continue
            label = 1 if label_raw == "TRUE" else 0

            url = row.get("url", "").strip()
            text = row.get("text", "").strip()

            # Extract URL path (strip domain)
            try:
                parsed = urlparse(url if url.startswith("http") else "https://example.com" + url)
                url_path = parsed.path or "/"
            except Exception:
                url_path = url

            rows.append({
                "url_path": url_path,
                "text": text,
                "url_full": url,
                "label": label,
            })
    return rows


def build_features_and_labels(rows):
    """Build feature matrix and label vector."""
    import pandas as pd
    df = pd.DataFrame(rows)
    X = df[["url_path", "text"]]
    y = df["label"].values
    return X, y, df


def train_and_evaluate():
    """Train multiple models and pick the best."""
    rows = load_data()
    X, y, df = build_features_and_labels(rows)

    print("Loaded %d examples (%d stories, %d furniture)" % (
        len(rows), sum(y), len(y) - sum(y)
    ))
    print()

    # Feature extraction: character n-grams on URL path + text
    url_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        max_features=20000,
    )
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        max_features=20000,
    )

    preprocessor = ColumnTransformer([
        ("url_ngrams", url_vectorizer, "url_path"),
        ("text_ngrams", text_vectorizer, "text"),
    ])

    # Try LogisticRegression and MultinomialNB
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "LogisticRegression_C10": LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced"),
        "LogisticRegression_C01": LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name = None
    best_score = 0
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("features", preprocessor),
            ("classifier", model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
        mean_f1 = scores.mean()
        acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        mean_acc = acc_scores.mean()

        print("%-30s  F1=%.4f (+/- %.4f)  Acc=%.4f" % (name, mean_f1, scores.std(), mean_acc))

        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = name
            best_pipeline = pipeline

    print()
    print("Best: %s (F1=%.4f)" % (best_name, best_score))
    print()

    # Train best model on full data and show classification report
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    print("=== Full-data classification report ===")
    print(classification_report(y, y_pred, target_names=["furniture", "story"]))

    # Show some errors
    errors = []
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            errors.append({
                "true": "story" if y[i] == 1 else "furniture",
                "pred": "story" if y_pred[i] == 1 else "furniture",
                "text": df.iloc[i]["text"][:50],
                "url": df.iloc[i]["url_path"][:60],
            })
    print("Errors on training data: %d / %d" % (len(errors), len(y)))
    print("Sample errors:")
    for e in errors[:10]:
        print("  true=%s pred=%s | [%s] %s" % (e["true"], e["pred"], e["text"], e["url"]))

    # Save
    outpath = "article_classifier.joblib"
    joblib.dump(best_pipeline, outpath)
    print()
    print("Saved pipeline to %s" % outpath)

    # Demo bulk prediction
    print()
    print("=== Demo bulk prediction ===")
    import pandas as pd
    test_df = pd.DataFrame([
        {"url_path": "/2025/01/15/us/politics/trump-rally.html", "text": "Trump Holds Rally in Iowa"},
        {"url_path": "/section/us", "text": "U.S."},
        {"url_path": "/subscription", "text": "Subscribe for $1/week"},
        {"url_path": "/2025/12/31/world/europe/ukraine-war.html", "text": "Another New Year at War"},
        {"url_path": "/about/newsteam/", "text": "Meet the News Team"},
    ])
    preds = best_pipeline.predict(test_df[["url_path", "text"]])
    for i, row in test_df.iterrows():
        print("  %s -> %s | [%s] %s" % (
            "STORY" if preds[i] == 1 else "FURN ",
            row["url_path"][:40],
            row["text"][:30],
            "",
        ))

    return best_pipeline


if __name__ == "__main__":
    train_and_evaluate()
