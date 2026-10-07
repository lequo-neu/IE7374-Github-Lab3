# train_and_save_model.py
# IE7374 Lab 3 — Modified from original
# Changes:
#   - Dataset: Iris -> Wine dataset (178 samples, 13 features, 3 classes)
#   - Model: RandomForestClassifier -> GradientBoostingClassifier
#   - Metrics: accuracy only -> accuracy + precision logged to console
#   - Cloud artifact: model only -> model + companion metadata JSON, both uploaded to GCS
#   - Post-upload: verify blob existence in GCS before reporting success

import json
from datetime import datetime

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split


def download_data():
    from sklearn.datasets import load_wine
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target)
    return features, target, wine.target_names.tolist()


def preprocess_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted"), 4),
    }


def upload_to_gcs(bucket_name, local_path, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def verify_blob_exists(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise RuntimeError(f"Verification failed: {blob_name} not found in {bucket_name}")
    blob.reload()
    size_kb = blob.size / 1024
    print(f"Verified: {blob_name} ({size_kb:.1f} KB) is accessible in GCS.")


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bucket_name = "ie7374-lab3-kevin"
    model_blob = f"trained_models/model_{timestamp}.joblib"
    metadata_blob = f"trained_models/model_{timestamp}_metadata.json"

    print(f"Run timestamp: {timestamp}")

    X, y, class_names = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)} samples.")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy:           {metrics['accuracy']}")
    print(f"Precision weighted: {metrics['precision_weighted']}")

    metadata = {
        "timestamp": timestamp,
        "dataset": "Wine (sklearn)",
        "model": "GradientBoostingClassifier",
        "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1},
        "classes": class_names,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "gcs_model_path": f"gs://{bucket_name}/{model_blob}",
    }

    joblib.dump(model, "model.joblib")
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nUploading model to {model_blob} ...")
    upload_to_gcs(bucket_name, "model.joblib", model_blob)

    print(f"Uploading metadata to {metadata_blob} ...")
    upload_to_gcs(bucket_name, "model_metadata.json", metadata_blob)

    print("\nVerifying uploads ...")
    verify_blob_exists(bucket_name, model_blob)
    verify_blob_exists(bucket_name, metadata_blob)

    print(f"\nAll artifacts successfully stored:")
    print(f"  gs://{bucket_name}/{model_blob}")
    print(f"  gs://{bucket_name}/{metadata_blob}")


if __name__ == "__main__":
    main()
