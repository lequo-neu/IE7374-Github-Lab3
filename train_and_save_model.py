# train_and_save_model.py
# IE7374 Lab 3 — Modified from original
# Changes:
#   - Dataset: Iris → Wine dataset
#   - Model: RandomForestClassifier → GradientBoostingClassifier
#   - Metrics: accuracy only → accuracy + precision
#   - bucket_name: updated to Kevin's bucket

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from google.cloud import storage
import joblib
from datetime import datetime


def download_data():
    from sklearn.datasets import load_wine      # modified: Iris → Wine
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target)
    return features, target


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # modified: GradientBoostingClassifier instead of RandomForest
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename('model.joblib')


def main():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    # modified: thêm precision
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")

    # Change bucket_name into bucket Kevin which created at step 6
    bucket_name = "ie7374-github-lab3-kevin"
    timestamp   = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name   = f"trained_models/model_{timestamp}.joblib"

    save_model_to_gcs(model, bucket_name, blob_name)
    print(f"Model saved to gs://{bucket_name}/{blob_name}")


if __name__ == "__main__":
    main()