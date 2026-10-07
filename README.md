# IE7374 Lab 3 — GitHub Actions and Google Cloud Platform Integration

## Overview

This lab connects a GitHub Actions pipeline to Google Cloud Platform so that
every pipeline run produces a trained model and uploads it — along with a
companion metadata file — directly to a Google Cloud Storage bucket. The
pipeline authenticates with GCP using a service account, trains the model,
evaluates it, uploads both artifacts, and then verifies that each file is
actually accessible in the bucket before reporting success. The result is a
fully automated cloud artifact management system that runs on demand or on a
daily schedule.

## Changes Made to the Lab

The original lab trained a RandomForestClassifier on the Iris dataset,
uploaded only the model file to GCS, and stopped there with no confirmation
that the upload succeeded.

The changes fall into two categories.

**Source code (train_and_save_model.py)**

The dataset was changed from Iris to the Wine dataset, which has 178 samples,
13 numeric features, and 3 target classes representing wine cultivars. The
model was changed to GradientBoostingClassifier with n_estimators=100 and
learning_rate=0.1. Evaluation was extended from accuracy alone to accuracy and
weighted precision, both computed on a proper 80/20 train/test split.

Beyond the model itself, the script now generates a companion metadata JSON
file for every run. The metadata captures the timestamp, dataset name, model
class, hyperparameters, class names, sample counts, and the computed metrics,
together with the GCS path where the model was stored. This metadata file is
uploaded to GCS alongside the model so that every artifact in the bucket is
self-describing and traceable.

After both uploads, the script runs a verification pass that calls the GCS API
to confirm each blob exists and reports its file size. If either file is
missing, the script raises a RuntimeError and the pipeline fails immediately.
This closes a gap in the original lab where the upload could silently fail
without any signal.

**Pipeline (train-and-upload.yml)**

The single training step was renamed to reflect its expanded scope: train,
evaluate, and upload artifacts to GCS. Pip dependency caching was kept to
speed up repeated runs. The authentication step uses google-github-actions/auth
at v2, which is the current stable version of that action.

## Prerequisites

The following must be available before running locally.

Python 3.10 or later. Download from https://www.python.org/downloads and
verify with: python3 --version

A Google Cloud Platform account with a project, a service account that has
the Storage Admin role, and a GCS bucket named ie7374-lab3-kevin. The service
account JSON key must be downloaded and kept locally but never committed to
the repository.

The GCP_SA_KEY secret must be added to the GitHub repository before the
pipeline can run on GitHub Actions.

## Security Note

The file ie7374-lab3-a6b0350345a8.json contains the GCP service account
credentials. It is excluded by the .gitignore rule *.json and must never be
committed to git or pushed to GitHub. Before running git add, confirm the
key file is not staged by running git status and verifying it does not appear
in the list.

## Environment Setup

Clone the repository.

```
git clone https://github.com/lequo-neu/IE7374-Github-Lab3.git
cd IE7374-Github-Lab3
```

Create and activate a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
```

Install all dependencies. The key packages are scikit-learn 1.x, google-cloud-storage 2.x, joblib 1.x, and pandas 2.x.

```
pip install -r requirements.txt
```

Set up application default credentials so the script can reach GCP locally.
Point the environment variable at the service account key file.

```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/ie7374-lab3-a6b0350345a8.json"
```

## Running the Code Locally

With the virtual environment active and credentials exported, run the script directly.

```
python train_and_save_model.py
```

The script handles everything in sequence: loading data, splitting, training,
evaluating, writing the local model and metadata files, uploading both to GCS,
and verifying the uploads.

## What a Successful Run Looks Like

The terminal output will follow this pattern.

```
Run timestamp: 20261007153000
Training on 142 samples, evaluating on 36 samples.
Accuracy:           0.9722
Precision weighted: 0.9740

Uploading model to trained_models/model_20261007153000.joblib ...
Uploading metadata to trained_models/model_20261007153000_metadata.json ...

Verifying uploads ...
Verified: trained_models/model_20261007153000.joblib (X.X KB) is accessible in GCS.
Verified: trained_models/model_20261007153000_metadata.json (X.X KB) is accessible in GCS.

All artifacts successfully stored:
  gs://ie7374-lab3-kevin/trained_models/model_20261007153000.joblib
  gs://ie7374-lab3-kevin/trained_models/model_20261007153000_metadata.json
```

On GitHub, after triggering the workflow from the Actions tab, all steps
including "Train, evaluate, and upload artifacts to GCS" should show a green
checkmark. In Google Cloud Console, navigate to Cloud Storage and open the
ie7374-lab3-kevin bucket. Inside the trained_models folder, each run will
have produced two files: a joblib model file and a JSON metadata file, both
named with the same timestamp so they are always paired.

## Project Structure

```
IE7374-Github-Lab3/
    .github/
        workflows/
            train-and-upload.yml
    .gitignore
    README.md
    requirements.txt
    train_and_save_model.py
```

## CI/CD Pipeline Summary

The train-and-upload.yml workflow triggers on workflow_dispatch for manual
runs and on a daily cron schedule at 00:00 UTC. It checks out the code, sets
up Python 3.10, caches pip dependencies to reduce install time on repeated
runs, installs requirements, authenticates with GCP using the GCP_SA_KEY
repository secret, and runs train_and_save_model.py. The script itself handles
training, evaluation, upload, and verification, keeping the workflow file clean
and all logic in Python where it is testable.
