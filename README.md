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

**Pipeline (train-and-upload.yml)**

The single training step was renamed to reflect its expanded scope: train,
evaluate, and upload artifacts to GCS. Pip dependency caching was kept to
speed up repeated runs. The authentication step uses google-github-actions/auth
at v2, which is the current stable version of that action.

## Advanced Extension — Companion Metadata Upload and Post-Upload Verification

**Why this was chosen**

The original lab uploaded a model file to GCS and stopped. This creates two
problems that compound over time. First, a bucket full of anonymous joblib files
is unmanageable: there is no way to know which run produced which file, what
dataset it was trained on, what metrics it achieved, or where its GCS path is
without digging through pipeline logs. Second, there was no confirmation that
the upload actually succeeded. GCS upload calls can fail silently due to network
timeouts or permission issues, and without an explicit check the pipeline would
report success while the bucket remained unchanged.

Both problems are standard concerns in production MLOps systems. Artifact
lineage, the ability to trace a deployed model back to the exact run, data, and
parameters that produced it, is a core requirement for reproducibility and
auditability. Post-upload verification is a basic reliability practice: no
responsible system promotes an artifact to a registry without confirming it
arrived intact. These two additions together bring the lab's GCS integration
closer to what a real ML platform would require.

**How it works**

After evaluation, the script constructs a metadata dictionary that captures
everything needed to understand the artifact: the run timestamp, the dataset
name and class names, the model class and its hyperparameters, the number of
training and test samples, the computed accuracy and precision scores, and the
full GCS path where the model will be stored. This dictionary is written to a
local JSON file named model_TIMESTAMP_metadata.json, then uploaded to GCS
immediately after the model file, using the same trained_models/ prefix and
the same timestamp so the two files are always co-located and trivially paired.

After both uploads complete, the script calls the GCS API a second time to
reload each blob's metadata. If a blob does not exist at the expected path,
the reload raises an exception and the script raises a RuntimeError with a
message naming the missing file. If both blobs exist, their sizes in kilobytes
are printed to the console as confirmation. The pipeline only reaches the final
success message if both files passed verification, so any failure in the upload
or authentication chain surfaces as a clear, named error rather than a silent
no-op.

**What it produces**

Every run leaves two paired files in the GCS bucket: a joblib model and a JSON
metadata file sharing the same timestamp prefix. Anyone opening the bucket can
read the metadata file directly in the GCP Console browser and immediately know
the model's provenance without checking any external log. The verification step
also means that a green pipeline run is a genuine guarantee of artifact
availability, not just evidence that the upload function was called.

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
