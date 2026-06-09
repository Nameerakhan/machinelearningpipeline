# Data Pipeline with DVC and MLflow for Machine Learning

This repository demonstrates a reproducible, end-to-end machine learning
pipeline using DVC (Data Version Control) for data and model versioning and
MLflow for experiment tracking. The example trains a Random Forest Classifier
on the Pima Indians Diabetes dataset and provides clear stages for data
preprocessing, training, and evaluation.

**Contents**
- **Overview**: What this project does and why.
- **Key features**: DVC + MLflow integration and reproducible stages.
- **Quickstart**: Install, configure, and run the pipeline locally.
- **Usage**: How to run experiments and inspect results.

## Overview

This pipeline is split into explicit stages (preprocessing, training,
evaluation) managed by DVC. Each stage captures inputs, outputs, and
parameters so you can reproduce any experiment by re-running the pipeline.

MLflow is used to log metrics, parameters, and model artifacts for easy
comparison of runs and model promotion.

## Key Features

- Reproducible pipeline stages with `dvc repro`.
- Data and model versioning via DVC (supports remote storage like S3/DagsHub).
- Experiment tracking and model logging with MLflow UI.
- Example model: Random Forest classifier on the Pima Indians Diabetes dataset.

## Repository Structure

- `mlpipeline/` — pipeline scripts and DVC stage definitions.
- `data/` — raw and processed datasets (tracked by DVC).
- `models/` — trained model artifacts (tracked by DVC/MLflow).
- `README.md` — this file.

Adjust paths above if you organized files differently in your workspace.

## Requirements

- Python 3.8+
- pip
- DVC (installed globally or in the virtualenv)
- MLflow

Install Python dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
pip install dvc mlflow
```

## Quickstart — initialize and run

1. Initialize DVC (if not already initialized in the repo):

```bash
cd mlpipeline
dvc init
```

2. (Optional) Configure DVC remote storage for large datasets and models:

```bash
dvc remote add -d myremote s3://my-bucket/path   # or use DagsHub, etc.
dvc push
```

3. Run the full pipeline (reproduces all stages):

```bash
cd mlpipeline
dvc repro
```

4. View MLflow UI to compare runs:

```bash
mlflow ui --host 0.0.0.0 --port 5000
# then open http://localhost:5000 in your browser
```

## Typical workflow

- Make a change to a preprocessing, training script, or parameters.
- Run `dvc repro` to re-run affected stages.
- Use `dvc metrics show` or MLflow UI to inspect metrics.
- Commit parameter and pipeline changes to Git; use `dvc push` to upload
  large data/model artifacts to the DVC remote.

## DVC notes

- Each DVC stage should be declared in `dvc.yaml` under `mlpipeline/`.
- Use `dvc add` to track large files/directories (datasets, model artifacts).
- Use `dvc checkout` to restore data for a given Git commit.

## MLflow notes

- MLflow runs are logged by the training stage; models and metrics are
  recorded for comparison and deployment.
- You can change the MLflow tracking URI via environment variable
  `MLFLOW_TRACKING_URI` to point to a remote server if needed.

## Data source

The example uses the Pima Indians Diabetes dataset (public). Ensure any
datasets used comply with their licenses and privacy requirements.

## Reproducibility and CI

- Pin package versions in `requirements.txt` for reproducible
  environments.
- In CI, use `dvc pull` to fetch datasets before running `dvc repro`.

## License

This project is provided as an example. Add an appropriate LICENSE file
if you intend to publish or distribute the code.

---

If you'd like, I can also:
- add a `requirements.txt` or `pyproject.toml` in `mlpipeline/`;
- initialize or configure a DVC remote (S3/DagsHub) and add CI steps;
- commit the README changes and create a Git tag.

Tell me which next step you'd like.
