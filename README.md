# Bid Bot Detection — MLOps Pipeline

Binary classification system that identifies automated bidding bots in online auction data. Built as a full MLOps pipeline with feature engineering, XGBoost training, MLflow tracking, and a FastAPI inference server.

---

## Problem

Online auction platforms are vulnerable to shill bidding by automated bots. This project classifies bidders as human (`0`) or bot (`1`) using behavioural features aggregated from raw bid-level data.

**Dataset**: [Facebook Recruiting IV: Human or Robot](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-robot)

---

## Project Structure

```
bid_bot_detection/
├── app.py                          # FastAPI inference server
├── main.py                         # Entry point for full pipeline retraining
├── config/
│   └── config.yaml                 # Artifact paths and data paths
├── params.yaml                     # Model hyperparameters and prediction threshold
├── schema.yaml                     # Feature schema and target column
├── data/
│   └── raw/
│       ├── train.csv               # Bidder labels (bidder_id, outcome)
│       ├── test.csv                # Test bidders for submission
│       └── bids.csv                # Raw bid-level events (7.6M rows)
├── src/datascience/
│   ├── components/
│   │   ├── data_transformation.py  # Feature engineering + preprocessing pipeline
│   │   ├── model_trainer.py        # XGBoost training
│   │   └── model_evaluation.py     # Metrics, ROC curve, MLflow logging
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all stages
│   │   ├── stage_02_data_transformation.py
│   │   ├── stage_03_model_trainer.py
│   │   └── stage_04_model_evaluation.py
│   ├── config/configuration.py     # ConfigurationManager
│   ├── entity/config_entity.py     # Typed config dataclasses
│   ├── utils/common.py             # read_yaml, save_json helpers
│   ├── logger.py                   # Centralised logger
│   └── exception.py                # BidBotException
├── research/
│   ├── 02_data_transformation.ipynb
│   └── 03_model_training.ipynb     # GridSearchCV experiments + MLflow
├── artifacts/                      # Generated at runtime (gitignored)
│   ├── data_transformation/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── preprocessor.joblib     # BidBotFeatureEngineer + RobustScaler
│   ├── model_trainer/
│   │   └── model.joblib
│   └── model_evaluation/
│       ├── metrics.json
│       └── roc_curve.png
└── templates/
    └── index.html                  # FastAPI Jinja UI
```

---

## Engineered Features

Raw bids data is aggregated to the bidder level inside a custom sklearn transformer (`BidBotFeatureEngineer`):

| Feature            | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `mean_time_diff`   | Average time gap between a bidder's consecutive bids       |
| `total_bids`       | Total number of bids placed                                |
| `total_auctions`   | Number of distinct auctions participated in                |
| `bids_per_auction` | `total_bids / total_auctions`                              |
| `mean_response`    | Mean response time to the previous bid in the same auction |
| `min_response`     | Minimum response time in any auction                       |
| `ip_entropy`       | Shannon entropy of IP address distribution                 |
| `url_entropy`      | Shannon entropy of URL distribution                        |

Features are scaled with `RobustScaler` (chosen for outlier robustness). The full transformer + scaler is saved as `preprocessor.joblib` for consistent inference.

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/fakhrulfaiz/bid-bot-detection.git
cd bid_bot_detection
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Add raw data

Place these three files in `data/raw/`:

- `train.csv` — bidder labels
- `test.csv` — test bidders
- `bids.csv` — raw bid events

### 3. Configure MLflow (optional)

Create a `.env` file in the project root for remote MLflow tracking (e.g. DagsHub):

```
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
```

Without a `.env`, MLflow logs to a local `mlruns/` directory automatically.

---

## Training

Run the full pipeline (data transformation → model training → evaluation):

```bash
dvc repro
# or equivalently:
python main.py
```

DVC only re-runs stages whose dependencies changed. If only `params.yaml` changed, it skips the 5-minute data transformation and re-runs model training and evaluation only.

### Sync data and artifacts with Google Drive

```bash
# Push local cache (data + artifacts) to GDrive
dvc push

# Pull on a new machine (gets data + artifacts)
dvc pull
```

Raw data files (`train.csv`, `test.csv`, `bids.csv`) and all pipeline artifacts (`preprocessor.joblib`, `model.joblib`, etc.) are stored on Google Drive. Only the small `.dvc` pointer files and `dvc.lock` are committed to git.

---

## Inference Server

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

### Predict

Upload a CSV in `test.csv` format (must have a `bidder_id` column). The server:

1. Loads `data/raw/bids.csv` from disk
2. Filters bids to the uploaded bidder IDs
3. Runs `BidBotFeatureEngineer` → `RobustScaler` → `XGBClassifier`
4. Applies prediction threshold (default `0.4`, set in `params.yaml`)
5. Returns a downloadable `predictions.csv` in submission format

### Retrain

Click **Retrain Model** in the UI or call `GET /train`. Reruns the full pipeline and updates all artifacts.

> **Note**: After retraining from a Jupyter notebook, always restart the server so the reloaded `preprocessor.joblib` is picked up.

---

## Model

|                       |                                                              |
| --------------------- | ------------------------------------------------------------ |
| Algorithm             | XGBoost (`XGBClassifier`)                                    |
| Hyperparameter search | `GridSearchCV` (5-fold, ROC AUC) — research notebook only    |
| Production params     | Fixed in `params.yaml`                                       |
| Class imbalance       | `scale_pos_weight` computed dynamically from training labels |
| Scaler                | `RobustScaler`                                               |
| Prediction threshold  | `0.4` (tuned for recall on bots)                             |
| ROC AUC               | ~0.91                                                        |

Hyperparameters are set in `params.yaml`:

```yaml
prediction_threshold: 0.4

XGBoost:
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
```

---

## MLflow Tracking

Each evaluation run logs to MLflow:

- Model parameters
- ROC AUC
- Per-threshold precision, recall, F1, accuracy
- ROC curve plot artifact
- Trained XGBoost model

View locally: `mlflow ui` → `http://127.0.0.1:5000`

Remote tracking is configured via `MLFLOW_TRACKING_URI` in `.env`.

---
