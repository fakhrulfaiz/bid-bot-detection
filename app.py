import io
import joblib
import pandas as pd
import yaml

# Must be imported before joblib.load so pickle can locate the custom transformer
from src.datascience.components.data_transformation import BidBotFeatureEngineer  # noqa: F401

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Bid Bot Detector")
templates = Jinja2Templates(directory="templates")

BIDS_PATH = "data/raw/bids.csv"
PREPROCESSOR_PATH = "artifacts/data_transformation/preprocessor.joblib"
MODEL_PATH = "artifacts/model_trainer/model.joblib"
SCHEMA_PATH = "schema.yaml"
PARAMS_PATH = "params.yaml"


def _feature_columns() -> list[str]:
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    target = schema["TARGET_COLUMN"]["name"]
    return [c for c in schema["COLUMNS"] if c != target]


def _threshold() -> float:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f).get("prediction_threshold", 0.4)


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    bidder_df = pd.read_csv(io.BytesIO(raw))
    bidder_ids = bidder_df["bidder_id"].unique()

    bids = pd.read_csv(BIDS_PATH)
    bidder_bids = bids[bids["bidder_id"].isin(bidder_ids)]

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    feat_cols = _feature_columns()

    engineer = preprocessor.named_steps["engineer"]
    scaler = preprocessor.named_steps["scaler"]

    features_df = engineer.transform(bidder_bids)
    features_df = features_df.reindex(bidder_ids).fillna(0)
    X_scaled = scaler.transform(features_df[feat_cols])

    y_prob = model.predict_proba(X_scaled)[:, 1]
    predictions = (y_prob >= _threshold()).astype(float)

    result = pd.DataFrame({"bidder_id": bidder_ids, "prediction": predictions})

    buf = io.StringIO()
    result.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/train")
async def train():
    try:
        from src.datascience.pipeline.training_pipeline import run
        run()
        return "Training complete."
    except Exception as exc:
        return f"Training failed: {exc}"
