# app.py
# top — MUST come before importing tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only; prevents TF from trying to init CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: reduce TF logs

import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# optional weather proxy library
try:
    import requests
except Exception:
    requests = None

# Request schema
class PredictRequest(BaseModel):
    SOIL: str
    SOWN: str
    SOIL_PH: float
    TEMP: float
    RELATIVE_HUMIDITY: float
    N: float
    P: float
    K: float

app = FastAPI(title="Crop Sage API")

# allow requests from your frontend (use "*" for quick dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Globals for artifacts
model = None
soil_encoder = None
sown_encoder = None
water_source_encoder = None
crop_name_encoder = None
scaler = None
cd_scaler = None
wr_scaler = None

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_FILE = os.environ.get("MODEL_FILE", "crop_model.keras")

@app.on_event("startup")
def load_artifacts():
    global model, soil_encoder, sown_encoder, water_source_encoder, crop_name_encoder, scaler, cd_scaler, wr_scaler
    base = ART_DIR

    # model
    model_path = os.path.join(base, MODEL_FILE)
    model = tf.keras.models.load_model(model_path)

    # pickles
    with open(os.path.join(base, "soil_encoder.pkl"), "rb") as f:
        soil_encoder = pickle.load(f)
    with open(os.path.join(base, "sown_encoder.pkl"), "rb") as f:
        sown_encoder = pickle.load(f)
    with open(os.path.join(base, "water_source_encoder.pkl"), "rb") as f:
        water_source_encoder = pickle.load(f)
    with open(os.path.join(base, "crop_name_encoder.pkl"), "rb") as f:
        crop_name_encoder = pickle.load(f)
    with open(os.path.join(base, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "cd_scaler.pkl"), "rb") as f:
        cd_scaler = pickle.load(f)
    with open(os.path.join(base, "wr_scaler.pkl"), "rb") as f:
        wr_scaler = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

# Internal helper that performs prediction logic given a PredictRequest (so endpoints can reuse)
def do_predict_logic(req: PredictRequest) -> Dict[str, Any]:
    # encode categorical
    try:
        soil_enc = int(soil_encoder.transform([req.SOIL])[0])
        sown_enc = int(sown_encoder.transform([req.SOWN])[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown category: {e}")

    # numeric array and scaling (same columns used in training)
    numeric = np.array([[req.SOIL_PH, req.TEMP, req.RELATIVE_HUMIDITY, req.N, req.P, req.K]], dtype=np.float32)
    numeric_scaled = scaler.transform(numeric)        # scaler was fit on numeric cols only

    # build final feature vector [SOIL_enc, SOWN_enc, numeric_scaled...]
    features = np.concatenate([[soil_enc, sown_enc], numeric_scaled[0]]).reshape(1, -1).astype(np.float32)

    # predict
    preds = model.predict(features, verbose=0)

    # The original code assumed model.predict returns a list of outputs:
    # preds[0] -> crop softmax (shape (1, n_crop_classes))
    # preds[1] -> water source softmax (shape (1, n_ws_classes))
    # preds[2] -> crop_duration scaled (shape (1, 1))
    # preds[3] -> water_required scaled (shape (1, 1))
    try:
        crop_prob = preds[0][0]    # e.g. array of crop probabilities
        ws_prob = preds[1][0]      # array of water-source probabilities
        cd_scaled = np.array(preds[2]).reshape(-1, 1)
        wr_scaled = np.array(preds[3]).reshape(-1, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {e}")

    # indices -> labels
    crop_idx = int(np.argmax(crop_prob))
    ws_idx = int(np.argmax(ws_prob))
    crop_label = crop_name_encoder.inverse_transform([crop_idx])[0]
    ws_label = water_source_encoder.inverse_transform([ws_idx])[0]

    # inverse-scale regression outputs
    cd_orig = float(cd_scaler.inverse_transform(cd_scaled)[0, 0])
    wr_orig = float(wr_scaler.inverse_transform(wr_scaled)[0, 0])

    return {
        "crop": {"label": str(crop_label), "confidence": float(crop_prob[crop_idx])},
        "water_source": {"label": str(ws_label), "confidence": float(ws_prob[ws_idx])},
        "crop_duration": cd_orig,
        "water_required": wr_orig
    }

# Keep your original endpoints but make them call the shared logic
@app.post("/predict")
def predict(req: PredictRequest):
    return do_predict_logic(req)

@app.get("/labels")
def labels():
    return {
        "crop_classes": list(crop_name_encoder.classes_),
        "water_source_classes": list(water_source_encoder.classes_)
    }

# ---- API aliases your frontend expects ----

@app.get("/api/labels")
def api_labels():
    return {
        "crop_classes": list(crop_name_encoder.classes_),
        "water_source_classes": list(water_source_encoder.classes_)
    }

@app.get("/api/soil-types")
def api_soil_types():
    # returns list of known soil categories (encoder classes)
    return {"soil_types": list(soil_encoder.classes_)}

@app.post("/api/predict-crop")
def api_predict_crop(req: PredictRequest):
    # reuse same logic
    return do_predict_logic(req)

# Optional: simple weather proxy endpoint that frontend used.
# Example usage: GET /api/weather?lat=22.73&lon=87.51
@app.get("/api/weather")
def api_weather(lat: float, lon: float):
    if requests is None:
        raise HTTPException(status_code=501, detail="Weather proxy requires 'requests' package on the server.")
    # This example uses Open-Meteo (no API key). Modify params as needed.
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {e}")
    return r.json()
