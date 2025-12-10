# main.py
# top — MUST come before importing tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only; prevents TF from trying to init CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF logs

import pickle
import numpy as np
import tensorflow as tf
from typing import Union, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

# optional weather proxy
try:
    import requests
except Exception:
    requests = None

# ---------- Request schema ----------
class PredictRequest(BaseModel):
    SOIL: str
    # SOWN can be a month number (1-12) or a season string like 'kharif'
    SOWN: Union[int, str]
    SOIL_PH: float
    TEMP: float
    RELATIVE_HUMIDITY: float
    N: float
    P: float
    K: float

    @validator("SOWN")
    def validate_sown(cls, v):
        # Accept int months 1..12 or non-empty string
        if isinstance(v, int):
            if not (1 <= v <= 12):
                raise ValueError("SOWN month must be between 1 and 12")
        elif isinstance(v, str):
            if not v:
                raise ValueError("SOWN string must be non-empty")
        else:
            raise ValueError("SOWN must be integer month (1-12) or season string")
        return v

# ---------- App setup ----------
app = FastAPI(title="Crop Sage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- Globals for artifacts ----------
model: Optional[tf.keras.Model] = None
soil_encoder = None
sown_encoder = None
water_source_encoder = None
crop_name_encoder = None
scaler = None
cd_scaler = None
wr_scaler = None

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_FILE = os.environ.get("MODEL_FILE", "crop_model.keras")

# ---------- Helpers ----------
def month_to_season_label(month: int) -> str:
    """
    Map month number (1-12) to season label used by the encoder.
    Mapping (India typical):
      - Mar(3) - May(5)  -> 'zaid'
      - Jun(6) - Sep(9)  -> 'kharif'
      - Oct(10) - Feb(2) -> 'rabi'
    """
    if not isinstance(month, int):
        raise ValueError("month must be int")
    if 3 <= month <= 5:
        return "zaid"
    if 6 <= month <= 9:
        return "kharif"
    # months 10,11,12,1,2
    return "rabi"

def ensure_loaded():
    """Raise helpful error if artifacts not loaded"""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Check server startup logs.")

# ---------- Startup: load artifacts ----------
@app.on_event("startup")
def load_artifacts():
    global model, soil_encoder, sown_encoder, water_source_encoder, crop_name_encoder, scaler, cd_scaler, wr_scaler
    base = ART_DIR

    # load model
    model_path = os.path.join(base, MODEL_FILE)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    # load pickled encoders & scalers
    def _load_pickle(name):
        p = os.path.join(base, name)
        if not os.path.exists(p):
            raise RuntimeError(f"Required artifact missing: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    soil_encoder = _load_pickle("soil_encoder.pkl")
    sown_encoder = _load_pickle("sown_encoder.pkl")
    water_source_encoder = _load_pickle("water_source_encoder.pkl")
    crop_name_encoder = _load_pickle("crop_name_encoder.pkl")
    scaler = _load_pickle("scaler.pkl")
    cd_scaler = _load_pickle("cd_scaler.pkl")
    wr_scaler = _load_pickle("wr_scaler.pkl")

# ---------- Simple health endpoint ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Core prediction logic (shared) ----------
def do_predict_logic(payload: PredictRequest) -> Dict[str, Any]:
    """
    Shared prediction routine; accepts PredictRequest and returns structured result.
    Handles SOWN as month number or string by mapping to encoder's expected category.
    """
    ensure_loaded()

    # Normalize and encode SOIL
    soil_val = str(payload.SOIL).strip()
    if soil_val == "":
        raise HTTPException(status_code=400, detail="SOIL must be a non-empty string")
    try:
        soil_enc = int(soil_encoder.transform([soil_val])[0])
    except Exception as e:
        # Helpful message if unknown category
        raise HTTPException(status_code=400, detail=f"Unknown SOIL category '{soil_val}': {e}")

    # Handle SOWN: month number or existing season string
    sown_input = payload.SOWN
    if isinstance(sown_input, int):
        # convert month -> season label
        try:
            sown_label = month_to_season_label(int(sown_input))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid SOWN month: {e}")
    else:
        sown_label = str(sown_input).strip()

    if sown_label == "":
        raise HTTPException(status_code=400, detail="SOWN must be a valid month (1-12) or non-empty season string")

    try:
        sown_enc = int(sown_encoder.transform([sown_label])[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown SOWN category '{sown_label}': {e}")

    # Numeric array and scaling (same numeric columns used in training)
    numeric = np.array([[payload.SOIL_PH, payload.TEMP, payload.RELATIVE_HUMIDITY, payload.N, payload.P, payload.K]], dtype=np.float32)
    try:
        numeric_scaled = scaler.transform(numeric)  # scaler expects shape (n_samples, n_numeric_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

    # Build final feature vector: [soil_enc, sown_enc, numeric_scaled...]
    try:
        features = np.concatenate([[soil_enc, sown_enc], numeric_scaled[0]]).reshape(1, -1).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature vector build failed: {e}")

    # Predict
    try:
        preds = model.predict(features, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Model is expected to output a list-like of 4 outputs:
    # preds[0] -> crop softmax (shape (1, n_crop_classes))
    # preds[1] -> water source softmax (shape (1, n_ws_classes))
    # preds[2] -> crop_duration scaled (shape (1, 1))
    # preds[3] -> water_required scaled (shape (1, 1))
    try:
        crop_prob = preds[0][0]
        ws_prob = preds[1][0]
        cd_scaled = np.array(preds[2]).reshape(-1, 1)
        wr_scaled = np.array(preds[3]).reshape(-1, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape or type: {e}")

    # indices -> labels
    crop_idx = int(np.argmax(crop_prob))
    ws_idx = int(np.argmax(ws_prob))

    try:
        crop_label = crop_name_encoder.inverse_transform([crop_idx])[0]
    except Exception:
        # Some encoders map differently; try safe fallback
        try:
            crop_label = str(crop_name_encoder.classes_[crop_idx])
        except Exception:
            crop_label = str(crop_idx)

    try:
        ws_label = water_source_encoder.inverse_transform([ws_idx])[0]
    except Exception:
        try:
            ws_label = str(water_source_encoder.classes_[ws_idx])
        except Exception:
            ws_label = str(ws_idx)

    # inverse-scale regression outputs
    try:
        cd_orig = float(cd_scaler.inverse_transform(cd_scaled)[0, 0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop-duration inverse-scaling failed: {e}")
    try:
        wr_orig = float(wr_scaler.inverse_transform(wr_scaled)[0, 0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Water-required inverse-scaling failed: {e}")

    result = {
        "crop": {"label": str(crop_label), "confidence": float(crop_prob[crop_idx])},
        "water_source": {"label": str(ws_label), "confidence": float(ws_prob[ws_idx])},
        "crop_duration": cd_orig,
        "water_required": wr_orig
    }
    return result

# ---------- Endpoints ----------
@app.post("/predict")
def predict(req: PredictRequest):
    return do_predict_logic(req)

@app.get("/labels")
def labels():
    ensure_loaded()
    return {
        "crop_classes": list(getattr(crop_name_encoder, "classes_", [])),
        "water_source_classes": list(getattr(water_source_encoder, "classes_", []))
    }

# API aliases expected by frontend
@app.get("/api/labels")
def api_labels():
    return labels()

@app.get("/api/soil-types")
def api_soil_types():
    ensure_loaded()
    # soil_encoder.classes_ expected to be an array-like of soil labels
    classes = getattr(soil_encoder, "classes_", None)
    if classes is None:
        # if encoder is a dict or custom, try to fall back to keys if available
        try:
            # e.g., if it's a mapping object with `.classes_` missing
            return {"soil_types": list(soil_encoder)}
        except Exception:
            raise HTTPException(status_code=500, detail="Soil encoder does not expose classes.")
    return {"soil_types": list(classes)}

@app.post("/api/predict-crop")
def api_predict_crop(req: PredictRequest):
    return do_predict_logic(req)

# Optional weather proxy endpoint
@app.get("/api/weather")
def api_weather(lat: float, lon: float):
    if requests is None:
        raise HTTPException(status_code=501, detail="Weather proxy requires 'requests' package on the server.")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {e}")
    return r.json()
