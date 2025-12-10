# backend/main.py
# Flexible artifact loader - tries multiple candidate filenames so repo naming mismatches don't crash startup.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
from typing import Union, Dict, Any, Optional, List
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Crop Sage API - flexible artifacts loader")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# default artifacts dir (can be overridden with ARTIFACTS_DIR env var)
ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

# Candidate filenames (relative to ART_DIR)
MODEL_CANDIDATES = [
    "model.keras",
    "crop_model.keras",
    "best_final.keras",
    "best_stage1.keras",
    "model",          # sometimes a folder named 'model'
]

SCALER_X_CANDIDATES = ["scaler_x.pkl", "scaler.pkl", "scaler_x", "scaler"]
DUR_SCALER_CANDIDATES = ["dur_scaler.pkl", "cd_scaler.pkl", "dur_scaler", "cd_scaler"]
WREQ_SCALER_CANDIDATES = ["wreq_scaler.pkl", "wr_scaler.pkl", "wreq_scaler", "wr_scaler"]

LE_NAME_CANDIDATES = ["le_name.pkl", "crop_name_encoder.pkl", "name_encoder.pkl", "le_name"]
LE_WATER_CANDIDATES = ["le_water.pkl", "water_source_encoder.pkl", "water_encoder.pkl", "le_water"]

SOIL_TO_IDX_CANDIDATES = ["soil_to_idx.pkl", "soil_encoder.pkl", "soil_encoder_map.pkl", "soil_to_idx"]
SOWN_TO_IDX_CANDIDATES = ["sown_to_idx.pkl", "sown_encoder.pkl", "sown_encoder_map.pkl", "sown_to_idx"]

# Globals
model: Optional[tf.keras.Model] = None
scaler_x = None
dur_scaler = None
wreq_scaler = None
le_name = None
le_water = None
soil_to_idx = {}
sown_to_idx = {}

# ---------- helpers ----------
def abs_paths(candidates: List[str]) -> List[str]:
    return [os.path.join(ART_DIR, p) for p in candidates]

def load_pickle_any(candidates: List[str]):
    """
    Try candidates in order, return the first pickle loaded.
    Raises RuntimeError if none exist.
    """
    for p in abs_paths(candidates):
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    # if none found, helpful error
    raise RuntimeError(f"Required artifact missing. Tried: {candidates} in {ART_DIR}")

def find_first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in abs_paths(candidates):
        if os.path.exists(p):
            return p
    return None

def safe_list(obj):
    if obj is None:
        return []
    if hasattr(obj, "classes_"):
        try:
            return list(getattr(obj, "classes_"))
        except Exception:
            pass
    try:
        return list(obj)
    except Exception:
        return []

def month_name_to_number_token(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = s.strip().lower()
    month_map = {
        "jan":"1","january":"1","feb":"2","february":"2","mar":"3","march":"3","apr":"4","april":"4",
        "may":"5","jun":"6","june":"6","jul":"7","july":"7","aug":"8","august":"8","sep":"9","sept":"9","september":"9",
        "oct":"10","october":"10","nov":"11","november":"11","dec":"12","december":"12"
    }
    return month_map.get(m)

def season_to_representative_month_token(season: str) -> Optional[str]:
    if not isinstance(season, str):
        return None
    s = season.strip().lower()
    if s in ("kharif", "khareef"):
        return "7"
    if s in ("rabi",):
        return "11"
    if s in ("zaid", "zayad"):
        return "4"
    return None

def ensure_loaded():
    if model is None or scaler_x is None or dur_scaler is None or wreq_scaler is None:
        raise HTTPException(status_code=503, detail="Model artifacts not fully loaded. Check server logs.")

# ---------- startup: load artifacts ----------
@app.on_event("startup")
def startup_load():
    global model, scaler_x, dur_scaler, wreq_scaler, le_name, le_water, soil_to_idx, sown_to_idx

    # model - try candidate names and load Keras model from first existing
    model_path = find_first_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        raise RuntimeError(f"Model file not found. Tried: {MODEL_CANDIDATES} in {ART_DIR}")
    # tf.keras.models.load_model supports both folder and file-based SavedModel
    model = tf.keras.models.load_model(model_path)

    # pickles & scalers - try candidate names
    scaler_x = load_pickle_any(SCALER_X_CANDIDATES)
    dur_scaler = load_pickle_any(DUR_SCALER_CANDIDATES)
    wreq_scaler = load_pickle_any(WREQ_SCALER_CANDIDATES)
    le_name = load_pickle_any(LE_NAME_CANDIDATES)
    le_water = load_pickle_any(LE_WATER_CANDIDATES)

    # mapping-based artifacts (soil and sown)
    # try to load either direct maps or encoder-like pickles
    soil_to_idx = load_pickle_any(SOIL_TO_IDX_CANDIDATES)
    sown_to_idx = load_pickle_any(SOWN_TO_IDX_CANDIDATES)

# ---------- request schema ----------
class PredictRequest(BaseModel):
    SOIL: str
    SOWN: Union[int, str]
    SOIL_PH: float
    TEMP: float
    RELATIVE_HUMIDITY: float
    N: float
    P: float
    K: float

    @validator("SOWN")
    def validate_sown(cls, v):
        if isinstance(v, int):
            if not (1 <= v <= 12):
                raise ValueError("SOWN month must be 1..12")
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("SOWN must be non-empty string or month number")
        else:
            raise ValueError("SOWN must be integer month or string")
        return v

# ---------- mapping helpers (match training mapping style) ----------
def map_soil_token(soil_raw: str) -> int:
    s = str(soil_raw).strip()
    # direct
    if s in soil_to_idx:
        return int(soil_to_idx[s])
    # lowercase
    if s.lower() in soil_to_idx:
        return int(soil_to_idx[s.lower()])
    # title
    if s.title() in soil_to_idx:
        return int(soil_to_idx[s.title()])
    # fallback unknown index used in training
    return int(len(soil_to_idx))

def map_sown_token(sown_raw: Union[int, str]) -> int:
    # ints -> token string
    if isinstance(sown_raw, int):
        token = str(int(sown_raw))
    else:
        s = str(sown_raw).strip()
        if s.isdigit():
            token = str(int(s))
        else:
            month_token = month_name_to_number_token(s)
            if month_token is not None:
                token = month_token
            else:
                season_token = season_to_representative_month_token(s)
                if season_token is not None:
                    token = season_token
                else:
                    # normalization trials
                    if s in sown_to_idx:
                        token = s
                    elif s.lower() in sown_to_idx:
                        token = s.lower()
                    elif s.title() in sown_to_idx:
                        token = s.title()
                    else:
                        token = s
    if token in sown_to_idx:
        return int(sown_to_idx[token])
    try:
        if token.lstrip("0").isdigit():
            t = str(int(token))
            if t in sown_to_idx:
                return int(sown_to_idx[t])
    except Exception:
        pass
    return int(len(sown_to_idx))

# ---------- prediction routine ----------
def predict_from_payload(payload: PredictRequest) -> Dict[str, Any]:
    ensure_loaded()

    soil_idx = map_soil_token(payload.SOIL)
    sown_idx = map_sown_token(payload.SOWN)

    numeric = np.array([[payload.SOIL_PH, payload.TEMP, payload.RELATIVE_HUMIDITY, payload.N, payload.P, payload.K]], dtype=np.float32)
    try:
        numeric_scaled = scaler_x.transform(numeric).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

    # prepare inputs matching model inputs names used during training
    soil_in = np.array([soil_idx], dtype=np.int32)
    sown_in = np.array([sown_idx], dtype=np.int32)
    num_in = numeric_scaled

    try:
        preds = model.predict({"soil_in": soil_in, "sown_in": sown_in, "num_in": num_in}, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        name_probs = np.array(preds[0])[0]
        water_probs = np.array(preds[1])[0]
        dur_scaled = np.array(preds[2]).reshape(-1)[0]
        wreq_scaled = np.array(preds[3]).reshape(-1)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {e}")

    name_idx = int(np.argmax(name_probs))
    water_idx = int(np.argmax(water_probs))
    name_conf = float(name_probs[name_idx])
    water_conf = float(water_probs[water_idx])

    try:
        crop_label = str(le_name.inverse_transform([name_idx])[0])
    except Exception:
        crop_label = str(safe_list(le_name)[name_idx] if name_idx < len(safe_list(le_name)) else name_idx)

    try:
        water_label = str(le_water.inverse_transform([water_idx])[0])
    except Exception:
        water_label = str(safe_list(le_water)[water_idx] if water_idx < len(safe_list(le_water)) else water_idx)

    try:
        crop_duration = float(dur_scaler.inverse_transform(np.array([[dur_scaled]]))[0, 0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop duration inverse scaling failed: {e}")

    try:
        water_required = float(wreq_scaler.inverse_transform(np.array([[wreq_scaled]]))[0, 0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Water required inverse scaling failed: {e}")

    return {
        "crop": {"label": crop_label, "confidence": name_conf},
        "water_source": {"label": water_label, "confidence": water_conf},
        "crop_duration": crop_duration,
        "water_required": water_required
    }

# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict-crop")
def api_predict_crop(req: PredictRequest):
    return predict_from_payload(req)

@app.post("/predict")
def predict_alias(req: PredictRequest):
    return predict_from_payload(req)

@app.get("/api/soil-types")
def api_soil_types():
    ensure_loaded()
    return {"soil_types": list(soil_to_idx.keys())}

@app.get("/api/labels")
def api_labels():
    ensure_loaded()
    return {
        "crop_classes": safe_list(le_name),
        "water_source_classes": safe_list(le_water)
    }

@app.get("/api/debug/encoders")
def api_debug_encoders():
    ensure_loaded()
    return {
        "soil_to_idx_keys": list(soil_to_idx.keys()),
        "sown_to_idx_keys": list(sown_to_idx.keys()),
        "le_name_classes": safe_list(le_name),
        "le_water_classes": safe_list(le_water)
    }
