# main.py
# Matches your training artifacts from fixed_robust_crop_model.py
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

# App
app = FastAPI(title="Crop Sage API - model-compatible")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Artifacts directory and filenames (match your training saver)
ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_FILE = os.path.join(ART_DIR, "model.keras")
SCALER_X_FILE = os.path.join(ART_DIR, "scaler_x.pkl")
DUR_SCALER_FILE = os.path.join(ART_DIR, "dur_scaler.pkl")
WREQ_SCALER_FILE = os.path.join(ART_DIR, "wreq_scaler.pkl")
LE_NAME_FILE = os.path.join(ART_DIR, "le_name.pkl")
LE_WATER_FILE = os.path.join(ART_DIR, "le_water.pkl")
SOIL_TO_IDX_FILE = os.path.join(ART_DIR, "soil_to_idx.pkl")
SOWN_TO_IDX_FILE = os.path.join(ART_DIR, "sown_to_idx.pkl")

# Globals
model: Optional[tf.keras.Model] = None
scaler_x = None
dur_scaler = None
wreq_scaler = None
le_name = None
le_water = None
soil_to_idx: Dict[str, int] = {}
sown_to_idx: Dict[str, int] = {}

# ---------- Utilities ----------
def load_pickle(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Required artifact missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_list(obj) -> List[str]:
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
    """
    Convert a month name or abbreviation to token string "1".."12".
    Returns None if not a month name.
    """
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    month_map = {
        "jan":"1","january":"1","feb":"2","february":"2","mar":"3","march":"3","apr":"4","april":"4",
        "may":"5","jun":"6","june":"6","jul":"7","july":"7","aug":"8","august":"8","sep":"9","sept":"9","september":"9",
        "oct":"10","october":"10","nov":"11","november":"11","dec":"12","december":"12"
    }
    return month_map.get(s)

def season_to_representative_month_token(season: str) -> Optional[str]:
    """
    If frontend sends season words like 'kharif','rabi','zaid', map to representative month token.
    These are heuristics — change if you prefer other representative months.
    """
    if not isinstance(season, str):
        return None
    s = season.strip().lower()
    if s in ("kharif", "khareef", "kharif "):
        return "7"   # July (monsoon)
    if s in ("rabi",):
        return "11"  # November (typical rabi sowing month)
    if s in ("zaid", "zayed", "zaid "):
        return "4"   # April
    return None

def ensure_loaded():
    if model is None:
        raise HTTPException(status_code=503, detail="Server artifacts not loaded; restart and check logs.")

# ---------- Startup: load artifacts ----------
@app.on_event("startup")
def startup_load():
    global model, scaler_x, dur_scaler, wreq_scaler, le_name, le_water, soil_to_idx, sown_to_idx
    # model
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"Model file not found: {MODEL_FILE}")
    model = tf.keras.models.load_model(MODEL_FILE)

    # pickles
    scaler_x = load_pickle(SCALER_X_FILE)
    dur_scaler = load_pickle(DUR_SCALER_FILE)
    wreq_scaler = load_pickle(WREQ_SCALER_FILE)
    le_name = load_pickle(LE_NAME_FILE)
    le_water = load_pickle(LE_WATER_FILE)
    soil_to_idx = load_pickle(SOIL_TO_IDX_FILE)
    sown_to_idx = load_pickle(SOWN_TO_IDX_FILE)

# ---------- Request schema ----------
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
    def sown_must_be_month_or_string(cls, v):
        if isinstance(v, int):
            if not (1 <= v <= 12):
                raise ValueError("SOWN month must be 1..12")
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("SOWN must be non-empty string or month number")
        else:
            raise ValueError("SOWN must be integer month or string")
        return v

# ---------- Helper to map inputs to model indices ----------
def map_soil_token(soil_raw: str) -> int:
    s = str(soil_raw).strip()
    # try direct match
    if s in soil_to_idx:
        return int(soil_to_idx[s])
    # try lowercase variant
    s_low = s.lower()
    if s_low in soil_to_idx:
        return int(soil_to_idx[s_low])
    # try title-case
    s_title = s.title()
    if s_title in soil_to_idx:
        return int(soil_to_idx[s_title])
    # fallback: unknown index (training used len(soil_to_idx) as unknown)
    return int(len(soil_to_idx))

def map_sown_token(sown_raw: Union[int, str]) -> int:
    """
    Convert frontend SOWN -> model sown_idx int.
    Accepts: integer 1..12 -> "1".."12" tokens, month names, numeric strings,
             or season names (kharif/rabi/zaid) -> mapped to representative month token.
    If token not present in sown_to_idx, returns len(sown_to_idx) as unknown index (matches training).
    """
    # if int -> token string
    if isinstance(sown_raw, int):
        token = str(int(sown_raw))
    else:
        s = str(sown_raw).strip()
        # numeric string?
        if s.isdigit():
            token = str(int(s))
        else:
            # month name -> number token
            month_token = month_name_to_number_token(s)
            if month_token is not None:
                token = month_token
            else:
                # season heuristic mapping (kharif/rabi/zaid)
                season_token = season_to_representative_month_token(s)
                if season_token is not None:
                    token = season_token
                else:
                    # normalization attempts
                    s_low = s.lower()
                    s_title = s.title()
                    if s in sown_to_idx:
                        token = s
                    elif s_low in sown_to_idx:
                        token = s_low
                    elif s_title in sown_to_idx:
                        token = s_title
                    else:
                        # fallback to original string token (maybe training used those strings)
                        token = s
    # map token to index
    if token in sown_to_idx:
        return int(sown_to_idx[token])
    # try normalized numeric token forms (strip leading zeros)
    try:
        if token.lstrip("0").isdigit():
            t = str(int(token))
            if t in sown_to_idx:
                return int(sown_to_idx[t])
    except Exception:
        pass
    # unknown fallback index (matches training code map_or_unknown behavior)
    return int(len(sown_to_idx))

# ---------- Prediction logic ----------
def predict_from_payload(payload: PredictRequest) -> Dict[str, Any]:
    ensure_loaded()
    # soil index
    soil_token = str(payload.SOIL).strip()
    soil_idx = map_soil_token(soil_token)

    # sown index
    sown_idx = map_sown_token(payload.SOWN)

    # numeric features and scale (order MUST match training NUM_INPUTS: ["SOIL_PH","TEMP","RELATIVE_HUMIDITY","N","P","K"])
    numeric = np.array([[payload.SOIL_PH, payload.TEMP, payload.RELATIVE_HUMIDITY, payload.N, payload.P, payload.K]], dtype=np.float32)
    try:
        numeric_scaled = scaler_x.transform(numeric).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Numeric scaler transform failed: {e}")

    # build arrays consistent with training shapes
    soil_in = np.array([soil_idx], dtype=np.int32)
    sown_in = np.array([sown_idx], dtype=np.int32)
    num_in = numeric_scaled  # already shape (1, n_features)

    # model predict. training model outputs: [name_out, water_out, dur_out, wreq_out]
    try:
        preds = model.predict({"soil_in": soil_in, "sown_in": sown_in, "num_in": num_in}, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # parse outputs safely
    try:
        name_probs = np.array(preds[0])[0]      # shape (n_name_classes,)
        water_probs = np.array(preds[1])[0]     # shape (n_water_classes,)
        dur_scaled = np.array(preds[2]).reshape(-1)[0]   # single value
        wreq_scaled = np.array(preds[3]).reshape(-1)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {e}")

    # pick indices
    name_idx = int(np.argmax(name_probs))
    water_idx = int(np.argmax(water_probs))
    name_conf = float(name_probs[name_idx])
    water_conf = float(water_probs[water_idx])

    # decode labels via LabelEncoders saved during training (le_name, le_water)
    try:
        crop_label = str(le_name.inverse_transform([name_idx])[0])
    except Exception:
        # fallback to class list
        crop_label = str(safe_list(le_name)[name_idx] if name_idx < len(safe_list(le_name)) else name_idx)

    try:
        water_label = str(le_water.inverse_transform([water_idx])[0])
    except Exception:
        water_label = str(safe_list(le_water)[water_idx] if water_idx < len(safe_list(le_water)) else water_idx)

    # inverse-scale regression outputs
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

# ---------- Endpoints ----------
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
    # expose keys (useful to populate frontend select; they are the original training tokens)
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
