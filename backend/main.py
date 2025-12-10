# backend/main.py
# Robust Crop Sage backend - flexible artifact loader + adaptive prediction
# Paste into backend/main.py, commit & redeploy.

import os
# Must set before importing tensorflow to avoid GPU init attempts
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
from typing import List, Any, Optional, Dict, Union
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

# ---------- configuration ----------
ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

MODEL_CANDIDATES = [
    "model.keras",
    "crop_model.keras",
    "best_final.keras",
    "best_stage1.keras",
    "model",
]

SCALER_X_CANDIDATES = ["scaler_x.pkl", "scaler.pkl", "scaler_x", "scaler"]
DUR_SCALER_CANDIDATES = ["dur_scaler.pkl", "cd_scaler.pkl", "dur_scaler", "cd_scaler"]
WREQ_SCALER_CANDIDATES = ["wreq_scaler.pkl", "wr_scaler.pkl", "wreq_scaler", "wr_scaler"]

LE_NAME_CANDIDATES = ["le_name.pkl", "crop_name_encoder.pkl", "name_encoder.pkl", "le_name"]
LE_WATER_CANDIDATES = ["le_water.pkl", "water_source_encoder.pkl", "water_encoder.pkl", "le_water"]

SOIL_TO_IDX_CANDIDATES = ["soil_to_idx.pkl", "soil_encoder.pkl", "soil_encoder_map.pkl", "soil_to_idx"]
SOWN_TO_IDX_CANDIDATES = ["sown_to_idx.pkl", "sown_encoder.pkl", "sown_encoder_map.pkl", "sown_to_idx"]

# ---------- helper functions (defined before usage) ----------
def abs_paths(candidates: List[str]) -> List[str]:
    return [os.path.join(ART_DIR, p) for p in candidates]

def find_first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in abs_paths(candidates):
        if os.path.exists(p):
            return p
    return None

def load_pickle_any(candidates: List[str]) -> Any:
    """
    Try candidate filenames in ART_DIR; return first pickle-loaded object.
    Raises RuntimeError if none exist.
    """
    for p in abs_paths(candidates):
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise RuntimeError(f"Required artifact missing. Tried: {candidates} in {ART_DIR}")

def safe_list(obj) -> List[Any]:
    """
    Return list for encoder-like objects or lists. Useful for JSON responses.
    """
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

def to_mapping(obj: Any, name: str) -> Dict[str, int]:
    """
    Convert obj into a mapping token->index.
    Accepts dict, LabelEncoder-like (.classes_), list/ndarray.
    """
    if isinstance(obj, dict):
        return { str(k): int(v) for k, v in obj.items() }
    if hasattr(obj, "classes_"):
        try:
            classes = list(obj.classes_)
            return { str(c): int(i) for i, c in enumerate(classes) }
        except Exception:
            raise RuntimeError(f"Failed to convert LabelEncoder for {name} to mapping.")
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            classes = list(obj)
            return { str(c): int(i) for i, c in enumerate(classes) }
        except Exception:
            pass
    raise RuntimeError(f"Unsupported artifact type for {name}; expected dict, LabelEncoder, or list of classes.")

# ---------- FastAPI app and globals ----------
app = FastAPI(title="Crop Sage API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for development; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model: Optional[tf.keras.Model] = None
scaler_x = None
dur_scaler = None
wreq_scaler = None
le_name = None
le_water = None
soil_to_idx: Dict[str, int] = {}
sown_to_idx: Dict[str, int] = {}

# ---------- small SOWN parsing helpers ----------
def parse_sown_token(x: Union[str, int]) -> str:
    if isinstance(x, int):
        return str(int(x))
    s = str(x).strip().lower()
    month_map = {
        "jan":"1","january":"1","feb":"2","february":"2","mar":"3","march":"3","apr":"4","april":"4",
        "may":"5","jun":"6","june":"6","jul":"7","july":"7","aug":"8","august":"8","sep":"9","sept":"9","september":"9",
        "oct":"10","october":"10","nov":"11","november":"11","dec":"12","december":"12"
    }
    if s in month_map:
        return month_map[s]
    if s in ("kharif","khareef"):
        return "7"
    if s in ("rabi",):
        return "11"
    if s in ("zaid","zayad"):
        return "4"
    if s.isdigit():
        try:
            v = int(s)
            if 1 <= v <= 12:
                return str(v)
        except Exception:
            pass
    return s

# ---------- startup: load model + artifacts ----------
@app.on_event("startup")
def startup_load():
    global model, scaler_x, dur_scaler, wreq_scaler, le_name, le_water, soil_to_idx, sown_to_idx

    # load model (first candidate found)
    model_path = find_first_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        raise RuntimeError(f"Model file not found. Tried: {MODEL_CANDIDATES} in {ART_DIR}")
    model = tf.keras.models.load_model(model_path)

    # load pickled scalers and label encoders (try candidate filenames)
    scaler_x = load_pickle_any(SCALER_X_CANDIDATES)
    dur_scaler = load_pickle_any(DUR_SCALER_CANDIDATES)
    wreq_scaler = load_pickle_any(WREQ_SCALER_CANDIDATES)
    le_name = load_pickle_any(LE_NAME_CANDIDATES)
    le_water = load_pickle_any(LE_WATER_CANDIDATES)

    # soil/sown artifacts -> convert to mapping token->index
    raw_soil = load_pickle_any(SOIL_TO_IDX_CANDIDATES)
    soil_to_idx = to_mapping(raw_soil, "soil_to_idx")

    raw_sown = load_pickle_any(SOWN_TO_IDX_CANDIDATES)
    sown_to_idx = to_mapping(raw_sown, "sown_to_idx")

    # some helpful startup logging
    try:
        print("Startup: loaded model from:", model_path)
        print("Startup: scaler_x type:", type(scaler_x), "dur_scaler:", type(dur_scaler), "wreq_scaler:", type(wreq_scaler))
        print("Startup: le_name classes (sample):", safe_list(le_name)[:10], "count:", len(safe_list(le_name)))
        print("Startup: soil tokens (sample):", list(soil_to_idx.keys())[:10])
        print("Startup: sown tokens (sample):", list(sown_to_idx.keys())[:10])
    except Exception:
        pass

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
    def sown_valid(cls, v):
        if isinstance(v, int):
            if not (1 <= v <= 12):
                raise ValueError("SOWN integer must be 1..12")
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("SOWN string cannot be empty")
        else:
            raise ValueError("SOWN must be int 1..12 or non-empty string")
        return v

def ensure_loaded():
    if model is None or scaler_x is None or dur_scaler is None or wreq_scaler is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Check server logs.")

# ---------- mapping helpers ----------
def map_soil_token(soil_raw: str) -> int:
    s = str(soil_raw).strip()
    if s in soil_to_idx:
        return int(soil_to_idx[s])
    if s.lower() in soil_to_idx:
        return int(soil_to_idx[s.lower()])
    if s.title() in soil_to_idx:
        return int(soil_to_idx[s.title()])
    return int(len(soil_to_idx))

def map_sown_token(sown_raw: Union[int, str]) -> int:
    token = parse_sown_token(sown_raw)
    if token in sown_to_idx:
        return int(sown_to_idx[token])
    if token.lstrip("0") in sown_to_idx:
        return int(sown_to_idx[token.lstrip("0")])
    if token.lower() in sown_to_idx:
        return int(sown_to_idx[token.lower()])
    return int(len(sown_to_idx))

# ---------- helper to build single input vector ----------
def build_single_input_vector(soil_idx: int, sown_idx: int, numeric_scaled: np.ndarray) -> np.ndarray:
    # numeric_scaled: shape (1, M)
    soil_float = float(soil_idx)
    sown_float = float(sown_idx)
    flattened = numeric_scaled.flatten().astype(np.float32)
    vec = np.concatenate([[soil_float, sown_float], flattened], axis=None)
    return vec.reshape(1, -1).astype(np.float32)

# ---------- adaptive prediction function ----------
def predict_from_payload(payload: PredictRequest) -> Dict[str, Any]:
    ensure_loaded()

    soil_idx = map_soil_token(payload.SOIL)
    sown_idx = map_sown_token(payload.SOWN)

    numeric = np.array([[payload.SOIL_PH, payload.TEMP, payload.RELATIVE_HUMIDITY, payload.N, payload.P, payload.K]], dtype=np.float32)
    try:
        numeric_scaled = scaler_x.transform(numeric).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

    # Inspect model inputs
    try:
        model_inputs = getattr(model, "inputs", None)
        model_input_names = [inp.name.split(":")[0] for inp in model_inputs] if model_inputs is not None else []
        num_model_inputs = len(model_inputs) if model_inputs is not None else 0
    except Exception:
        model_input_names = []
        num_model_inputs = 0

    try:
        if num_model_inputs >= 3:
            # multi-input model: try to match names, else fallback to expected keys
            lowered = [n.lower() for n in model_input_names]
            key_map = {}
            for n in model_input_names:
                nl = n.lower()
                if "soil" in nl:
                    key_map["soil_in"] = n
                elif "sown" in nl or "season" in nl:
                    key_map["sown_in"] = n
                elif "num" in nl or "input" in nl or "numeric" in nl:
                    key_map["num_in"] = n

            # Assemble dict for prediction, prefer detected names
            pred_input = {
                key_map.get("soil_in", model_input_names[0] if model_input_names else "soil_in"): np.array([soil_idx], dtype=np.int32),
                key_map.get("sown_in", model_input_names[1] if len(model_input_names) > 1 else "sown_in"): np.array([sown_idx], dtype=np.int32),
                key_map.get("num_in", model_input_names[2] if len(model_input_names) > 2 else "num_in"): numeric_scaled
            }
            preds = model.predict(pred_input, verbose=0)
        else:
            # single-input model: build single vector and call predict accordingly
            single_vec = build_single_input_vector(soil_idx, sown_idx, numeric_scaled)
            if num_model_inputs == 1 and len(model_input_names) == 1:
                single_name = model_input_names[0]
                # try dict with name first, fallback to array
                try:
                    preds = model.predict({single_name: single_vec}, verbose=0)
                except Exception:
                    preds = model.predict(single_vec, verbose=0)
            else:
                preds = model.predict(single_vec, verbose=0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # parse outputs: [name_probs, water_probs, dur_scaled, wreq_scaled]
    try:
        name_probs = np.array(preds[0])[0]
        water_probs = np.array(preds[1])[0]
        dur_scaled = float(np.array(preds[2]).reshape(-1)[0])
        wreq_scaled = float(np.array(preds[3]).reshape(-1)[0])
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

# alias
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

# optional local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
