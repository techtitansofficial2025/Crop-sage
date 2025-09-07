# top of backend/main.py — MUST come before importing tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only; prevents TF from trying to init CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: reduce TF logs
import pickle, numpy as np, tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
    with open(os.path.join(base, "soil_encoder.pkl"), "rb") as f: soil_encoder = pickle.load(f)
    with open(os.path.join(base, "sown_encoder.pkl"), "rb") as f: sown_encoder = pickle.load(f)
    with open(os.path.join(base, "water_source_encoder.pkl"), "rb") as f: water_source_encoder = pickle.load(f)
    with open(os.path.join(base, "crop_name_encoder.pkl"), "rb") as f: crop_name_encoder = pickle.load(f)
    with open(os.path.join(base, "scaler.pkl"), "rb") as f: scaler = pickle.load(f)
    with open(os.path.join(base, "cd_scaler.pkl"), "rb") as f: cd_scaler = pickle.load(f)
    with open(os.path.join(base, "wr_scaler.pkl"), "rb") as f: wr_scaler = pickle.load(f)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
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
    crop_prob = preds[0][0]         # softmax vector
    ws_prob = preds[1][0]
    cd_scaled = preds[2].reshape(-1,1)
    wr_scaled = preds[3].reshape(-1,1)

    # indices -> labels
    crop_idx = int(np.argmax(crop_prob))
    ws_idx = int(np.argmax(ws_prob))
    crop_label = crop_name_encoder.inverse_transform([crop_idx])[0]
    ws_label = water_source_encoder.inverse_transform([ws_idx])[0]

    # inverse-scale regression outputs
    cd_orig = float(cd_scaler.inverse_transform(cd_scaled)[0,0])
    wr_orig = float(wr_scaler.inverse_transform(wr_scaled)[0,0])

    return {
        "crop": {"label": str(crop_label), "confidence": float(crop_prob[crop_idx])},
        "water_source": {"label": str(ws_label), "confidence": float(ws_prob[ws_idx])},
        "crop_duration": cd_orig,
        "water_required": wr_orig
    }

@app.get("/labels")
def labels():
    return {
        "crop_classes": list(crop_name_encoder.classes_),
        "water_source_classes": list(water_source_encoder.classes_)
    }
