"""Analysis server for the SAIRA demo page.
Run from the project root:  uvicorn api.main:app --port 8000
The page (docs/saira_demo.html) posts a photo to /analyze and gets concern scores back."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import CKPT_PATH, HEADS, SKIN_TYPES
from dataset import eval_tf, load_image

app = FastAPI(title="SAIRA analysis server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # tighten for production

_model, _dev = None, "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def load_model():
    global _model
    if CKPT_PATH.exists():
        from model import SAIRANet
        _model = SAIRANet(pretrained=False).to(_dev)
        _model.load_state_dict(torch.load(CKPT_PATH, map_location=_dev))
        _model.eval()


@app.get("/health")
def health():
    return {"model_loaded": _model is not None}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Train first: python src/train.py")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPG, PNG or WebP image.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image is over 10 MB.")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.jpg").suffix) as tmp:
        tmp.write(data); tmp.flush()
        try:
            img = load_image(tmp.name)
        except ValueError:
            raise HTTPException(422, "Could not read that image.")
    x = eval_tf()(img).unsqueeze(0).to(_dev)
    pred = _model.predict(x)
    return {"scores": {h: float(pred[h][0]) for h in HEADS},
            "skin_type": SKIN_TYPES[int(pred["skin_type"][0].argmax())]}
