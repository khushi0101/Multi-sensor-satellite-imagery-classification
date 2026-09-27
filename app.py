"""Offline tile classification service.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import hashlib
import io
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from classifier import SCRIPT_DIR, load_feature_extractor, preprocess


MODEL_PATH = SCRIPT_DIR / "models" / "classifier.joblib"
DB_PATH = SCRIPT_DIR / "results.db"
TILES_DIR = SCRIPT_DIR / "stored_tiles"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.9"))  # chosen from evaluate.py
MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_FORMATS = {"PNG"}

state = {}  # model objects, loaded once at startup


# ---------- Database ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_hash       TEXT NOT NULL,
                model_version   TEXT NOT NULL,
                filename        TEXT,
                tile_path       TEXT,
                predicted_label TEXT NOT NULL,
                confidence      REAL NOT NULL,
                probabilities   TEXT NOT NULL,   -- JSON: all class probabilities
                status          TEXT NOT NULL,   -- auto_accepted / needs_review
                threshold       REAL NOT NULL,
                latency_ms      REAL,
                created_at      TEXT NOT NULL,
                UNIQUE (tile_hash, model_version)
            )
        """)


def row_to_dict(row):
    d = dict(row)
    d["probabilities"] = json.loads(d["probabilities"])
    return d


# ---------- Startup: load model once ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    state["clf"] = bundle["classifier"]
    state["classes"] = bundle["classes"]
    state["model_version"] = bundle["model_version"]
    state["net"] = load_feature_extractor()
    TILES_DIR.mkdir(exist_ok=True)
    init_db()
    yield


app = FastAPI(title="Tile Classifier", lifespan=lifespan)


# ---------- Pipeline steps ----------
def validate(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "File too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode, catches corrupt files
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a valid image")
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format: {img.format}")
    return img.convert("RGB")


def predict(img: Image.Image) -> dict:
    with torch.no_grad():
        features = state["net"](preprocess(img).unsqueeze(0)).numpy()  # (1, 512)
    probs = state["clf"].predict_proba(features)[0]                   # (7,)
    classes = state["classes"]
    best = int(probs.argmax())
    return {
        "label": classes[best],
        "confidence": float(probs[best]),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


# ---------- Endpoints ----------
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    start = time.perf_counter()
    data = await file.read()

    # 1. Validate
    img = validate(data)

    # 2. Hash + dedupe (same tile + same model -> return stored result)
    tile_hash = hashlib.sha256(data).hexdigest()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM predictions WHERE tile_hash = ? AND model_version = ?",
            (tile_hash, state["model_version"]),
        ).fetchone()
    if existing:
        return {"duplicate": True, **row_to_dict(existing)}

    # 3-4. Preprocess + classify
    result = predict(img)

    # 5. Confidence policy
    status = "auto_accepted" if result["confidence"] >= CONFIDENCE_THRESHOLD else "needs_review"

    # 6. Store raw tile + result
    tile_path = TILES_DIR / f"{tile_hash}.png"
    if not tile_path.exists():
        img.save(tile_path)
    latency_ms = (time.perf_counter() - start) * 1000

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO predictions (tile_hash, model_version, filename, tile_path,
                   predicted_label, confidence, probabilities, status, threshold,
                   latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tile_hash, state["model_version"], file.filename, str(tile_path),
             result["label"], result["confidence"], json.dumps(result["probabilities"]),
             status, CONFIDENCE_THRESHOLD, latency_ms,
             datetime.now(timezone.utc).isoformat()),
        )
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (cur.lastrowid,)).fetchone()

    return {"duplicate": False, **row_to_dict(row)}


@app.get("/results")
def list_results(label: Optional[str] = None, status: Optional[str] = None,
                 min_conf: Optional[float] = None, max_conf: Optional[float] = None,
                 limit: int = 100):
    query, params = "SELECT * FROM predictions WHERE 1=1", []
    if label:
        query += " AND predicted_label = ?"; params.append(label)
    if status:
        query += " AND status = ?"; params.append(status)
    if min_conf is not None:
        query += " AND confidence >= ?"; params.append(min_conf)
    if max_conf is not None:
        query += " AND confidence <= ?"; params.append(max_conf)
    query += " ORDER BY created_at DESC LIMIT ?"; params.append(limit)
    with get_db() as conn:
        return [row_to_dict(r) for r in conn.execute(query, params).fetchall()]


@app.get("/results/{tile_hash}")
def get_result(tile_hash: str):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE tile_hash = ?", (tile_hash,)).fetchall()
    if not rows:
        raise HTTPException(404, "Tile not found")
    return [row_to_dict(r) for r in rows]


@app.get("/stats")
def stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_label = conn.execute(
            "SELECT predicted_label, COUNT(*) AS n, AVG(confidence) AS avg_conf "
            "FROM predictions GROUP BY predicted_label").fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status").fetchall()
    return {
        "total": total,
        "model_version": state["model_version"],
        "threshold": CONFIDENCE_THRESHOLD,
        "by_label": {r["predicted_label"]: {"count": r["n"], "avg_confidence": round(r["avg_conf"], 3)}
                     for r in by_label},
        "by_status": {r["status"]: r["n"] for r in by_status},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state.get("model_version")}