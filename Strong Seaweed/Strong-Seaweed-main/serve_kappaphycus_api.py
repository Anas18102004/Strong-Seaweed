import json
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


RELEASE_TAG = os.getenv("KAPPA_RELEASE_TAG", "v1.1r_base46_cmp")
BASE = Path(__file__).resolve().parent
RELEASE_DIR = BASE / "releases" / RELEASE_TAG
MODEL_PATH = RELEASE_DIR / "models" / f"xgboost_realtime_model_{RELEASE_TAG}.json"
MODEL_BUNDLE_PATH = RELEASE_DIR / "models" / f"xgboost_realtime_ensemble_{RELEASE_TAG}.pkl"
CALIBRATOR_PATH = RELEASE_DIR / "models" / f"xgboost_realtime_calibrator_{RELEASE_TAG}.pkl"
FEATURES_PATH = RELEASE_DIR / "models" / f"xgboost_realtime_features_{RELEASE_TAG}.json"
REPORT_PATH = RELEASE_DIR / "reports" / f"xgboost_realtime_report_{RELEASE_TAG}.json"
MASTER_PATH = BASE / "data" / "tabular" / "master_feature_matrix_v1_1_augmented.csv"


def _resolve_report_path() -> Path:
    if REPORT_PATH.exists():
        return REPORT_PATH
    candidates = sorted((RELEASE_DIR / "reports").glob("xgboost_realtime_report_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No report found in {RELEASE_DIR / 'reports'}")
    return candidates[0]


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_models():
    if MODEL_BUNDLE_PATH.exists():
        models = joblib.load(MODEL_BUNDLE_PATH)
    else:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        models = [model]

    calibrator = joblib.load(CALIBRATOR_PATH)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    report_path = _resolve_report_path()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(report["deployment_policy"]["recommended_threshold"])
    high_cutoff = float(report["deployment_policy"].get("high_confidence_cutoff", 0.80))
    medium_cutoff = float(report["deployment_policy"].get("medium_confidence_cutoff", 0.60))

    master_path = MASTER_PATH
    src = report.get("inference_feature_source")
    if src:
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = BASE / src_path
        if src_path.exists():
            master_path = src_path

    master = pd.read_csv(master_path)
    required = ["lon", "lat"] + features
    master = master.dropna(subset=required).reset_index(drop=True)

    return models, calibrator, features, threshold, high_cutoff, medium_cutoff, report, master


MODELS, CALIBRATOR, FEATURES, THRESHOLD, HIGH_CUTOFF, MEDIUM_CUTOFF, REPORT, MASTER = load_models()


def predict_point(lat: float, lon: float) -> dict:
    lat_arr = MASTER["lat"].to_numpy(dtype=np.float64)
    lon_arr = MASTER["lon"].to_numpy(dtype=np.float64)
    dist = haversine_km(lat, lon, lat_arr, lon_arr)
    idx = int(np.argmin(dist))
    nearest = MASTER.iloc[idx]

    x = nearest[FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
    member_preds = [m.predict_proba(x)[:, 1][0] for m in MODELS]
    p_raw = float(np.mean(member_preds))
    p_cal = float(CALIBRATOR.predict(np.array([p_raw], dtype=np.float32))[0])
    pred_label = int(p_cal >= THRESHOLD)
    if p_cal >= HIGH_CUTOFF:
        priority = "high"
    elif p_cal >= MEDIUM_CUTOFF:
        priority = "medium"
    else:
        priority = "low"

    return {
        "input": {"lat": lat, "lon": lon},
        "nearest_grid": {
            "lat": float(nearest["lat"]),
            "lon": float(nearest["lon"]),
            "distance_km": float(dist[idx]),
        },
        "kappaphycus": {
            "probability": p_cal,
            "probability_percent": round(p_cal * 100.0, 2),
            "raw_probability": p_raw,
            "pred_label": pred_label,
            "priority": priority,
        },
        "model": {
            "release_tag": RELEASE_TAG,
            "threshold": THRESHOLD,
        },
        "note": "Single-species production model currently available: Kappaphycus.",
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "release_tag": RELEASE_TAG,
                    "species": ["kappaphycus"],
                    "threshold": THRESHOLD,
                },
            )
            return

        if parsed.path == "/predict/kappaphycus":
            q = parse_qs(parsed.query)
            try:
                lat = float(q.get("lat", [None])[0])
                lon = float(q.get("lon", [None])[0])
            except (TypeError, ValueError):
                self._send_json(400, {"error": "lat and lon query params are required numeric values."})
                return
            self._send_json(200, predict_point(lat, lon))
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict/kappaphycus":
            self._send_json(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            lat = float(payload["lat"])
            lon = float(payload["lon"])
        except Exception:
            self._send_json(400, {"error": "JSON body with numeric lat and lon is required."})
            return
        self._send_json(200, predict_point(lat, lon))


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving kappaphycus model API on http://127.0.0.1:8000", flush=True)
    server.serve_forever()
