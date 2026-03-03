import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier


BASE = Path(__file__).resolve().parent

KAPPA_RELEASE = os.getenv("KAPPA_RELEASE_TAG", "v1.1r_base46_cmp")
KAPPA_RELEASE_FALLBACK = "kappa_india_gulf_v2_prod_ready_v3"
MULTI_RELEASE = os.getenv("MULTI_RELEASE", "multi_species_cop_india_v5b_rich_relaxed_soft_hn")
MULTI_RELEASE_FALLBACK = "multi_species_cop_india_v2_prod"
KAPPA_MAX_DISTANCE_KM = float(os.getenv("KAPPA_MAX_DISTANCE_KM", "250"))

KAPPA_MASTER = BASE / "data" / "tabular" / "master_feature_matrix_kappa_india_gulf_v2_hardmerge4_augmented.csv"

INDIA_PHYSICS_NC = BASE / "data" / "netcdf" / "india_physics_2025w01.nc"
INDIA_WAVES_NC = BASE / "data" / "netcdf" / "india_waves_2025w01.nc"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_kappa():
    release_dir = BASE / "releases" / KAPPA_RELEASE
    if not release_dir.exists():
        release_dir = BASE / "releases" / KAPPA_RELEASE_FALLBACK
    rel = release_dir.name

    model_path = release_dir / "models" / f"xgboost_realtime_model_{rel}.json"
    bundle_path = release_dir / "models" / f"xgboost_realtime_ensemble_{rel}.pkl"
    cal_path = release_dir / "models" / f"xgboost_realtime_calibrator_{rel}.pkl"
    feat_path = release_dir / "models" / f"xgboost_realtime_features_{rel}.json"
    report_path = release_dir / "reports" / f"xgboost_realtime_report_{rel}.json"
    if not report_path.exists():
        candidates = sorted((release_dir / "reports").glob("xgboost_realtime_report_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No realtime report found in {release_dir / 'reports'}")
        report_path = candidates[0]

    if bundle_path.exists():
        models = joblib.load(bundle_path)
    else:
        m = XGBClassifier()
        m.load_model(model_path)
        models = [m]

    calibrator = joblib.load(cal_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(report["deployment_policy"]["recommended_threshold"])
    high_cutoff = float(report["deployment_policy"].get("high_confidence_cutoff", 0.80))
    medium_cutoff = float(report["deployment_policy"].get("medium_confidence_cutoff", 0.60))

    master_path = KAPPA_MASTER
    src = report.get("inference_feature_source")
    if src:
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = BASE / src_path
        if src_path.exists():
            master_path = src_path

    master = pd.read_csv(master_path)
    master = master.dropna(subset=["lon", "lat"] + features).reset_index(drop=True)
    return {
        "release": rel,
        "models": models,
        "calibrator": calibrator,
        "features": features,
        "threshold": threshold,
        "high_cutoff": high_cutoff,
        "medium_cutoff": medium_cutoff,
        "master": master,
    }


def _load_copernicus_grids():
    def _q(da, q: float):
        out = da.quantile(q, dim="time", skipna=True)
        if "quantile" in out.dims:
            out = out.squeeze("quantile", drop=True)
        return out

    def _grad_mag_2d(da):
        arr = np.asarray(da.values, dtype=np.float64)
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx * gx + gy * gy)
        return xr.DataArray(mag, coords=da.coords, dims=da.dims)

    phys = xr.open_dataset(INDIA_PHYSICS_NC)
    wav = xr.open_dataset(INDIA_WAVES_NC)
    try:
        so = phys["so"].mean(dim="depth", skipna=True) if "depth" in phys["so"].dims else phys["so"]
        uo = phys["uo"].mean(dim="depth", skipna=True) if "depth" in phys["uo"].dims else phys["uo"]
        vo = phys["vo"].mean(dim="depth", skipna=True) if "depth" in phys["vo"].dims else phys["vo"]
        current = np.sqrt(uo**2 + vo**2)
        wave = wav["VHM0"].interp(longitude=so["longitude"], latitude=so["latitude"], method="nearest")

        so_mean = so.mean(dim="time", skipna=True)
        so_std = so.std(dim="time", skipna=True)
        so_min = so.min(dim="time", skipna=True)
        so_max = so.max(dim="time", skipna=True)
        so_p10 = _q(so, 0.10)
        so_p90 = _q(so, 0.90)
        so_range = so_max - so_min
        so_cv = so_std / (np.abs(so_mean) + 1e-6)

        uo_mean = uo.mean(dim="time", skipna=True)
        vo_mean = vo.mean(dim="time", skipna=True)
        current_mean = current.mean(dim="time", skipna=True)
        current_std = current.std(dim="time", skipna=True)
        current_max = current.max(dim="time", skipna=True)
        current_p90 = _q(current, 0.90)
        current_cv = current_std / (np.abs(current_mean) + 1e-6)

        wave_mean = wave.mean(dim="time", skipna=True)
        wave_std = wave.std(dim="time", skipna=True)
        wave_min = wave.min(dim="time", skipna=True)
        wave_max = wave.max(dim="time", skipna=True)
        wave_p90 = _q(wave, 0.90)
        wave_p95 = _q(wave, 0.95)
        wave_range = wave_max - wave_min
        wave_cv = wave_std / (np.abs(wave_mean) + 1e-6)

        so_grad = _grad_mag_2d(so_mean)
        current_grad = _grad_mag_2d(current_mean)
        wave_grad = _grad_mag_2d(wave_mean)
        features = {
            "lon_min": float(so_mean["longitude"].min().values),
            "lon_max": float(so_mean["longitude"].max().values),
            "lat_min": float(so_mean["latitude"].min().values),
            "lat_max": float(so_mean["latitude"].max().values),
            "so_mean": so_mean,
            "so_std": so_std,
            "so_min": so_min,
            "so_max": so_max,
            "so_p10": so_p10,
            "so_p90": so_p90,
            "so_range": so_range,
            "so_cv": so_cv,
            "uo_mean": uo_mean,
            "vo_mean": vo_mean,
            "current_mean": current_mean,
            "current_std": current_std,
            "current_max": current_max,
            "current_p90": current_p90,
            "current_cv": current_cv,
            "wave_mean": wave_mean,
            "wave_std": wave_std,
            "wave_min": wave_min,
            "wave_max": wave_max,
            "wave_p90": wave_p90,
            "wave_p95": wave_p95,
            "wave_range": wave_range,
            "wave_cv": wave_cv,
            "so_grad": so_grad,
            "current_grad": current_grad,
            "wave_grad": wave_grad,
        }
        feature_keys = [k for k, v in features.items() if isinstance(v, xr.DataArray)]
        medians = {}
        for k in feature_keys:
            arr = np.asarray(features[k].values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            medians[k] = float(np.median(finite)) if finite.size else 0.0
        features["_feature_medians"] = medians
        return features
    finally:
        phys.close()
        wav.close()


def _load_species_model(species_id: str):
    multi_release = MULTI_RELEASE
    release_dir = BASE / "releases" / multi_release
    if not release_dir.exists():
        release_dir = BASE / "releases" / MULTI_RELEASE_FALLBACK
        multi_release = MULTI_RELEASE_FALLBACK

    models_dir = release_dir / "models"
    reports_dir = release_dir / "reports"

    model_pkl = models_dir / f"xgb_{species_id}_{multi_release}.pkl"
    cal_pkl = models_dir / f"calibrator_{species_id}_{multi_release}.pkl"
    feat_json = models_dir / f"features_{species_id}_{multi_release}.json"
    rep_json = reports_dir / f"report_{species_id}_{multi_release}.json"
    if not (model_pkl.exists() and cal_pkl.exists() and feat_json.exists() and rep_json.exists()):
        return None
    return {
        "release": multi_release,
        "model": joblib.load(model_pkl),
        "calibrator": joblib.load(cal_pkl),
        "features": json.loads(feat_json.read_text(encoding="utf-8")),
        "report": json.loads(rep_json.read_text(encoding="utf-8")),
    }


KAPPA = _load_kappa()
COP = _load_copernicus_grids()
OTHER_MODELS = {
    "gracilaria_spp": _load_species_model("gracilaria_spp"),
    "ulva_spp": _load_species_model("ulva_spp"),
    "sargassum_spp": _load_species_model("sargassum_spp"),
}
LOADED_MULTI_RELEASES = sorted(
    {m["release"] for m in OTHER_MODELS.values() if isinstance(m, dict) and "release" in m}
)
ACTIVE_MULTI_RELEASE = LOADED_MULTI_RELEASES[0] if LOADED_MULTI_RELEASES else "none"


def _priority(p_cal: float, high: float = 0.8, medium: float = 0.6) -> str:
    if p_cal >= high:
        return "high"
    if p_cal >= medium:
        return "medium"
    return "low"


def predict_kappa(lat: float, lon: float) -> dict:
    master = KAPPA["master"]
    dist = haversine_km(
        lat,
        lon,
        master["lat"].to_numpy(dtype=np.float64),
        master["lon"].to_numpy(dtype=np.float64),
    )
    idx = int(np.argmin(dist))
    row = master.iloc[idx]
    x = row[KAPPA["features"]].to_numpy(dtype=np.float32).reshape(1, -1)
    preds = [m.predict_proba(x)[:, 1][0] for m in KAPPA["models"]]
    p_raw = float(np.mean(preds))
    p_cal = float(KAPPA["calibrator"].predict(np.array([p_raw], dtype=np.float32))[0])
    return {
        "probabilityPercent": round(p_cal * 100.0, 2),
        "priority": _priority(p_cal, KAPPA["high_cutoff"], KAPPA["medium_cutoff"]),
        "predLabel": int(p_cal >= KAPPA["threshold"]),
        "nearestGrid": {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance_km": float(dist[idx]),
        },
    }


def extract_runtime_features(lat: float, lon: float) -> dict | None:
    if not (COP["lat_min"] <= lat <= COP["lat_max"] and COP["lon_min"] <= lon <= COP["lon_max"]):
        return None
    vals = {"lon": float(lon), "lat": float(lat)}
    feature_keys = [
        "so_mean", "so_std", "so_min", "so_max", "so_p10", "so_p90", "so_range", "so_cv",
        "uo_mean", "vo_mean", "current_mean", "current_std", "current_max", "current_p90", "current_cv",
        "wave_mean", "wave_std", "wave_min", "wave_max", "wave_p90", "wave_p95", "wave_range", "wave_cv",
        "so_grad", "current_grad", "wave_grad",
    ]
    for k in feature_keys:
        v = float(COP[k].sel(latitude=lat, longitude=lon, method="nearest").values)
        if not np.isfinite(v):
            v = float(COP.get("_feature_medians", {}).get(k, 0.0))
        vals[k] = v
    return vals


def predict_other(model_bundle: dict | None, feat_vals: dict | None) -> tuple[bool, float | None, str, str]:
    if model_bundle is None:
        return False, None, "pending", "model_not_trained"
    if feat_vals is None:
        return False, None, "unknown", "out_of_coverage"
    features = model_bundle["features"]
    if not all(f in feat_vals for f in features):
        return False, None, "unknown", "feature_unavailable"
    x = np.array([[feat_vals[f] for f in features]], dtype=np.float32)
    p_raw = float(model_bundle["model"].predict_proba(x)[:, 1][0])
    p_cal = float(model_bundle["calibrator"].predict(np.array([p_raw], dtype=np.float32))[0])
    thr = float(model_bundle["report"]["threshold"]["threshold"])
    return True, round(p_cal * 100.0, 2), _priority(p_cal), ("genus_proxy_positive" if p_cal >= thr else "genus_proxy_negative")


def predict_species(lat: float, lon: float) -> dict:
    warnings = []
    k = predict_kappa(lat, lon)
    kappa_in_coverage = bool(k.get("nearestGrid")) and float(k["nearestGrid"].get("distance_km", 1e9)) <= KAPPA_MAX_DISTANCE_KM
    if not kappa_in_coverage:
        warnings.append("kappaphycus_model_out_of_coverage")
    feat_vals = extract_runtime_features(lat, lon)
    if feat_vals is None:
        warnings.append("india_wide_proxy_models_out_of_coverage")

    g_ready, g_prob, g_priority, g_reason = predict_other(OTHER_MODELS["gracilaria_spp"], feat_vals)
    u_ready, u_prob, u_priority, u_reason = predict_other(OTHER_MODELS["ulva_spp"], feat_vals)
    s_ready, s_prob, s_priority, s_reason = predict_other(OTHER_MODELS["sargassum_spp"], feat_vals)

    species = [
        {
            "speciesId": "kappaphycus_alvarezii",
            "displayName": "Kappaphycus alvarezii",
            "ready": bool(kappa_in_coverage),
            "probabilityPercent": k["probabilityPercent"] if kappa_in_coverage else None,
            "priority": k["priority"] if kappa_in_coverage else "unknown",
            "reason": (
                "dedicated_production_model_positive"
                if kappa_in_coverage and int(k.get("predLabel", 0)) == 1
                else "dedicated_production_model_negative"
                if kappa_in_coverage
                else "out_of_coverage"
            ),
        },
        {
            "speciesId": "gracilaria_edulis",
            "displayName": "Gracilaria edulis",
            "ready": g_ready,
            "probabilityPercent": g_prob,
            "priority": g_priority,
            "reason": g_reason,
        },
        {
            "speciesId": "ulva_lactuca",
            "displayName": "Ulva lactuca",
            "ready": u_ready,
            "probabilityPercent": u_prob,
            "priority": u_priority,
            "reason": u_reason,
        },
        {
            "speciesId": "sargassum_wightii",
            "displayName": "Sargassum wightii",
            "ready": s_ready,
            "probabilityPercent": s_prob,
            "priority": s_priority,
            "reason": s_reason,
        },
    ]

    eligible = [
        s
        for s in species
        if s["ready"]
        and s["probabilityPercent"] is not None
        and str(s.get("reason", "")).endswith("_positive")
    ]
    best = sorted(eligible, key=lambda x: float(x["probabilityPercent"]), reverse=True)[0] if eligible else None
    if best is None:
        warnings.append("no_species_meets_suitability_threshold")

    loaded_other_releases = sorted(
        {m["release"] for m in OTHER_MODELS.values() if isinstance(m, dict) and "release" in m}
    )
    multi_release_name = loaded_other_releases[0] if loaded_other_releases else "none"

    return {
        "input": {"lat": lat, "lon": lon},
        "source": "species-orchestrator-production",
        "modelRelease": f"{KAPPA['release']}+{multi_release_name}",
        "nearestGrid": k["nearestGrid"],
        "species": species,
        "bestSpecies": best,
        "warnings": warnings,
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
                    "source": "species-orchestrator-production",
                    "kappaphycus_release": KAPPA["release"],
                    "multispecies_release": ACTIVE_MULTI_RELEASE,
                    "species_ids": [
                        "kappaphycus_alvarezii",
                        "gracilaria_edulis",
                        "ulva_lactuca",
                        "sargassum_wightii",
                    ],
                },
            )
            return
        if parsed.path in ("/predict/species", "/predict/kappaphycus"):
            q = parse_qs(parsed.query)
            try:
                lat = float(q.get("lat", [None])[0])
                lon = float(q.get("lon", [None])[0])
            except (TypeError, ValueError):
                self._send_json(400, {"error": "lat and lon query params are required numeric values."})
                return
            self._send_json(200, predict_species(lat, lon))
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/predict/species", "/predict/kappaphycus"):
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
        self._send_json(200, predict_species(lat, lon))


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving species API on http://127.0.0.1:8000", flush=True)
    server.serve_forever()
