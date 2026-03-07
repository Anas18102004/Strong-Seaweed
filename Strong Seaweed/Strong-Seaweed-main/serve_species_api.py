import json
import os
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier


BASE = Path(__file__).resolve().parent

ACTIVE_RELEASE_FILE = BASE / "artifacts" / "reports" / "active_kappa_release.json"


def _resolve_kappa_release() -> str:
    env_tag = os.getenv("KAPPA_RELEASE_TAG", "").strip()
    if env_tag:
        return env_tag
    try:
        if ACTIVE_RELEASE_FILE.exists():
            data = json.loads(ACTIVE_RELEASE_FILE.read_text(encoding="utf-8"))
            tag = str(data.get("release_tag", "")).strip()
            if tag:
                return tag
    except Exception:
        pass
    return "v1.1r_base46_cmp"


KAPPA_RELEASE = _resolve_kappa_release()
KAPPA_RELEASE_FALLBACK = "kappa_india_gulf_v2_prod_ready_v3"
MULTI_RELEASE = os.getenv("MULTI_RELEASE", "multi_species_cop_india_v5b_rich_relaxed_soft_hn")
MULTI_RELEASE_FALLBACK = "multi_species_cop_india_v2_prod"
KAPPA_MAX_DISTANCE_KM = float(os.getenv("KAPPA_MAX_DISTANCE_KM", "250"))
KAPPA_THRESHOLD_OFFSET = float(os.getenv("KAPPA_THRESHOLD_OFFSET", "0.0"))
PROXY_THRESHOLD_OFFSET = float(os.getenv("PROXY_THRESHOLD_OFFSET", "0.0"))
FEATURE_STALENESS_DAYS = int(os.getenv("FEATURE_STALENESS_DAYS", "540"))
SCREENING_FALLBACK_FLOOR_PERCENT = float(os.getenv("SCREENING_FALLBACK_FLOOR_PERCENT", "10.0"))
RANKING_FALLBACK_FLOOR_PERCENT = float(os.getenv("RANKING_FALLBACK_FLOOR_PERCENT", "5.0"))
KAPPA_GEO_PRIOR_RADIUS_KM = float(os.getenv("KAPPA_GEO_PRIOR_RADIUS_KM", "120.0"))
KAPPA_GEO_PRIOR_PROB_FLOOR_PERCENT = float(os.getenv("KAPPA_GEO_PRIOR_PROB_FLOOR_PERCENT", "25.0"))
KAPPA_GEO_PRIOR_MAX_NEG_MARGIN_PERCENT = float(os.getenv("KAPPA_GEO_PRIOR_MAX_NEG_MARGIN_PERCENT", "80.0"))

# High-confidence cultivation anchors used only as a conservative screening prior.
KAPPA_CULTIVATION_ANCHORS = [
    {"name": "Mandapam", "lat": 9.28, "lon": 79.12},
    {"name": "Palk_Bay_Central", "lat": 9.35, "lon": 79.20},
    {"name": "Rameswaram", "lat": 9.29, "lon": 79.31},
]

KAPPA_MASTER = BASE / "data" / "tabular" / "master_feature_matrix_kappa_india_gulf_v2_hardmerge4_augmented.csv"

INDIA_PHYSICS_NC = BASE / "data" / "netcdf" / "india_physics_2025w01.nc"
INDIA_WAVES_NC = BASE / "data" / "netcdf" / "india_waves_2025w01.nc"


def _as_float(v):
    try:
        if v is None:
            return None
        out = float(v)
        if np.isfinite(out):
            return out
        return None
    except Exception:
        return None


def _normalize_form_input(form_input: dict | None) -> dict:
    raw = form_input if isinstance(form_input, dict) else {}
    overrides = raw.get("overrides") if isinstance(raw.get("overrides"), dict) else {}
    advanced = raw.get("advanced") if isinstance(raw.get("advanced"), dict) else {}
    return {
        "temperatureC": _as_float(overrides.get("temperatureC")),
        "salinityPpt": _as_float(overrides.get("salinityPpt")),
        "depthM": _as_float(raw.get("depthM")),
        "ph": _as_float(advanced.get("ph")),
        "turbidityNtu": _as_float(advanced.get("turbidityNtu")),
        "currentVelocityMs": _as_float(advanced.get("currentVelocityMs")),
        "waveHeightM": _as_float(advanced.get("waveHeightM")),
        "rainfallMm": _as_float(advanced.get("rainfallMm")),
        "tidalAmplitudeM": _as_float(advanced.get("tidalAmplitudeM")),
    }


def _apply_overrides_to_vector(feature_names: list[str], user_inputs: dict, target_values: dict, source_label: str) -> dict:
    feature_set = set(feature_names)
    applied = []
    ignored = []

    # Candidate mappings from user inputs to model feature names.
    mappings = [
        ("salinityPpt", ["so_mean", "sal_mean", "salinity", "salinity_ppt"]),
        ("currentVelocityMs", ["current_mean", "current_velocity", "current_velocity_ms"]),
        ("waveHeightM", ["wave_mean", "wave_height", "wave_height_m", "VHM0"]),
        ("temperatureC", ["thetao_mean", "sst_mean", "temperature", "temperature_c", "temp_mean", "temp_c"]),
        ("depthM", ["depth", "depth_m", "bathymetry", "bathymetry_m"]),
        ("ph", ["ph"]),  # currently no trained model feature
        ("turbidityNtu", ["turb_mean", "turbidity", "turbidity_ntu"]),
        ("rainfallMm", ["rain_mean", "rainfall", "rainfall_mm"]),
        ("tidalAmplitudeM", ["tidal_amplitude", "tidal_amplitude_m", "tide_range_m"]),
    ]

    for input_key, candidates in mappings:
        v = user_inputs.get(input_key)
        if v is None:
            continue
        target = next((c for c in candidates if c in feature_set), None)
        if target is None:
            ignored.append({"input": input_key, "value": v, "reason": "no_matching_feature"})
            continue
        target_values[target] = float(v)
        applied.append({"input": input_key, "feature": target, "value": float(v), "source": source_label})

    return {"applied": applied, "ignored": ignored}


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _distance_km_point(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(haversine_km(lat1, lon1, np.array([lat2]), np.array([lon2]))[0])


def _nearest_anchor(lat: float, lon: float) -> tuple[str | None, float]:
    best_name = None
    best_dist = 1e9
    for a in KAPPA_CULTIVATION_ANCHORS:
        d = _distance_km_point(lat, lon, float(a["lat"]), float(a["lon"]))
        if d < best_dist:
            best_dist = d
            best_name = str(a["name"])
    return best_name, float(best_dist)


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
        # Some grid cells can be all-NaN in source files; keep logs clean and use median fallback later.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
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
        time_max = None
        if "time" in phys:
            try:
                time_max = pd.to_datetime(phys["time"].max().values).isoformat()
            except Exception:
                time_max = None
        features = {
            "lon_min": float(so_mean["longitude"].min().values),
            "lon_max": float(so_mean["longitude"].max().values),
            "lat_min": float(so_mean["latitude"].min().values),
            "lat_max": float(so_mean["latitude"].max().values),
            "feature_timestamp": time_max,
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


def _confidence_band(probability_percent: float | None) -> str:
    if probability_percent is None:
        return "unknown"
    p = float(probability_percent)
    if p >= 80.0:
        return "high"
    if p >= 60.0:
        return "medium"
    return "low"


def _species_actionability(ready: bool, reason: str, confidence_band: str) -> str:
    if not ready:
        return "insufficient_data"
    if reason.endswith("_positive") and confidence_band in ("high", "medium"):
        return "recommended"
    if reason.endswith("_positive"):
        return "test_pilot_only"
    return "not_recommended"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _feature_is_stale(ts_iso: str | None, max_age_days: int) -> bool:
    if not ts_iso:
        return False
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).days
        return age_days > max_age_days
    except Exception:
        return False


def predict_kappa(lat: float, lon: float, user_inputs: dict | None = None) -> dict:
    master = KAPPA["master"]
    dist = haversine_km(
        lat,
        lon,
        master["lat"].to_numpy(dtype=np.float64),
        master["lon"].to_numpy(dtype=np.float64),
    )
    idx = int(np.argmin(dist))
    row = master.iloc[idx]
    row_vals = {f: float(row[f]) for f in KAPPA["features"]}
    kappa_override_diag = {"applied": [], "ignored": []}
    if isinstance(user_inputs, dict):
        kappa_override_diag = _apply_overrides_to_vector(KAPPA["features"], user_inputs, row_vals, "user_override")

    x = np.array([[row_vals[f] for f in KAPPA["features"]]], dtype=np.float32)
    preds = [m.predict_proba(x)[:, 1][0] for m in KAPPA["models"]]
    p_raw = float(np.mean(preds))
    p_cal = float(KAPPA["calibrator"].predict(np.array([p_raw], dtype=np.float32))[0])
    base_thr = float(KAPPA["threshold"])
    effective_thr = _clamp01(base_thr + KAPPA_THRESHOLD_OFFSET)
    return {
        "probabilityPercent": round(p_cal * 100.0, 2),
        "priority": _priority(p_cal, KAPPA["high_cutoff"], KAPPA["medium_cutoff"]),
        "predLabel": int(p_cal >= effective_thr),
        "baseThreshold": base_thr,
        "effectiveThreshold": effective_thr,
        "nearestGrid": {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance_km": float(dist[idx]),
        },
        "overrideDiagnostics": kappa_override_diag,
    }


def extract_runtime_features(lat: float, lon: float, user_inputs: dict | None = None) -> tuple[dict | None, dict]:
    diagnostics = {"applied": [], "ignored": []}
    if not (COP["lat_min"] <= lat <= COP["lat_max"] and COP["lon_min"] <= lon <= COP["lon_max"]):
        return None, diagnostics
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
    if isinstance(user_inputs, dict):
        diagnostics = _apply_overrides_to_vector(list(vals.keys()), user_inputs, vals, "user_override")
    return vals, diagnostics


def predict_other(model_bundle: dict | None, feat_vals: dict | None) -> dict:
    if model_bundle is None:
        return {
            "ready": False,
            "probabilityPercent": None,
            "priority": "pending",
            "reason": "model_not_trained",
            "thresholdPercent": None,
            "marginToThresholdPercent": None,
        }
    if feat_vals is None:
        return {
            "ready": False,
            "probabilityPercent": None,
            "priority": "unknown",
            "reason": "out_of_coverage",
            "thresholdPercent": None,
            "marginToThresholdPercent": None,
        }
    features = model_bundle["features"]
    if not all(f in feat_vals for f in features):
        return {
            "ready": False,
            "probabilityPercent": None,
            "priority": "unknown",
            "reason": "feature_unavailable",
            "thresholdPercent": None,
            "marginToThresholdPercent": None,
        }
    x = np.array([[feat_vals[f] for f in features]], dtype=np.float32)
    p_raw = float(model_bundle["model"].predict_proba(x)[:, 1][0])
    p_cal = float(model_bundle["calibrator"].predict(np.array([p_raw], dtype=np.float32))[0])
    base_thr = float(model_bundle["report"]["threshold"]["threshold"])
    thr = _clamp01(base_thr + PROXY_THRESHOLD_OFFSET)
    p_pct = round(p_cal * 100.0, 2)
    thr_pct = round(thr * 100.0, 2)
    margin_pct = round(p_pct - thr_pct, 2)
    return {
        "ready": True,
        "probabilityPercent": p_pct,
        "priority": _priority(p_cal),
        "reason": "genus_proxy_positive" if p_cal >= thr else "genus_proxy_negative",
        "baseThresholdPercent": round(base_thr * 100.0, 2),
        "thresholdPercent": thr_pct,
        "marginToThresholdPercent": margin_pct,
    }


def predict_species(lat: float, lon: float, form_input: dict | None = None) -> dict:
    user_inputs = _normalize_form_input(form_input)
    warnings = []
    k = predict_kappa(lat, lon, user_inputs)
    kappa_in_coverage = bool(k.get("nearestGrid")) and float(k["nearestGrid"].get("distance_km", 1e9)) <= KAPPA_MAX_DISTANCE_KM
    if not kappa_in_coverage:
        warnings.append("kappaphycus_model_out_of_coverage")
    feat_vals, proxy_override_diag = extract_runtime_features(lat, lon, user_inputs)
    if feat_vals is None:
        warnings.append("india_wide_proxy_models_out_of_coverage")

    g = predict_other(OTHER_MODELS["gracilaria_spp"], feat_vals)
    u = predict_other(OTHER_MODELS["ulva_spp"], feat_vals)
    s = predict_other(OTHER_MODELS["sargassum_spp"], feat_vals)

    kappa_prob = k["probabilityPercent"] if kappa_in_coverage else None
    kappa_thr_pct = round(float(k.get("effectiveThreshold", 0.0)) * 100.0, 2) if kappa_in_coverage else None
    kappa_base_thr_pct = round(float(k.get("baseThreshold", 0.0)) * 100.0, 2) if kappa_in_coverage else None
    kappa_margin = round(float(kappa_prob) - float(kappa_thr_pct), 2) if kappa_in_coverage else None
    kappa_reason = (
        "dedicated_production_model_positive"
        if kappa_in_coverage and int(k.get("predLabel", 0)) == 1
        else "dedicated_production_model_negative"
        if kappa_in_coverage
        else "out_of_coverage"
    )
    kappa_confidence = _confidence_band(kappa_prob)

    species = [
        {
            "speciesId": "kappaphycus_alvarezii",
            "displayName": "Kappaphycus alvarezii",
            "ready": bool(kappa_in_coverage),
            "probabilityPercent": kappa_prob,
            "priority": k["priority"] if kappa_in_coverage else "unknown",
            "reason": kappa_reason,
            "baseThresholdPercent": kappa_base_thr_pct,
            "thresholdPercent": kappa_thr_pct,
            "marginToThresholdPercent": kappa_margin,
            "confidenceBand": kappa_confidence,
            "actionability": _species_actionability(bool(kappa_in_coverage), kappa_reason, kappa_confidence),
        },
        {
            "speciesId": "gracilaria_edulis",
            "displayName": "Gracilaria edulis",
            "ready": g["ready"],
            "probabilityPercent": g["probabilityPercent"],
            "priority": g["priority"],
            "reason": g["reason"],
            "baseThresholdPercent": g.get("baseThresholdPercent"),
            "thresholdPercent": g["thresholdPercent"],
            "marginToThresholdPercent": g["marginToThresholdPercent"],
            "confidenceBand": _confidence_band(g["probabilityPercent"]),
            "actionability": _species_actionability(
                g["ready"], g["reason"], _confidence_band(g["probabilityPercent"])
            ),
        },
        {
            "speciesId": "ulva_lactuca",
            "displayName": "Ulva lactuca",
            "ready": u["ready"],
            "probabilityPercent": u["probabilityPercent"],
            "priority": u["priority"],
            "reason": u["reason"],
            "baseThresholdPercent": u.get("baseThresholdPercent"),
            "thresholdPercent": u["thresholdPercent"],
            "marginToThresholdPercent": u["marginToThresholdPercent"],
            "confidenceBand": _confidence_band(u["probabilityPercent"]),
            "actionability": _species_actionability(
                u["ready"], u["reason"], _confidence_band(u["probabilityPercent"])
            ),
        },
        {
            "speciesId": "sargassum_wightii",
            "displayName": "Sargassum wightii",
            "ready": s["ready"],
            "probabilityPercent": s["probabilityPercent"],
            "priority": s["priority"],
            "reason": s["reason"],
            "baseThresholdPercent": s.get("baseThresholdPercent"),
            "thresholdPercent": s["thresholdPercent"],
            "marginToThresholdPercent": s["marginToThresholdPercent"],
            "confidenceBand": _confidence_band(s["probabilityPercent"]),
            "actionability": _species_actionability(
                s["ready"], s["reason"], _confidence_band(s["probabilityPercent"])
            ),
        },
    ]

    ready_scored = [
        s
        for s in species
        if s["ready"] and s["probabilityPercent"] is not None
    ]
    top_candidates = sorted(ready_scored, key=lambda x: float(x["probabilityPercent"]), reverse=True)

    eligible = [
        s
        for s in species
        if s["ready"]
        and s["probabilityPercent"] is not None
        and str(s.get("reason", "")).endswith("_positive")
    ]
    best = sorted(eligible, key=lambda x: float(x["probabilityPercent"]), reverse=True)[0] if eligible else None
    decision_source = "model_threshold"
    if best is None:
        # Screening fallback: if nothing crosses strict threshold, still return top candidate when probability is reasonable.
        if ready_scored:
            top = sorted(ready_scored, key=lambda x: float(x["probabilityPercent"]), reverse=True)[0]
            if float(top["probabilityPercent"]) >= SCREENING_FALLBACK_FLOOR_PERCENT:
                top = dict(top)
                top["reason"] = "screening_fallback_top_ranked"
                top["actionability"] = "test_pilot_only"
                best = top
                decision_source = "screening_fallback"
                warnings.append("no_species_meets_threshold_using_screening_fallback")
            elif float(top["probabilityPercent"]) >= RANKING_FALLBACK_FLOOR_PERCENT:
                top = dict(top)
                top["reason"] = "ranking_fallback_low_confidence"
                top["actionability"] = "not_recommended"
                best = top
                decision_source = "ranking_fallback"
                warnings.append("no_species_meets_threshold_using_ranking_fallback")
                warnings.append("best_species_low_confidence")
            else:
                top = dict(top)
                top["reason"] = "ranking_fallback_ultra_low_confidence"
                top["actionability"] = "not_recommended"
                best = top
                decision_source = "ranking_fallback"
                warnings.append("no_species_meets_suitability_threshold")
                warnings.append("best_species_ultra_low_confidence")
        else:
            if species:
                fallback = dict(species[0])
                fallback["reason"] = "insufficient_data_no_model_coverage"
                fallback["actionability"] = "insufficient_data"
                best = fallback
                decision_source = "insufficient_data_fallback"
                warnings.append("no_species_meets_suitability_threshold")
                warnings.append("no_species_with_ready_scores")
            else:
                warnings.append("no_species_meets_suitability_threshold")

    # Conservative geo-prior rescue for known Kappaphycus belts.
    # This avoids obvious false negatives when dedicated model probability is moderate
    # but strict thresholding plus proxy overconfidence dominates the final pick.
    if kappa_in_coverage and kappa_prob is not None:
        anchor_name, anchor_distance_km = _nearest_anchor(lat, lon)
        kappa_entry = next((sp for sp in species if sp.get("speciesId") == "kappaphycus_alvarezii"), None)
        if kappa_entry is not None:
            kappa_margin_pct = float(kappa_entry.get("marginToThresholdPercent") or 0.0)
            eligible_geo_prior = (
                anchor_distance_km <= KAPPA_GEO_PRIOR_RADIUS_KM
                and float(kappa_prob) >= KAPPA_GEO_PRIOR_PROB_FLOOR_PERCENT
                and kappa_margin_pct >= (-1.0 * KAPPA_GEO_PRIOR_MAX_NEG_MARGIN_PERCENT)
            )
            if eligible_geo_prior and (best is None or best.get("speciesId") != "kappaphycus_alvarezii"):
                adjusted = dict(kappa_entry)
                adjusted["reason"] = "geo_prior_kappaphycus_screening"
                adjusted["actionability"] = "test_pilot_only"
                best = adjusted
                decision_source = "geo_prior"
                warnings.append("geo_prior_adjustment_applied")
                warnings.append(f"geo_prior_anchor={anchor_name}")

    loaded_other_releases = sorted(
        {m["release"] for m in OTHER_MODELS.values() if isinstance(m, dict) and "release" in m}
    )
    multi_release_name = loaded_other_releases[0] if loaded_other_releases else "none"
    provided_override_count = sum(1 for _, v in user_inputs.items() if v is not None)
    applied_count = len(proxy_override_diag.get("applied", [])) + len(k.get("overrideDiagnostics", {}).get("applied", []))
    if provided_override_count > 0 and applied_count == 0:
        warnings.append("user_overrides_provided_but_not_mapped_to_model_features")

    all_applied = k.get("overrideDiagnostics", {}).get("applied", []) + proxy_override_diag.get("applied", [])
    unique_applied_inputs = {item.get("input") for item in all_applied if item.get("input")}
    total_applied = len(all_applied)
    override_coverage = (len(unique_applied_inputs) / provided_override_count) if provided_override_count > 0 else 1.0
    if provided_override_count > 0 and override_coverage < 0.5:
        warnings.append("low_override_mapping_coverage")
    if _feature_is_stale(COP.get("feature_timestamp"), FEATURE_STALENESS_DAYS):
        warnings.append("environmental_features_stale")

    overall_actionability = "insufficient_data"
    if best is not None:
        overall_actionability = best.get("actionability", "test_pilot_only")
        if float(best.get("marginToThresholdPercent", 0.0) or 0.0) < 5.0:
            warnings.append("best_species_near_decision_boundary")

    return {
        "input": {"lat": lat, "lon": lon},
        "inputMode": "lat_lon_plus_overrides" if provided_override_count > 0 else "lat_lon_only",
        "source": "species-orchestrator-production",
        "decisionPolicyVersion": "v3_threshold_offsets_geo_prior",
        "modelRelease": f"{KAPPA['release']}+{multi_release_name}",
        "featureTimestamp": COP.get("feature_timestamp"),
        "nearestGrid": k["nearestGrid"],
        "species": species,
        "topCandidatesByProbability": top_candidates,
        "bestSpecies": best,
        "decisionSource": decision_source if best is not None else "none",
        "actionability": overall_actionability,
        "dataQuality": {
            "overrideCountProvided": provided_override_count,
            "overrideCountApplied": total_applied,
            "overrideCoverageScore": round(float(override_coverage), 3),
            "kappaphycusCoverage": bool(kappa_in_coverage),
            "proxyCoverage": bool(feat_vals is not None),
        },
        "appliedOverrides": {
            "kappaphycus": k.get("overrideDiagnostics", {}),
            "proxyModels": proxy_override_diag,
        },
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
            self._send_json(200, predict_species(lat, lon, None))
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
            form_input = payload.get("formInput") if isinstance(payload.get("formInput"), dict) else None
        except Exception:
            self._send_json(400, {"error": "JSON body with numeric lat and lon is required."})
            return
        self._send_json(200, predict_species(lat, lon, form_input))


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving species API on http://127.0.0.1:8000", flush=True)
    server.serve_forever()
