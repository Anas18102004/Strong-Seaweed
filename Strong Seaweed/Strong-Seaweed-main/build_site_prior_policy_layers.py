import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from project_paths import TABULAR_DIR, REPORTS_DIR, ensure_dirs, with_legacy

MASTER_IN = with_legacy(TABULAR_DIR / "master_feature_matrix.csv", "master_feature_matrix.csv")
TRAIN_IN = with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv")
MASTER_OUT = TABULAR_DIR / "master_feature_matrix_augmented.csv"
TRAIN_OUT = TABULAR_DIR / "training_dataset_augmented.csv"
META_OUT = REPORTS_DIR / "site_prior_policy_metadata.json"
SEASONAL_OUT = TABULAR_DIR / "seasonal_operational_weights.csv"


# User-provided validated/known clusters.
CLUSTERS = [
    {"name": "Mandapam_MARS", "lat": 9.2886, "lon": 79.1329, "confidence": "high"},
    {"name": "Mandapam_Seashore", "lat": 9.279058, "lon": 79.015753, "confidence": "high"},
    {"name": "Vedalai", "lat": 9.26, "lon": 79.11, "confidence": "moderate"},
    {"name": "Rameswaram_Pamban", "lat": 9.288, "lon": 79.313, "confidence": "high"},
    {"name": "Sambai", "lat": 9.51993, "lon": 78.89981, "confidence": "moderate"},
    {"name": "Thondi_Puthukudi", "lat": 9.7417, "lon": 79.0177, "confidence": "high"},
]

# Core zone anchors provided in the prompt.
CORE_ZONE_POINTS = [
    {"name": "Vaan_Island", "lat": 8.83639, "lon": 78.21047},
    {"name": "Shingle_Island", "lat": 9.24174, "lon": 79.23563},
]

# Named sensitive islands explicitly mentioned; only Shingle had coordinates provided.
INVASION_SENSITIVE_POINTS = [
    {"name": "Shingle_Island", "lat": 9.24174, "lon": 79.23563},
]

CONFIDENCE_WEIGHT = {"high": 1.0, "moderate": 0.7}


def min_distance_km(
    grid_lon: np.ndarray, grid_lat: np.ndarray, points: list[dict], geod: Geod
) -> np.ndarray:
    if not points:
        return np.full(grid_lon.shape, np.nan, dtype=np.float64)

    out = np.full(grid_lon.shape, np.inf, dtype=np.float64)
    dst_lon = grid_lon.astype(np.float64)
    dst_lat = grid_lat.astype(np.float64)
    src_lon = np.empty_like(dst_lon)
    src_lat = np.empty_like(dst_lat)

    for p in points:
        src_lon.fill(float(p["lon"]))
        src_lat.fill(float(p["lat"]))
        _, _, dist_m = geod.inv(src_lon, src_lat, dst_lon, dst_lat)
        out = np.minimum(out, dist_m / 1000.0)

    return out


def weighted_cluster_prior_score(
    grid_lon: np.ndarray, grid_lat: np.ndarray, clusters: list[dict], geod: Geod
) -> np.ndarray:
    # Smooth prior: higher near validated clusters, weighted by source confidence.
    # 15 km decay is conservative for coastal farm neighborhood effect.
    score = np.zeros(grid_lon.shape, dtype=np.float64)
    dst_lon = grid_lon.astype(np.float64)
    dst_lat = grid_lat.astype(np.float64)
    src_lon = np.empty_like(dst_lon)
    src_lat = np.empty_like(dst_lat)

    for c in clusters:
        src_lon.fill(float(c["lon"]))
        src_lat.fill(float(c["lat"]))
        _, _, dist_m = geod.inv(src_lon, src_lat, dst_lon, dst_lat)
        dist_km = dist_m / 1000.0
        w = CONFIDENCE_WEIGHT.get(c["confidence"], 0.5)
        # Peak at cluster center, decays with distance.
        contribution = w * np.exp(-dist_km / 15.0)
        score = np.maximum(score, contribution)

    return score


def add_site_prior_and_policy(df: pd.DataFrame) -> pd.DataFrame:
    geod = Geod(ellps="WGS84")
    lon = df["lon"].to_numpy(dtype=np.float64)
    lat = df["lat"].to_numpy(dtype=np.float64)

    df = df.copy()
    df["dist_to_known_cluster_km"] = min_distance_km(lon, lat, CLUSTERS, geod)
    df["site_prior_score"] = weighted_cluster_prior_score(lon, lat, CLUSTERS, geod)

    # Policy/ecology constraints kept separate from core biophysical suitability.
    df["dist_to_core_zone_anchor_km"] = min_distance_km(lon, lat, CORE_ZONE_POINTS, geod)
    df["policy_core_zone_exclusion"] = (
        df["dist_to_core_zone_anchor_km"] <= 5.0
    ).astype(np.int8)

    df["dist_to_invasion_sensitive_km"] = min_distance_km(
        lon, lat, INVASION_SENSITIVE_POINTS, geod
    )
    df["policy_invasion_risk_flag"] = (
        df["dist_to_invasion_sensitive_km"] <= 10.0
    ).astype(np.int8)

    # GO 2005 should remain an explicit policy-review flag (not forced exclusion).
    # This keeps governance checks separate from ecological suitability prediction.
    df["policy_go2005_review_required"] = 1

    return df


def write_seasonal_weights() -> None:
    rows = []
    for m in range(1, 13):
        if 2 <= m <= 10:
            phase = "active_farming_window"
            weight = 1.0
        else:
            phase = "monsoon_caution_window"
            weight = 0.4
        rows.append({"month": m, "phase": phase, "operational_weight": weight})
    pd.DataFrame(rows).to_csv(SEASONAL_OUT, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add site-prior and policy layers to master/training datasets.")
    p.add_argument("--master_in", type=str, default=str(MASTER_IN))
    p.add_argument("--train_in", type=str, default=str(TRAIN_IN))
    p.add_argument("--master_out", type=str, default=str(MASTER_OUT))
    p.add_argument("--train_out", type=str, default=str(TRAIN_OUT))
    p.add_argument("--meta_out", type=str, default=str(META_OUT))
    p.add_argument("--seasonal_out", type=str, default=str(SEASONAL_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    master_in = Path(args.master_in)
    train_in = Path(args.train_in)
    master_out = Path(args.master_out)
    train_out = Path(args.train_out)
    meta_out = Path(args.meta_out)
    seasonal_out = Path(args.seasonal_out)

    if not master_in.exists() or not train_in.exists():
        raise FileNotFoundError(
            "Expected master_feature_matrix.csv and training_dataset.csv in workspace."
        )

    master = pd.read_csv(master_in)
    train = pd.read_csv(train_in)

    master_aug = add_site_prior_and_policy(master)
    train_aug = add_site_prior_and_policy(train)

    master_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    seasonal_out.parent.mkdir(parents=True, exist_ok=True)
    master_aug.to_csv(master_out, index=False)
    train_aug.to_csv(train_out, index=False)
    # Keep seasonal defaults but allow tagged output path.
    rows = []
    for m in range(1, 13):
        if 2 <= m <= 10:
            phase = "active_farming_window"
            weight = 1.0
        else:
            phase = "monsoon_caution_window"
            weight = 0.4
        rows.append({"month": m, "phase": phase, "operational_weight": weight})
    pd.DataFrame(rows).to_csv(seasonal_out, index=False)

    metadata = {
        "cluster_count": len(CLUSTERS),
        "core_zone_anchor_count": len(CORE_ZONE_POINTS),
        "invasion_sensitive_point_count": len(INVASION_SENSITIVE_POINTS),
        "confidence_weights": CONFIDENCE_WEIGHT,
        "core_zone_exclusion_buffer_km": 5.0,
        "invasion_risk_buffer_km": 10.0,
        "notes": [
            "Policy flags are separated from environmental suitability features.",
            "GO 2005 is encoded as review-required flag, not hard exclusion in this script.",
            "Only user-provided coordinates were used for core/invasion anchors.",
        ],
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {master_out}")
    print(f"Saved: {train_out}")
    print(f"Saved: {seasonal_out}")
    print(f"Saved: {meta_out}")
    print("Added columns:")
    print(
        "dist_to_known_cluster_km, site_prior_score, dist_to_core_zone_anchor_km, "
        "policy_core_zone_exclusion, dist_to_invasion_sensitive_km, "
        "policy_invasion_risk_flag, policy_go2005_review_required"
    )


if __name__ == "__main__":
    main()
