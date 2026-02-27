import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from project_paths import (
    TABULAR_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine boundary positive candidates with spatial and feature diversity caps."
    )
    p.add_argument(
        "--grid_csv",
        type=Path,
        default=TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv",
    )
    p.add_argument(
        "--training_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_optimized_plus_hn.csv",
    )
    p.add_argument(
        "--model_bundle",
        type=Path,
        default=Path("releases/v1_1/models/xgboost_realtime_ensemble_v1_1.pkl"),
    )
    p.add_argument(
        "--calibrator",
        type=Path,
        default=Path("releases/v1_1/models/xgboost_realtime_calibrator_v1_1.pkl"),
    )
    p.add_argument(
        "--feature_json",
        type=Path,
        default=Path("releases/v1_1/models/xgboost_realtime_features_v1_1.json"),
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=EXPERIMENTS_DIR / "v1_1_boundary_positive_candidates.csv",
    )
    p.add_argument(
        "--output_report",
        type=Path,
        default=REPORTS_DIR / "v1_1_boundary_positive_candidates_report.json",
    )
    p.add_argument("--target_count", type=int, default=40)
    p.add_argument("--p_cal_min", type=float, default=0.55)
    p.add_argument("--p_cal_max", type=float, default=0.90)
    p.add_argument("--p_raw_std_min", type=float, default=0.02)
    p.add_argument("--min_dist_km_from_pos", type=float, default=2.0)
    p.add_argument("--max_dist_km_from_pos", type=float, default=25.0)
    p.add_argument("--geo_eps_km", type=float, default=6.0)
    p.add_argument("--max_per_geo_cluster", type=int, default=4)
    p.add_argument("--max_per_subregion", type=int, default=8)
    p.add_argument("--max_per_feature_bin", type=int, default=3)
    return p.parse_args()


def to_utm_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def load_models(path: Path):
    obj = joblib.load(path)
    return obj if isinstance(obj, list) else [obj]


def predict_ensemble(models, X: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    dm = xgb.DMatrix(X[feat_cols].to_numpy(dtype=np.float32), feature_names=feat_cols)
    preds = []
    for m in models:
        preds.append(m.get_booster().predict(dm))
    mat = np.vstack(preds)
    return mat.mean(axis=0), mat.std(axis=0)


def quantile_bin(series: pd.Series, q: int = 4) -> pd.Series:
    if series.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    return pd.qcut(series, q=q, labels=False, duplicates="drop").astype(int)


def subregion_ids(df: pd.DataFrame, lon_bins: int = 4, lat_bins: int = 3) -> np.ndarray:
    lon = pd.qcut(df["lon"], q=lon_bins, labels=False, duplicates="drop").astype(int)
    lat = pd.qcut(df["lat"], q=lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat * (int(lon.max()) + 1) + lon).to_numpy()


def geo_cluster_ids(df: pd.DataFrame, eps_km: float) -> np.ndarray:
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))
    eps = eps_km / 6371.0
    return DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    for p in [args.grid_csv, args.training_csv, args.model_bundle, args.calibrator, args.feature_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    grid = pd.read_csv(args.grid_csv)
    tr = pd.read_csv(args.training_csv)
    feat_cols = json.loads(args.feature_json.read_text(encoding="utf-8"))
    models = load_models(args.model_bundle)
    calibrator = joblib.load(args.calibrator)

    miss = [c for c in feat_cols if c not in grid.columns]
    if miss:
        raise ValueError(f"Missing model feature columns in grid: {miss}")

    raw_mean, raw_std = predict_ensemble(models, grid, feat_cols)
    p_cal = calibrator.predict(raw_mean)
    df = grid.copy()
    df["p_raw"] = raw_mean
    df["p_raw_std"] = raw_std
    df["p_calibrated"] = p_cal

    # Exclude existing positive cells from training.
    pos = tr[tr["label"] == 1][["lon", "lat"]].drop_duplicates()
    pos_keys = set(zip(pos["lon"].round(8), pos["lat"].round(8)))
    df = df[
        ~df[["lon", "lat"]]
        .apply(lambda r: (round(float(r["lon"]), 8), round(float(r["lat"]), 8)) in pos_keys, axis=1)
    ].copy()

    # Boundary-positive window + uncertainty filter.
    c = df[
        (df["p_calibrated"] >= args.p_cal_min)
        & (df["p_calibrated"] <= args.p_cal_max)
        & (df["p_raw_std"] >= args.p_raw_std_min)
    ].copy()

    # Ecological plausibility domain.
    if "shallow_mask" in c.columns:
        c = c[c["shallow_mask"] >= 1].copy()
    if "depth" in c.columns:
        c = c[(c["depth"] > 0.5) & (c["depth"] <= 8.0)].copy()
    if "distance_to_shore" in c.columns:
        c = c[c["distance_to_shore"] <= 7000].copy()

    # Distance band from known positives to avoid duplicates and far noise.
    px, py = to_utm_xy(pos["lon"].to_numpy(), pos["lat"].to_numpy())
    cx, cy = to_utm_xy(c["lon"].to_numpy(), c["lat"].to_numpy())
    tree = cKDTree(np.c_[px, py])
    d_m, _ = tree.query(np.c_[cx, cy], k=1)
    c["dist_to_pos_km"] = d_m / 1000.0
    c = c[
        (c["dist_to_pos_km"] >= args.min_dist_km_from_pos)
        & (c["dist_to_pos_km"] <= args.max_dist_km_from_pos)
    ].copy()

    if c.empty:
        raise RuntimeError("No candidates after filters. Relax p_cal or distance thresholds.")

    # Diversity tags.
    c = c.reset_index(drop=True)
    c["geo_cluster_id"] = geo_cluster_ids(c, args.geo_eps_km)
    c["subregion_id"] = subregion_ids(c)
    c["feat_bin_depth"] = quantile_bin(c["depth"], q=4) if "depth" in c.columns else 0
    c["feat_bin_turb"] = quantile_bin(c["turb_mean"], q=4) if "turb_mean" in c.columns else 0
    c["feat_bin_wave"] = quantile_bin(c["wave_mean"], q=4) if "wave_mean" in c.columns else 0
    c["feature_bin_id"] = (
        c["feat_bin_depth"].astype(str)
        + "_"
        + c["feat_bin_turb"].astype(str)
        + "_"
        + c["feat_bin_wave"].astype(str)
    )

    # Priority: high calibrated score, then high uncertainty, then farther from existing positives.
    c = c.sort_values(
        by=["p_calibrated", "p_raw_std", "dist_to_pos_km"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_rows = []
    geo_counts: dict[int, int] = {}
    sub_counts: dict[int, int] = {}
    feat_counts: dict[str, int] = {}

    for row in c.itertuples(index=False):
        g = int(row.geo_cluster_id)
        s = int(row.subregion_id)
        f = str(row.feature_bin_id)
        if geo_counts.get(g, 0) >= args.max_per_geo_cluster:
            continue
        if sub_counts.get(s, 0) >= args.max_per_subregion:
            continue
        if feat_counts.get(f, 0) >= args.max_per_feature_bin:
            continue
        selected_rows.append(row)
        geo_counts[g] = geo_counts.get(g, 0) + 1
        sub_counts[s] = sub_counts.get(s, 0) + 1
        feat_counts[f] = feat_counts.get(f, 0) + 1
        if len(selected_rows) >= args.target_count:
            break

    sel = pd.DataFrame(selected_rows)
    if sel.empty:
        raise RuntimeError("No candidates selected after diversity caps.")

    # Verification-pack schema: ready to copy into v1_1_data_plan.
    out = pd.DataFrame()
    out["record_id"] = [f"bp_v1_1_{i:04d}" for i in range(1, len(sel) + 1)]
    out["source_type"] = "satellite_digitized"
    out["source_name"] = "boundary_positive_mining_v1_1"
    out["source_reference"] = "model-guided candidate; requires manual verification"
    out["citation_url"] = ""
    out["species"] = "Kappaphycus alvarezii"
    out["eventDate"] = ""
    out["year"] = np.nan
    out["lon"] = sel["lon"].to_numpy()
    out["lat"] = sel["lat"].to_numpy()
    out["label"] = 1
    out["coordinate_precision_km"] = 1.0
    out["species_confirmed"] = False
    out["confidence_score"] = np.clip(sel["p_calibrated"].to_numpy(), 0, 1)
    out["is_verified"] = False
    out["qa_reviewer"] = ""
    out["qa_status"] = "pending"
    out["rationale"] = "high_model_score_with_uncertainty_and_diversity"
    out["notes"] = (
        "Verify by literature/satellite before ingestion. "
        "Auto-mined boundary-positive candidate."
    )
    out["p_raw"] = sel["p_raw"].to_numpy()
    out["p_raw_std"] = sel["p_raw_std"].to_numpy()
    out["p_calibrated"] = sel["p_calibrated"].to_numpy()
    out["dist_to_pos_km"] = sel["dist_to_pos_km"].to_numpy()
    out["geo_cluster_id"] = sel["geo_cluster_id"].to_numpy()
    out["subregion_id"] = sel["subregion_id"].to_numpy()
    out["feature_bin_id"] = sel["feature_bin_id"].to_numpy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    report = {
        "input_grid_rows": int(len(grid)),
        "after_exclude_existing_positive": int(len(df)),
        "after_boundary_filters": int(len(c)),
        "final_selected": int(len(out)),
        "mean_p_calibrated": float(out["p_calibrated"].mean()),
        "mean_p_raw_std": float(out["p_raw_std"].mean()),
        "min_dist_to_pos_km": float(out["dist_to_pos_km"].min()),
        "max_dist_to_pos_km": float(out["dist_to_pos_km"].max()),
        "output_csv": str(args.output_csv),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved boundary-positive candidates: {args.output_csv} | rows={len(out)}")
    print(f"Saved report: {args.output_report}")
    print(
        "Candidate stats -> mean p_cal {:.4f}, mean p_raw_std {:.4f}, dist_to_pos_km [{:.2f}, {:.2f}]".format(
            report["mean_p_calibrated"],
            report["mean_p_raw_std"],
            report["min_dist_to_pos_km"],
            report["max_dist_to_pos_km"],
        )
    )


if __name__ == "__main__":
    main()

