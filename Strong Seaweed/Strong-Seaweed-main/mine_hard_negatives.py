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
    REALTIME_MODELS_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    OUTPUTS_DIR,
    ensure_dirs,
    with_legacy,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine ecologically plausible hard negatives for v1.1.")
    p.add_argument("--grid_csv", type=Path, default=with_legacy(TABULAR_DIR / "master_feature_matrix_augmented.csv", "master_feature_matrix_augmented.csv"))
    p.add_argument("--pred_csv", type=Path, default=with_legacy(OUTPUTS_DIR / "realtime_ranked_candidates.csv", "realtime_ranked_candidates.csv"))
    p.add_argument("--training_csv", type=Path, default=with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv"))
    p.add_argument("--model_bundle", type=Path, default=with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_ensemble.pkl", "xgboost_realtime_ensemble.pkl"))
    p.add_argument("--feature_json", type=Path, default=with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_features.json", "xgboost_realtime_features.json"))
    p.add_argument("--output_csv", type=Path, default=EXPERIMENTS_DIR / "v1_1_hard_negatives.csv")
    p.add_argument("--output_report", type=Path, default=REPORTS_DIR / "v1_1_hard_negatives_report.json")

    p.add_argument("--target_count", type=int, default=20)
    p.add_argument("--p_raw_min", type=float, default=0.35)
    p.add_argument("--p_raw_max", type=float, default=0.65)
    p.add_argument("--depth_min", type=float, default=1.0)
    p.add_argument("--depth_max", type=float, default=8.0)
    p.add_argument("--min_dist_km", type=float, default=3.0)
    p.add_argument("--min_feature_dist", type=float, default=0.35)
    p.add_argument("--max_shap_dominance", type=float, default=0.65)
    p.add_argument("--geo_eps_km", type=float, default=5.0)
    p.add_argument("--max_per_geo_cluster", type=int, default=3)
    p.add_argument("--max_per_subregion", type=int, default=6)
    p.add_argument("--max_per_feature_bin", type=int, default=3)
    return p.parse_args()


def to_utm_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def compute_geo_cluster_ids(df: pd.DataFrame, eps_km: float) -> np.ndarray:
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))
    # Haversine with radians and Earth radius in km.
    eps = eps_km / 6371.0
    labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)
    return labels


def subregion_ids(df: pd.DataFrame, lon_bins: int = 4, lat_bins: int = 3) -> np.ndarray:
    lon = pd.qcut(df["lon"], q=lon_bins, labels=False, duplicates="drop").astype(int)
    lat = pd.qcut(df["lat"], q=lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat * (int(lon.max()) + 1) + lon).to_numpy()


def quantile_bin(series: pd.Series, q: int = 4) -> pd.Series:
    if series.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    return pd.qcut(series, q=q, labels=False, duplicates="drop").astype(int)


def load_models(path: Path):
    models = joblib.load(path)
    if isinstance(models, list):
        return models
    return [models]


def mean_shap_contribs(models, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    dm = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=feature_cols)
    acc = np.zeros((len(X), len(feature_cols) + 1), dtype=float)
    for m in models:
        contrib = m.get_booster().predict(dm, pred_contribs=True)
        acc += contrib
    acc /= max(len(models), 1)
    return acc[:, :-1]


def main() -> None:
    ensure_dirs()
    args = parse_args()
    for p in [args.grid_csv, args.pred_csv, args.training_csv, args.model_bundle, args.feature_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    grid = pd.read_csv(args.grid_csv)
    pred = pd.read_csv(args.pred_csv)
    train = pd.read_csv(args.training_csv)
    feature_cols = json.loads(args.feature_json.read_text(encoding="utf-8"))
    models = load_models(args.model_bundle)

    if "p_raw" not in pred.columns:
        raise ValueError("pred_csv must contain p_raw column. Re-run score_realtime_production.py first.")

    df = grid.merge(pred[["lon", "lat", "p_raw", "p_calibrated"]], on=["lon", "lat"], how="inner")
    n_all = len(df)

    # Baseline plausible shallow coastal filter.
    shallow = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if "shallow_mask" in df.columns:
        shallow &= df["shallow_mask"] >= 1
    if "depth" in df.columns:
        shallow &= (df["depth"] >= args.depth_min) & (df["depth"] <= args.depth_max)

    # Boundary mining on raw model probabilities.
    boundary = (df["p_raw"] >= args.p_raw_min) & (df["p_raw"] <= args.p_raw_max)
    c = df[shallow & boundary].copy()
    n_boundary = len(c)

    # Ecological stress rules (hard but plausible).
    base_for_quantiles = df[shallow].copy()
    stress_flags = []

    if "turb_mean" in c.columns:
        q = float(base_for_quantiles["turb_mean"].quantile(0.75))
        flag = c["turb_mean"] >= q
        c["stress_turb_high"] = flag.astype(int)
        stress_flags.append("stress_turb_high")
    if "wave_std" in c.columns:
        q = float(base_for_quantiles["wave_std"].quantile(0.75))
        flag = c["wave_std"] >= q
        c["stress_wave_std_high"] = flag.astype(int)
        stress_flags.append("stress_wave_std_high")
    if "sal_shock_days" in c.columns:
        q = float(base_for_quantiles["sal_shock_days"].quantile(0.75))
        flag = c["sal_shock_days"] >= q
        c["stress_sal_shock_high"] = flag.astype(int)
        stress_flags.append("stress_sal_shock_high")
    if "slope" in c.columns:
        q = float(base_for_quantiles["slope"].quantile(0.75))
        flag = c["slope"] >= q
        c["stress_slope_upper"] = flag.astype(int)
        stress_flags.append("stress_slope_upper")

    if not stress_flags:
        raise RuntimeError("No stress features available for hard-negative mining.")

    c["stress_count"] = c[stress_flags].sum(axis=1)
    c = c[c["stress_count"] >= 1].copy()
    n_stress = len(c)

    # Remove known positives and enforce spatial distance from positives.
    pos = train[train["label"] == 1][["lon", "lat"]].drop_duplicates().copy()
    # Stay within v1.0 training support to avoid out-of-domain negatives.
    bbox_mask = (
        (c["lon"] >= train["lon"].min())
        & (c["lon"] <= train["lon"].max())
        & (c["lat"] >= train["lat"].min())
        & (c["lat"] <= train["lat"].max())
    )
    c = c[bbox_mask].copy()
    pos_key = set(zip(pos["lon"].round(8), pos["lat"].round(8)))
    c["is_positive_cell"] = [
        (round(lon, 8), round(lat, 8)) in pos_key for lon, lat in zip(c["lon"], c["lat"])
    ]
    c = c[~c["is_positive_cell"]].copy()

    px, py = to_utm_xy(pos["lon"].to_numpy(), pos["lat"].to_numpy())
    cx, cy = to_utm_xy(c["lon"].to_numpy(), c["lat"].to_numpy())
    tree_pos = cKDTree(np.c_[px, py])
    dist_m, _ = tree_pos.query(np.c_[cx, cy], k=1)
    c["dist_to_nearest_positive_km"] = dist_m / 1000.0
    c = c[c["dist_to_nearest_positive_km"] >= args.min_dist_km].copy()
    n_spatial = len(c)

    # Feature distance in (turb_mean, slope) space.
    for req in ["turb_mean", "slope"]:
        if req not in c.columns or req not in pos.columns and req not in train.columns:
            raise RuntimeError(f"Missing required feature for feature-distance filter: {req}")

    pos_feat = (
        train[train["label"] == 1][["turb_mean", "slope"]]
        .dropna()
        .to_numpy(dtype=float)
    )
    cand_feat = c[["turb_mean", "slope"]].to_numpy(dtype=float)
    all_feat = df[["turb_mean", "slope"]].dropna().to_numpy(dtype=float)
    mu = np.mean(all_feat, axis=0)
    sd = np.std(all_feat, axis=0)
    sd = np.where(sd <= 1e-9, 1.0, sd)

    pos_z = (pos_feat - mu) / sd
    cand_z = (cand_feat - mu) / sd
    tree_feat = cKDTree(pos_z)
    fdist, _ = tree_feat.query(cand_z, k=1)
    c["feature_dist_turb_slope"] = fdist
    c = c[c["feature_dist_turb_slope"] >= args.min_feature_dist].copy()
    n_feature = len(c)

    # SHAP safeguard: reject single-feature-dominated candidates.
    available_features = [f for f in feature_cols if f in c.columns]
    if len(available_features) < 5:
        raise RuntimeError("Too few model features available in candidate table for SHAP filtering.")
    shap_vals = mean_shap_contribs(models, c[available_features], available_features)
    abs_shap = np.abs(shap_vals)
    denom = np.clip(abs_shap.sum(axis=1), 1e-9, None)
    dominance = abs_shap.max(axis=1) / denom
    top_idx = np.argmax(abs_shap, axis=1)
    c["shap_dominance_ratio"] = dominance
    c["shap_top_feature"] = [available_features[i] for i in top_idx]
    c = c[c["shap_dominance_ratio"] <= args.max_shap_dominance].copy()
    n_shap = len(c)

    if len(c) == 0:
        raise RuntimeError("No candidates left after filters. Relax thresholds.")

    # Diversity tags.
    c = c.reset_index(drop=True)
    c["geo_cluster_id"] = compute_geo_cluster_ids(c, args.geo_eps_km)
    c["subregion_id"] = subregion_ids(c)
    c["feat_bin_turb"] = quantile_bin(c["turb_mean"], q=4)
    c["feat_bin_slope"] = quantile_bin(c["slope"], q=4)
    c["feature_bin_id"] = c["feat_bin_turb"].astype(str) + "_" + c["feat_bin_slope"].astype(str)

    # Selection priority: near boundary center and higher stress.
    c["boundary_closeness"] = np.abs(c["p_raw"] - 0.5)
    c = c.sort_values(
        by=["boundary_closeness", "stress_count", "dist_to_nearest_positive_km"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    # Cap-based selection.
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

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise RuntimeError("No curated hard negatives selected after diversity caps.")

    # Governance fields for ingestion.
    selected["record_id"] = [
        f"hn_v1_1_{i:04d}" for i in range(1, len(selected) + 1)
    ]
    selected["source_type"] = "derived_hard_negative"
    selected["source_name"] = "mine_hard_negatives_v1_1"
    selected["citation_url"] = ""
    selected["species"] = "Kappaphycus alvarezii"
    selected["eventDate"] = ""
    selected["year"] = np.nan
    selected["label"] = 0
    selected["is_verified"] = True
    selected["qa_status"] = "approved"
    selected["rationale"] = selected[stress_flags].apply(
        lambda r: ",".join([k for k in stress_flags if int(r[k]) == 1]),
        axis=1,
    )
    selected["notes"] = (
        "Boundary-mined hard negative; ecologically plausible but stressed."
    )

    out_cols = [
        "record_id",
        "source_type",
        "source_name",
        "citation_url",
        "species",
        "eventDate",
        "year",
        "lon",
        "lat",
        "label",
        "is_verified",
        "qa_status",
        "p_raw",
        "p_calibrated",
        "stress_count",
        "dist_to_nearest_positive_km",
        "feature_dist_turb_slope",
        "shap_dominance_ratio",
        "shap_top_feature",
        "rationale",
        "notes",
    ]
    selected = selected[out_cols].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)

    report = {
        "total_grid_rows": int(n_all),
        "after_boundary_filter": int(n_boundary),
        "after_stress_filter": int(n_stress),
        "after_spatial_distance_filter": int(n_spatial),
        "after_feature_distance_filter": int(n_feature),
        "after_shap_safeguard": int(n_shap),
        "final_curated_count": int(len(selected)),
        "mean_p_raw_selected": float(selected["p_raw"].mean()),
        "min_distance_to_positive_km_selected": float(selected["dist_to_nearest_positive_km"].min()),
        "mean_distance_to_positive_km_selected": float(selected["dist_to_nearest_positive_km"].mean()),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_csv": str(args.output_csv),
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved hard negatives: {args.output_csv} | rows={len(selected)}")
    print(f"Saved report: {args.output_report}")
    print(f"Total candidates mined: {n_shap}")
    print(f"Final curated count: {len(selected)}")
    print(f"Mean p_raw selected: {selected['p_raw'].mean():.4f}")
    print(
        f"Min distance to nearest positive (km): "
        f"{selected['dist_to_nearest_positive_km'].min():.4f}"
    )


if __name__ == "__main__":
    main()
