import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from project_paths import (
    TABULAR_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine regime-stratified positive candidates for verification."
    )
    p.add_argument(
        "--grid_csv",
        type=Path,
        default=TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv",
    )
    p.add_argument(
        "--pred_csv",
        type=Path,
        default=OUTPUTS_DIR / "realtime_ranked_candidates_v1_1.csv",
    )
    p.add_argument(
        "--training_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_merged46_plus_hn30_augmented.csv",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=EXPERIMENTS_DIR / "v1_1p1_regime_positive_candidates.csv",
    )
    p.add_argument(
        "--output_report",
        type=Path,
        default=REPORTS_DIR / "v1_1p1_regime_positive_candidates_report.json",
    )
    p.add_argument("--target_count", type=int, default=60)
    p.add_argument("--min_dist_km_from_pos", type=float, default=2.5)
    p.add_argument("--max_dist_km_from_pos", type=float, default=40.0)
    p.add_argument("--p_cal_min", type=float, default=0.45)
    p.add_argument("--p_cal_max", type=float, default=0.95)
    p.add_argument("--max_per_regime_bin", type=int, default=2)
    p.add_argument("--max_per_subregion", type=int, default=6)
    return p.parse_args()


def to_utm_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def qbin(series: pd.Series, q: int = 4) -> pd.Series:
    if series.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    return pd.qcut(series, q=q, labels=False, duplicates="drop").astype(int)


def subregion_ids(df: pd.DataFrame, lon_bins: int = 4, lat_bins: int = 3) -> np.ndarray:
    lon = pd.qcut(df["lon"], q=lon_bins, labels=False, duplicates="drop").astype(int)
    lat = pd.qcut(df["lat"], q=lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat * (int(lon.max()) + 1) + lon).to_numpy()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    for p in [args.grid_csv, args.pred_csv, args.training_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    grid = pd.read_csv(args.grid_csv)
    pred = pd.read_csv(args.pred_csv)
    tr = pd.read_csv(args.training_csv)

    if "p_calibrated" not in pred.columns:
        raise ValueError("pred_csv missing p_calibrated. Run score_realtime_production first.")

    # Merge predicted scores with full feature grid.
    df = grid.merge(
        pred[["lon", "lat", "p_raw", "p_raw_std", "p_calibrated"]],
        on=["lon", "lat"],
        how="inner",
    )
    n_grid = len(df)

    # Exclude known positives already in training.
    pos = tr[tr["label"] == 1][["lon", "lat"]].drop_duplicates()
    pos_keys = set(zip(pos["lon"].round(8), pos["lat"].round(8)))
    df = df[
        ~df[["lon", "lat"]]
        .apply(lambda r: (round(float(r["lon"]), 8), round(float(r["lat"]), 8)) in pos_keys, axis=1)
    ].copy()
    n_after_exclude_pos = len(df)

    # Ecological plausibility gates.
    c = df.copy()
    if "shallow_mask" in c.columns:
        c = c[c["shallow_mask"] >= 1].copy()
    if "depth" in c.columns:
        c = c[(c["depth"] > 0.5) & (c["depth"] <= 8.0)].copy()
    if "distance_to_shore" in c.columns:
        c = c[c["distance_to_shore"] <= 5000].copy()

    # Score window for candidate positives.
    c = c[
        (c["p_calibrated"] >= args.p_cal_min)
        & (c["p_calibrated"] <= args.p_cal_max)
    ].copy()

    if c.empty:
        raise RuntimeError("No candidates after ecological+score filters.")

    # Distance band from known positives: near enough to be plausible, far enough to add information.
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
        raise RuntimeError("No candidates in distance band. Relax min/max distance thresholds.")

    # Regime bins force environmental diversity.
    for req in ["depth", "turb_mean", "wave_mean", "sal_std"]:
        if req not in c.columns:
            raise RuntimeError(f"Missing required feature for regime bins: {req}")

    c = c.reset_index(drop=True)
    c["bin_depth"] = qbin(c["depth"], q=4)
    c["bin_turb"] = qbin(c["turb_mean"], q=4)
    c["bin_wave"] = qbin(c["wave_mean"], q=4)
    c["bin_salstd"] = qbin(c["sal_std"], q=4)
    c["regime_bin"] = (
        c["bin_depth"].astype(str)
        + "_"
        + c["bin_turb"].astype(str)
        + "_"
        + c["bin_wave"].astype(str)
        + "_"
        + c["bin_salstd"].astype(str)
    )
    c["subregion_id"] = subregion_ids(c)

    # Prioritize higher confidence and farther from existing positives.
    c = c.sort_values(
        by=["p_calibrated", "dist_to_pos_km", "p_raw_std"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected = []
    regime_counts: dict[str, int] = {}
    subregion_counts: dict[int, int] = {}

    for row in c.itertuples(index=False):
        rb = str(row.regime_bin)
        sr = int(row.subregion_id)
        if regime_counts.get(rb, 0) >= args.max_per_regime_bin:
            continue
        if subregion_counts.get(sr, 0) >= args.max_per_subregion:
            continue
        selected.append(row)
        regime_counts[rb] = regime_counts.get(rb, 0) + 1
        subregion_counts[sr] = subregion_counts.get(sr, 0) + 1
        if len(selected) >= args.target_count:
            break

    sel = pd.DataFrame(selected)
    if sel.empty:
        raise RuntimeError("No candidates selected after diversity caps.")

    # Output in ingestion-ready governance schema (pending verification).
    out = pd.DataFrame()
    out["record_id"] = [f"rp_v1_1p1_{i:04d}" for i in range(1, len(sel) + 1)]
    out["source_type"] = "satellite_digitized"
    out["source_name"] = "regime_positive_mining_v1_1p1"
    out["source_reference"] = "model-guided regime-stratified candidate; needs manual verification"
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
    out["rationale"] = "regime_diversity_candidate"
    out["notes"] = (
        "verify_by_multidate_satellite>=3_dates,persistent_raft>=2_dates,"
        "not_port_shipping,verified_dates=YYYY-MM-DD;YYYY-MM-DD"
    )
    out["p_raw"] = sel["p_raw"].to_numpy()
    out["p_raw_std"] = sel["p_raw_std"].to_numpy()
    out["p_calibrated"] = sel["p_calibrated"].to_numpy()
    out["dist_to_pos_km"] = sel["dist_to_pos_km"].to_numpy()
    out["regime_bin"] = sel["regime_bin"].to_numpy()
    out["subregion_id"] = sel["subregion_id"].to_numpy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    report = {
        "input_grid_rows": int(n_grid),
        "after_excluding_existing_positives": int(n_after_exclude_pos),
        "after_filters": int(len(c)),
        "selected_count": int(len(out)),
        "selected_regime_bins": int(out["regime_bin"].nunique()),
        "selected_subregions": int(out["subregion_id"].nunique()),
        "mean_p_calibrated": float(out["p_calibrated"].mean()),
        "mean_dist_to_pos_km": float(out["dist_to_pos_km"].mean()),
        "min_dist_to_pos_km": float(out["dist_to_pos_km"].min()),
        "max_dist_to_pos_km": float(out["dist_to_pos_km"].max()),
        "output_csv": str(args.output_csv),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved regime-positive candidates: {args.output_csv} | rows={len(out)}")
    print(f"Saved report: {args.output_report}")
    print(
        "Selected bins/subregions: {}/{} | p_cal mean {:.4f} | dist_to_pos_km [{:.2f}, {:.2f}]".format(
            report["selected_regime_bins"],
            report["selected_subregions"],
            report["mean_p_calibrated"],
            report["min_dist_to_pos_km"],
            report["max_dist_to_pos_km"],
        )
    )


if __name__ == "__main__":
    main()
