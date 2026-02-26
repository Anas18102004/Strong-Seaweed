import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from project_paths import TABULAR_DIR, REALTIME_MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, ensure_dirs, with_legacy

MODEL_PATH = with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_model.json", "xgboost_realtime_model.json")
MODEL_BUNDLE_PATH = with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_ensemble.pkl", "xgboost_realtime_ensemble.pkl")
CALIBRATOR_PATH = with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_calibrator.pkl", "xgboost_realtime_calibrator.pkl")
FEATURES_PATH = with_legacy(REALTIME_MODELS_DIR / "xgboost_realtime_features.json", "xgboost_realtime_features.json")
REPORT_PATH = with_legacy(REPORTS_DIR / "xgboost_realtime_report.json", "xgboost_realtime_report.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score real-time seaweed suitability candidates.")
    p.add_argument(
        "--input",
        type=Path,
        default=with_legacy(TABULAR_DIR / "master_feature_matrix_augmented.csv", "master_feature_matrix_augmented.csv"),
        help="Input CSV with lon, lat and model features.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "realtime_ranked_candidates.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--snapshot-tag",
        type=str,
        default="",
        help="Optional tag to also save a versioned snapshot (e.g., v1_0_baseline).",
    )
    p.add_argument(
        "--release_tag",
        type=str,
        default="",
        help="Optional release tag to load artifacts from releases/<tag>/models and report from releases/<tag>/reports.",
    )
    return p.parse_args()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(max(a, 0.0)))


def warn_scope_and_resolution(df: pd.DataFrame, report: dict) -> None:
    training_bbox = (
        report.get("training_region", {}).get("bbox", {})
    )
    if training_bbox:
        lon_min = float(training_bbox["lon_min"])
        lon_max = float(training_bbox["lon_max"])
        lat_min = float(training_bbox["lat_min"])
        lat_max = float(training_bbox["lat_max"])
        outside = (
            (df["lon"] < lon_min)
            | (df["lon"] > lon_max)
            | (df["lat"] < lat_min)
            | (df["lat"] > lat_max)
        ).sum()
        if outside > 0:
            print(
                "WARNING: "
                f"{int(outside)} input rows are outside training region bbox "
                f"(lon [{lon_min:.3f}, {lon_max:.3f}], lat [{lat_min:.3f}, {lat_max:.3f}])."
            )

    # Resolution warning for very close-point comparison
    if len(df) >= 2:
        coords = df[["lat", "lon"]].drop_duplicates().to_numpy()
        min_d = float("inf")
        for i in range(len(coords)):
            lat1, lon1 = float(coords[i][0]), float(coords[i][1])
            for j in range(i + 1, len(coords)):
                lat2, lon2 = float(coords[j][0]), float(coords[j][1])
                d = haversine_km(lat1, lon1, lat2, lon2)
                if d < min_d:
                    min_d = d
        if min_d < 5.0:
            print(
                "WARNING: "
                f"Minimum pairwise input distance is {min_d:.2f} km (<5 km). "
                "Micro-site comparisons are outside model validity scope."
            )


def main() -> None:
    ensure_dirs()
    args = parse_args()
    model_path = MODEL_PATH
    model_bundle_path = MODEL_BUNDLE_PATH
    calibrator_path = CALIBRATOR_PATH
    features_path = FEATURES_PATH
    report_path = REPORT_PATH
    release_snap_dir = None

    if args.release_tag.strip():
        safe = args.release_tag.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        rel_base = Path("releases") / safe
        model_path = rel_base / "models" / f"xgboost_realtime_model_{safe}.json"
        model_bundle_path = rel_base / "models" / f"xgboost_realtime_ensemble_{safe}.pkl"
        calibrator_path = rel_base / "models" / f"xgboost_realtime_calibrator_{safe}.pkl"
        features_path = rel_base / "models" / f"xgboost_realtime_features_{safe}.json"
        report_path = rel_base / "reports" / f"xgboost_realtime_report_{safe}.json"
        release_snap_dir = rel_base / "snapshots"
        release_snap_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    for p in [calibrator_path, features_path, report_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    features = json.loads(features_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(report["deployment_policy"]["recommended_threshold"])

    df = pd.read_csv(args.input)
    required = ["lon", "lat"] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    warn_scope_and_resolution(df, report)

    models = None
    if model_bundle_path.exists():
        models = joblib.load(model_bundle_path)
    elif model_path.exists():
        m = XGBClassifier()
        m.load_model(model_path)
        models = [m]
    else:
        raise FileNotFoundError(f"Missing model artifacts: {model_bundle_path} / {model_path}")
    calibrator = joblib.load(calibrator_path)

    X = df[features].to_numpy(dtype=np.float32)
    member_preds = [m.predict_proba(X)[:, 1] for m in models]
    raw = np.mean(np.vstack(member_preds), axis=0)
    raw_std = np.std(np.vstack(member_preds), axis=0)
    cal = calibrator.predict(raw)

    out = df[["lon", "lat"]].copy()
    out["p_raw"] = raw
    out["p_raw_std"] = raw_std
    out["p_calibrated"] = cal
    out["pred_label"] = (cal >= threshold).astype(np.int32)
    out["priority"] = np.where(
        cal >= 0.80,
        "high",
        np.where(cal >= 0.60, "medium", "low"),
    )
    out = out.sort_values("p_calibrated", ascending=False).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    if args.snapshot_tag:
        if release_snap_dir is not None:
            snap = release_snap_dir / f"realtime_ranked_candidates_{args.snapshot_tag}.csv"
        else:
            snap = args.output.with_name(f"{args.output.stem}_{args.snapshot_tag}.csv")
        out.to_csv(snap, index=False)
        print(f"Saved snapshot: {snap}")

    print(f"Scored rows: {len(out)}")
    print(f"Saved: {args.output}")
    print(f"Threshold: {threshold:.3f}")
    print("Predicted positives:", int(out["pred_label"].sum()))
    print("High priority:", int((out["priority"] == "high").sum()))
    print("Medium priority:", int((out["priority"] == "medium").sum()))


if __name__ == "__main__":
    main()
