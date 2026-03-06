import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from project_paths import TABULAR_DIR, ensure_dirs


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-species XGBoost models from multispecies training CSVs.")
    p.add_argument("--dataset_glob", type=str, default="training_dataset_*_multispecies_india_v1.csv")
    p.add_argument("--release_tag", type=str, default="multi_species_v1")
    p.add_argument("--min_positive", type=int, default=20)
    p.add_argument("--min_total_rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def spatial_groups(df: pd.DataFrame, lon_bins: int = 4, lat_bins: int = 3) -> np.ndarray:
    lon = pd.qcut(df["lon"], q=lon_bins, labels=False, duplicates="drop").astype(int)
    lat = pd.qcut(df["lat"], q=lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat * (int(lon.max()) + 1) + lon).to_numpy()


def group_folds(groups: np.ndarray, y: np.ndarray, min_pos_test: int = 2):
    folds = []
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        if y[te].sum() < min_pos_test:
            continue
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        if y[te].sum() == 0 or y[te].sum() == len(te):
            continue
        folds.append((tr, te))
    return folds


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.75,
    min_threshold: float = 0.5,
    max_threshold: float = 0.95,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    valid = np.where(
        (precision >= min_precision)
        & (thresholds >= min_threshold)
        & (thresholds <= max_threshold)
    )[0]
    if len(valid) > 0:
        i = valid[np.argmax(recall[valid])]
    else:
        bounded = np.where((thresholds >= min_threshold) & (thresholds <= max_threshold))[0]
        if len(bounded) > 0:
            f1_bounded = 2 * precision[bounded] * recall[bounded] / np.clip(precision[bounded] + recall[bounded], 1e-9, None)
            i = int(bounded[np.argmax(f1_bounded)])
        else:
            # If everything falls outside the bounded window, use the most conservative threshold available.
            i = int(np.argmax(thresholds))
    return {"threshold": float(thresholds[i]), "precision": float(precision[i]), "recall": float(recall[i])}


def train_one(df: pd.DataFrame, seed: int):
    drop_cols = {
        "label",
        "label_weight",
        "species_id",
        "species_target",
        "sample_type",
        "nearest_hp_km",
        "soft_similarity",
        "lon",
        "lat",
    }
    feat_cols = [c for c in df.columns if c not in drop_cols]
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in feat_cols if not c.startswith("soft_") and not c.startswith("nearest_hp")]
    feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 1]

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    if "label_weight" in df.columns:
        w = pd.to_numeric(df["label_weight"], errors="coerce").fillna(1.0).clip(lower=0.05).to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)
    groups = spatial_groups(df)
    folds = group_folds(groups, y, min_pos_test=2)
    if len(folds) < 2:
        raise RuntimeError(f"insufficient spatial folds: {len(folds)}")

    oof_raw = np.full(len(df), np.nan, dtype=np.float32)
    aucs, aps, briers = [], [], []
    for tr, te in folds:
        pos = int(y[tr].sum())
        neg = int(len(tr) - pos)
        spw = neg / max(pos, 1)
        m = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_estimators=120,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.50,
            colsample_bytree=0.50,
            min_child_weight=20,
            gamma=1.0,
            reg_alpha=2.0,
            reg_lambda=10.0,
            max_delta_step=2,
            scale_pos_weight=spw,
            n_jobs=-1,
        )
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof_raw[te] = p
        aucs.append(float(roc_auc_score(y[te], p)))
        aps.append(float(average_precision_score(y[te], p)))
        briers.append(float(brier_score_loss(y[te], p)))

    good = ~np.isnan(oof_raw)
    y_oof = y[good]
    p_oof = oof_raw[good]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_oof, y_oof)
    p_cal = np.clip(calibrator.predict(p_oof), 0.0, 1.0)
    thr = pick_threshold(y_oof, p_cal, min_precision=0.75, min_threshold=0.5)

    pos_full = int(y.sum())
    neg_full = int(len(y) - pos_full)
    spw_full = neg_full / max(pos_full, 1)
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_estimators=120,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.50,
        colsample_bytree=0.50,
        min_child_weight=20,
        gamma=1.0,
        reg_alpha=2.0,
        reg_lambda=10.0,
        max_delta_step=2,
        scale_pos_weight=spw_full,
        n_jobs=-1,
    )
    final_model.fit(X, y, sample_weight=w)

    metrics = {
        "n_rows": int(len(df)),
        "n_positive": pos_full,
        "n_negative": neg_full,
        "n_features": int(len(feat_cols)),
        "n_folds": int(len(folds)),
        "spatial_auc_mean": float(np.mean(aucs)),
        "spatial_auc_std": float(np.std(aucs)),
        "spatial_ap_mean": float(np.mean(aps)),
        "spatial_brier_mean": float(np.mean(briers)),
        "oof_auc_raw": float(roc_auc_score(y_oof, np.clip(p_oof, 0.0, 1.0))),
        "oof_auc_calibrated": float(roc_auc_score(y_oof, p_cal)),
        "oof_ap_raw": float(average_precision_score(y_oof, np.clip(p_oof, 0.0, 1.0))),
        "oof_ap_calibrated": float(average_precision_score(y_oof, p_cal)),
        "oof_brier_raw": float(brier_score_loss(y_oof, np.clip(p_oof, 0.0, 1.0))),
        "oof_brier_calibrated": float(brier_score_loss(y_oof, p_cal)),
        "threshold": thr,
    }
    return final_model, calibrator, feat_cols, metrics


def main() -> None:
    ensure_dirs()
    args = parse_args()
    dataset_paths = sorted(TABULAR_DIR.glob(args.dataset_glob))
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset files match: {args.dataset_glob}")

    release_base = Path("releases") / args.release_tag
    models_dir = release_base / "models"
    reports_dir = release_base / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {"release_tag": args.release_tag, "species": []}

    for path in dataset_paths:
        df = pd.read_csv(path)
        if "species_id" not in df.columns or "label" not in df.columns:
            log(f"[SKIP] {path.name}: missing species_id/label")
            continue
        species_id = str(df["species_id"].dropna().iloc[0]) if len(df) else path.stem
        df = df.dropna(subset=["lon", "lat", "label"]).drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].astype(int)

        pos = int(df["label"].sum())
        if len(df) < int(args.min_total_rows) or pos < int(args.min_positive):
            log(f"[SKIP] {species_id}: rows={len(df)} pos={pos} below thresholds")
            summary["species"].append(
                {"species_id": species_id, "status": "skipped", "reason": "insufficient_data", "rows": int(len(df)), "pos": pos}
            )
            continue

        log(f"[TRAIN] {species_id}: rows={len(df)} pos={pos}")
        try:
            model, calibrator, feat_cols, metrics = train_one(df, args.seed)
        except Exception as e:
            log(f"[FAIL] {species_id}: {e}")
            summary["species"].append({"species_id": species_id, "status": "failed", "error": str(e)})
            continue

        model_json = models_dir / f"xgb_{species_id}_{args.release_tag}.json"
        model_pkl = models_dir / f"xgb_{species_id}_{args.release_tag}.pkl"
        cal_pkl = models_dir / f"calibrator_{species_id}_{args.release_tag}.pkl"
        feat_json = models_dir / f"features_{species_id}_{args.release_tag}.json"
        report_json = reports_dir / f"report_{species_id}_{args.release_tag}.json"

        model.save_model(model_json)
        joblib.dump(model, model_pkl)
        joblib.dump(calibrator, cal_pkl)
        feat_json.write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
        report_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        summary["species"].append(
            {
                "species_id": species_id,
                "status": "trained",
                "rows": int(len(df)),
                "pos": pos,
                "model_json": str(model_json),
                "model_pkl": str(model_pkl),
                "calibrator_pkl": str(cal_pkl),
                "features_json": str(feat_json),
                "report_json": str(report_json),
                "metrics": metrics,
            }
        )
        log(
            f"[OK] {species_id}: auc={metrics['spatial_auc_mean']:.4f} ap={metrics['spatial_ap_mean']:.4f} "
            f"thr={metrics['threshold']['threshold']:.3f}"
        )

    summary_path = reports_dir / f"multispecies_training_summary_{args.release_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
