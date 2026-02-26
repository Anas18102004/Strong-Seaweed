import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from project_paths import TABULAR_DIR, REALTIME_MODELS_DIR, REPORTS_DIR, DIAGNOSTICS_DIR, DOCS_DIR, ensure_dirs, with_legacy


DATASETS = [
    with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv"),
    with_legacy(TABULAR_DIR / "training_dataset_augmented.csv", "training_dataset_augmented.csv"),
]
INFERENCE_FEATURE_SOURCE = with_legacy(TABULAR_DIR / "master_feature_matrix.csv", "master_feature_matrix.csv")
INFERENCE_FEATURE_SOURCE_AUG = with_legacy(TABULAR_DIR / "master_feature_matrix_augmented.csv", "master_feature_matrix_augmented.csv")

MODEL_PATH = REALTIME_MODELS_DIR / "xgboost_realtime_model.json"
MODEL_BUNDLE_PATH = REALTIME_MODELS_DIR / "xgboost_realtime_ensemble.pkl"
CALIBRATOR_PATH = REALTIME_MODELS_DIR / "xgboost_realtime_calibrator.pkl"
FEATURES_PATH = REALTIME_MODELS_DIR / "xgboost_realtime_features.json"
REPORT_PATH = REPORTS_DIR / "xgboost_realtime_report.json"
PRED_PATH = DIAGNOSTICS_DIR / "xgboost_realtime_oof_predictions.csv"

RANDOM_STATE = 42
ENSEMBLE_SEEDS = [42, 52, 62]
SINGLE_SEEDS = [42]
MODEL_VERSION = "v1.0"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train deployable real-time seaweed model.")
    p.add_argument("--fast", action="store_true", help="Use fewer hyperparameter samples for faster runs.")
    p.add_argument("--n_iter_single", type=int, default=None, help="Override hyperparameter trials for single model mode.")
    p.add_argument("--n_iter_ensemble", type=int, default=None, help="Override hyperparameter trials for ensemble mode.")
    p.add_argument("--production", action="store_true", help="Apply production safety constraints for thresholding.")
    p.add_argument(
        "--min_threshold",
        type=float,
        default=0.50,
        help="Minimum allowed deployment threshold in production mode.",
    )
    p.add_argument(
        "--release_tag",
        type=str,
        default="",
        help="Optional release tag (e.g., v1.1). If set, writes artifacts to releases/<tag>/.",
    )
    p.add_argument(
        "--dataset_paths",
        nargs="*",
        default=[],
        help="Optional explicit training dataset CSV paths. Overrides default training_dataset*.csv.",
    )
    p.add_argument(
        "--inference_feature_source",
        type=str,
        default="",
        help="Optional explicit inference feature matrix CSV path.",
    )
    return p.parse_args()


def resolve_output_paths(release_tag: str) -> dict[str, Path]:
    tag = (release_tag or "").strip()
    if not tag:
        return {
            "model": MODEL_PATH,
            "bundle": MODEL_BUNDLE_PATH,
            "calibrator": CALIBRATOR_PATH,
            "features": FEATURES_PATH,
            "report": REPORT_PATH,
            "pred": PRED_PATH,
            "model_card": DOCS_DIR / f"MODEL_CARD_{MODEL_VERSION}.md",
            "version": MODEL_VERSION,
        }

    safe = tag.replace("/", "_").replace("\\", "_").replace(" ", "_")
    release_base = Path("releases") / safe
    (release_base / "models").mkdir(parents=True, exist_ok=True)
    (release_base / "reports").mkdir(parents=True, exist_ok=True)
    (release_base / "snapshots").mkdir(parents=True, exist_ok=True)
    return {
        "model": release_base / "models" / f"xgboost_realtime_model_{safe}.json",
        "bundle": release_base / "models" / f"xgboost_realtime_ensemble_{safe}.pkl",
        "calibrator": release_base / "models" / f"xgboost_realtime_calibrator_{safe}.pkl",
        "features": release_base / "models" / f"xgboost_realtime_features_{safe}.json",
        "report": release_base / "reports" / f"xgboost_realtime_report_{safe}.json",
        "pred": release_base / "reports" / f"xgboost_realtime_oof_predictions_{safe}.csv",
        "model_card": release_base / "reports" / f"MODEL_CARD_{safe}.md",
        "version": safe,
    }


def spatial_groups(df: pd.DataFrame, lon_bins: int = 4, lat_bins: int = 3) -> np.ndarray:
    lon = pd.qcut(df["lon"], q=lon_bins, labels=False, duplicates="drop").astype(int)
    lat = pd.qcut(df["lat"], q=lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat * (int(lon.max()) + 1) + lon).to_numpy()


def group_folds(groups: np.ndarray, y: np.ndarray, min_pos_test: int = 4):
    folds = []
    for g in np.unique(groups):
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if y[test_idx].sum() < min_pos_test:
            continue
        # Require both classes in both splits for valid ranking metrics.
        if y[train_idx].sum() == 0:
            continue
        if y[train_idx].sum() == len(train_idx):
            continue
        if y[test_idx].sum() == 0:
            continue
        if y[test_idx].sum() == len(test_idx):
            continue
        folds.append((train_idx, test_idx, int(g)))
    return folds


def fit_ensemble(X_train: np.ndarray, y_train: np.ndarray, params: dict, seeds: list[int]):
    models = []
    for seed in seeds:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        models.append(model)
    return models


def predict_ensemble(models, X_eval: np.ndarray) -> np.ndarray:
    preds = [m.predict_proba(X_eval)[:, 1] for m in models]
    return np.mean(np.vstack(preds), axis=0)


def evaluate_params(X: np.ndarray, y: np.ndarray, folds, params: dict, seeds: list[int]):
    aucs, aps, briers = [], [], []
    for tr, te, _ in folds:
        models = fit_ensemble(X[tr], y[tr], params, seeds)
        pred = predict_ensemble(models, X[te])
        if np.unique(y[te]).size == 2:
            aucs.append(float(roc_auc_score(y[te], pred)))
        aps.append(float(average_precision_score(y[te], pred)))
        briers.append(float(brier_score_loss(y[te], pred)))

    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "ap_mean": float(np.mean(aps)),
        "brier_mean": float(np.mean(briers)),
    }


def oof_predictions(
    X: np.ndarray, y: np.ndarray, folds, params: dict, seeds: list[int]
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te, _ in folds:
        models = fit_ensemble(X[tr], y[tr], params, seeds)
        pred[te] = predict_ensemble(models, X[te])
    return pred


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.75,
    min_threshold: float = 0.0,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall has one extra element; align to thresholds.
    precision = precision[:-1]
    recall = recall[:-1]
    thresholds = thresholds

    valid = np.where((precision >= min_precision) & (thresholds >= min_threshold))[0]
    if len(valid) > 0:
        i = valid[np.argmax(recall[valid])]
    else:
        # fallback: maximize F1 subject to threshold floor.
        floor_valid = np.where(thresholds >= min_threshold)[0]
        candidate_idx = floor_valid if len(floor_valid) > 0 else np.arange(len(thresholds))
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
        i = int(candidate_idx[np.argmax(f1[candidate_idx])])

    return {
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
    }


def train_dataset(
    df: pd.DataFrame,
    name: str,
    feat_cols: list[str],
    n_iter_single: int,
    n_iter_ensemble: int,
    min_threshold: float,
) -> dict:
    feat_cols = [c for c in feat_cols if c in df.columns]
    feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 1]
    if len(feat_cols) < 8:
        raise RuntimeError(f"{name}: too few usable features ({len(feat_cols)})")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = spatial_groups(df)
    folds = group_folds(groups, y, min_pos_test=4)
    if len(folds) < 4:
        folds = group_folds(groups, y, min_pos_test=2)
    if len(folds) < 4:
        folds = group_folds(groups, y, min_pos_test=1)
    if len(folds) < 2:
        raise RuntimeError(f"{name}: insufficient spatial folds ({len(folds)})")

    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / max(pos, 1)

    space = {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 2, 4, 6],
        "gamma": [0.0, 0.1, 0.3, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "scale_pos_weight": [spw],
    }
    log(f"{name}: rows={len(df)}, positives={int(y.sum())}, features={len(feat_cols)}, folds={len(folds)}")

    best = None
    mode_specs = [
        ("single", SINGLE_SEEDS, n_iter_single),
        ("ensemble", ENSEMBLE_SEEDS, n_iter_ensemble),
    ]

    for mode_name, seeds, n_iter in mode_specs:
        log(f"{name}: tuning mode={mode_name}, seeds={len(seeds)}, trials={n_iter}")
        candidates = list(
            ParameterSampler(
                space,
                n_iter=n_iter,
                random_state=RANDOM_STATE + (0 if mode_name == "single" else 1000),
            )
        )
        for i, p in enumerate(candidates, start=1):
            score = evaluate_params(X, y, folds, p, seeds)
            rec = {"params": p, "mode": mode_name, "seeds": seeds, **score}
            if best is None:
                best = rec
                continue
            # Prioritize AP (rarer-class quality), then calibration error, then AUC.
            if (
                rec["ap_mean"] > best["ap_mean"]
                or (
                    np.isclose(rec["ap_mean"], best["ap_mean"])
                    and rec["brier_mean"] < best["brier_mean"]
                )
                or (
                    np.isclose(rec["ap_mean"], best["ap_mean"])
                    and np.isclose(rec["brier_mean"], best["brier_mean"])
                    and rec["auc_mean"] > best["auc_mean"]
                )
            ):
                best = rec
            if i % 5 == 0 or i == len(candidates):
                log(
                    f"{name}: {mode_name} trial {i}/{len(candidates)} | "
                    f"best_ap={best['ap_mean']:.4f} best_auc={best['auc_mean']:.4f} "
                    f"best_brier={best['brier_mean']:.4f} mode={best['mode']}"
                )

    raw_oof_full = oof_predictions(X, y, folds, best["params"], best["seeds"])
    good = ~np.isnan(raw_oof_full)
    raw_oof = raw_oof_full[good]
    y_oof = y[good]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_oof, y_oof)
    cal_oof = calibrator.predict(raw_oof)

    threshold_info = pick_threshold(
        y_oof,
        cal_oof,
        min_precision=0.75,
        min_threshold=min_threshold,
    )

    final_models = fit_ensemble(X, y, best["params"], best["seeds"])
    train_raw = predict_ensemble(final_models, X)
    train_cal = calibrator.predict(train_raw)

    oof_cal_full = np.full(len(y), np.nan, dtype=np.float32)
    oof_cal_full[good] = calibrator.predict(raw_oof_full[good]).astype(np.float32)

    dataset_result = {
        "name": name,
        "df": df,
        "feature_cols": feat_cols,
        "models": final_models,
        "calibrator": calibrator,
        "oof_raw_full": raw_oof_full,
        "oof_cal_full": oof_cal_full,
        "oof_raw": raw_oof,
        "oof_cal": cal_oof,
        "oof_y": y_oof,
        "train_raw": train_raw,
        "train_cal": train_cal,
        "metrics": {
            "n_samples": int(len(df)),
            "n_presence": pos,
            "n_background": neg,
            "n_features": int(len(feat_cols)),
            "training_mode": best["mode"],
            "n_ensemble_members": int(len(best["seeds"])),
            "n_spatial_folds": int(len(folds)),
            "best_params": best["params"],
            "spatial_auc_mean": best["auc_mean"],
            "spatial_auc_std": best["auc_std"],
            "spatial_ap_mean": best["ap_mean"],
            "spatial_brier_mean": best["brier_mean"],
            "oof_auc_raw": float(roc_auc_score(y_oof, raw_oof)),
            "oof_auc_calibrated": float(roc_auc_score(y_oof, cal_oof)),
            "oof_ap_raw": float(average_precision_score(y_oof, raw_oof)),
            "oof_ap_calibrated": float(average_precision_score(y_oof, cal_oof)),
            "oof_brier_raw": float(brier_score_loss(y_oof, raw_oof)),
            "oof_brier_calibrated": float(brier_score_loss(y_oof, cal_oof)),
            "threshold": threshold_info,
        },
    }
    return dataset_result


def build_training_configs(
    datasets: list[tuple[str, pd.DataFrame]],
    deployable_columns: set[str],
) -> list[tuple[str, pd.DataFrame, list[str]]]:
    configs: list[tuple[str, pd.DataFrame, list[str]]] = []

    # Per-dataset deployable configs
    for name, df in datasets:
        base_features = [c for c in df.columns if c not in {"label", "lon", "lat"}]
        deployable_features = [c for c in base_features if c in deployable_columns]
        if len(deployable_features) >= 8:
            configs.append((f"{name}_deployable", df.copy(), deployable_features))

    # Combined deployable config using common features across datasets
    if len(datasets) >= 2:
        common = set(deployable_columns)
        for _, df in datasets:
            common &= set(df.columns)
        common -= {"label", "lon", "lat"}

        if len(common) >= 8:
            merged_parts = []
            keep_cols = ["lon", "lat", "label"] + sorted(common)
            for _, df in datasets:
                tmp = df[keep_cols].copy()
                tmp = tmp.dropna().drop_duplicates(subset=["lon", "lat", "label"])
                merged_parts.append(tmp)
            merged = pd.concat(merged_parts, ignore_index=True)
            merged = merged.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
            if len(merged) > 150 and int(merged["label"].sum()) > 20:
                configs.append(("combined_deployable", merged, sorted(common)))

    return configs


def main() -> None:
    ensure_dirs()
    args = parse_args()
    out_paths = resolve_output_paths(args.release_tag)
    n_iter_single = args.n_iter_single if args.n_iter_single is not None else (20 if args.fast else 45)
    n_iter_ensemble = args.n_iter_ensemble if args.n_iter_ensemble is not None else (10 if args.fast else 25)
    min_threshold = float(args.min_threshold) if args.production else 0.0

    log(
        "Starting training | "
        f"fast={args.fast} | production={args.production} | "
        f"min_threshold={min_threshold:.3f} | "
        f"n_iter_single={n_iter_single} | n_iter_ensemble={n_iter_ensemble}"
    )
    dataset_paths = [Path(p) for p in args.dataset_paths] if args.dataset_paths else DATASETS
    datasets = []
    for p in dataset_paths:
        if p.exists():
            df = pd.read_csv(p)
            # Final hygiene for training stability.
            df = df.drop_duplicates(subset=["lon", "lat", "label"]).dropna().reset_index(drop=True)
            datasets.append((p.stem, df))

    if not datasets:
        raise FileNotFoundError("No training dataset found.")

    inference_source = (
        Path(args.inference_feature_source)
        if args.inference_feature_source.strip()
        else (
            INFERENCE_FEATURE_SOURCE_AUG
            if INFERENCE_FEATURE_SOURCE_AUG.exists()
            else INFERENCE_FEATURE_SOURCE
        )
    )
    if not inference_source.exists():
        raise FileNotFoundError(
            f"Missing inference feature source: {INFERENCE_FEATURE_SOURCE_AUG} / {INFERENCE_FEATURE_SOURCE}"
        )
    deployable_columns = set(pd.read_csv(inference_source, nrows=5).columns)
    deployable_columns -= {"lon", "lat", "label"}

    configs = build_training_configs(datasets, deployable_columns)
    if not configs:
        raise RuntimeError("No deployable training configuration found.")

    log(f"Found {len(configs)} deployable training configs.")
    results = [
        train_dataset(df, name, feat_cols, n_iter_single, n_iter_ensemble, min_threshold)
        for name, df, feat_cols in configs
    ]
    best = max(results, key=lambda r: r["metrics"]["spatial_auc_mean"])
    best_df = best["df"]
    region_bbox = {
        "lon_min": float(best_df["lon"].min()),
        "lon_max": float(best_df["lon"].max()),
        "lat_min": float(best_df["lat"].min()),
        "lat_max": float(best_df["lat"].max()),
    }

    # Save deployable artifacts.
    # Keep first model as JSON for compatibility with legacy scoring scripts.
    best["models"][0].save_model(out_paths["model"])
    joblib.dump(best["models"], out_paths["bundle"])
    joblib.dump(best["calibrator"], out_paths["calibrator"])
    out_paths["features"].write_text(json.dumps(best["feature_cols"], indent=2), encoding="utf-8")

    pred_df = best["df"][["lon", "lat", "label"]].copy()
    pred_df["oof_raw"] = best["oof_raw_full"]
    pred_df["oof_calibrated"] = best["oof_cal_full"]
    pred_df["train_calibrated_fullfit"] = best["train_cal"]
    pred_df.to_csv(out_paths["pred"], index=False)

    report: dict[str, Any] = {
        "model_version": out_paths["version"],
        "model_name": "Gulf of Mannar Regional Ecological Screening Model",
        "selected_dataset": best["name"],
        "inference_feature_source": str(inference_source),
        "training_region": {
            "name": "Gulf of Mannar",
            "bbox": region_bbox,
        },
        "resolution": {
            "effective_resolution_km": 22.0,
            "coarsest_layer": "wave_height (~0.20 degree)",
            "coarsest_layer_resolution_km": 22.0,
            "secondary_resolution_km": 9.2,
        },
        "use_case": {
            "intended": "regional_screening",
            "not_valid_for": "micro-site (<5 km) decisions",
            "notes": "Model predicts ecological suitability, not farm installation decisions.",
        },
        "selected_metrics": best["metrics"],
        "all_datasets": [
            {"name": r["name"], **r["metrics"]}
            for r in results
        ],
        "deployment_policy": {
            "recommended_threshold": best["metrics"]["threshold"]["threshold"],
            "production_mode": bool(args.production),
            "min_threshold_floor": float(min_threshold),
            "high_confidence_cutoff": 0.80,
            "medium_confidence_cutoff": 0.60,
            "notes": "Use calibrated probabilities. Treat 0.60-0.80 as field-verification priority."
        },
    }
    REPORT_PATH_USED = out_paths["report"]
    REPORT_PATH_USED.write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_card_path = out_paths["model_card"]
    model_card = f"""# Model Card: {report['model_name']} ({MODEL_VERSION})

## Scope
- Region: {report['training_region']['name']}
- Bounding box: lon [{region_bbox['lon_min']:.6f}, {region_bbox['lon_max']:.6f}], lat [{region_bbox['lat_min']:.6f}, {region_bbox['lat_max']:.6f}]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: {best['name']}
- Samples: {best['metrics']['n_samples']}
- Positives: {best['metrics']['n_presence']}
- Negatives: {best['metrics']['n_background']}
- Features: {best['metrics']['n_features']}
- Spatial folds: {best['metrics']['n_spatial_folds']}

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: {best['metrics']['spatial_auc_mean']:.4f} +/- {best['metrics']['spatial_auc_std']:.4f}
- Spatial AP: {best['metrics']['spatial_ap_mean']:.4f}
- OOF calibrated AP: {best['metrics']['oof_ap_calibrated']:.4f}
- OOF calibrated Brier: {best['metrics']['oof_brier_calibrated']:.4f}

## Deployment Policy
- Threshold: {best['metrics']['threshold']['threshold']:.6f}
- Precision at threshold: {best['metrics']['threshold']['precision']:.4f}
- Recall at threshold: {best['metrics']['threshold']['recall']:.4f}
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited ({best['metrics']['n_samples']} rows, {best['metrics']['n_presence']} positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
"""
    model_card_path.write_text(model_card, encoding="utf-8")

    print(f"Saved model: {out_paths['model']}")
    print(f"Saved model bundle: {out_paths['bundle']}")
    print(f"Saved calibrator: {out_paths['calibrator']}")
    print(f"Saved features: {out_paths['features']}")
    print(f"Saved report: {REPORT_PATH_USED}")
    print(f"Saved model card: {model_card_path}")
    print(f"Saved predictions: {out_paths['pred']}")
    print(f"Selected dataset: {best['name']}")
    print(
        "Spatial AUC: {:.4f} +/- {:.4f}".format(
            best["metrics"]["spatial_auc_mean"], best["metrics"]["spatial_auc_std"]
        )
    )
    print(
        "OOF calibrated AP: {:.4f}, Brier: {:.4f}".format(
            best["metrics"]["oof_ap_calibrated"], best["metrics"]["oof_brier_calibrated"]
        )
    )
    print(
        "Recommended threshold: {:.3f} (precision {:.3f}, recall {:.3f})".format(
            best["metrics"]["threshold"]["threshold"],
            best["metrics"]["threshold"]["precision"],
            best["metrics"]["threshold"]["recall"],
        )
    )


if __name__ == "__main__":
    main()
