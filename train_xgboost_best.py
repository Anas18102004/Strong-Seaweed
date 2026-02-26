import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from xgboost import XGBClassifier
from project_paths import TABULAR_DIR, REPORTS_DIR, DIAGNOSTICS_DIR, with_legacy, ensure_dirs

BASELINE_PATH = with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv")
AUG_PATH = with_legacy(TABULAR_DIR / "training_dataset_augmented.csv", "training_dataset_augmented.csv")

BEST_MODEL_PATH = REPORTS_DIR / "xgboost_best_model.json"
BEST_METRICS_PATH = REPORTS_DIR / "xgboost_best_metrics.json"
BEST_IMPORTANCE_PATH = DIAGNOSTICS_DIR / "xgboost_best_feature_importance.csv"
BEST_PRED_PATH = DIAGNOSTICS_DIR / "xgboost_best_training_predictions.csv"

RANDOM_STATE = 42


def make_spatial_groups(df: pd.DataFrame, n_lon_bins: int = 3, n_lat_bins: int = 2) -> np.ndarray:
    lon_bin = pd.qcut(df["lon"], q=n_lon_bins, labels=False, duplicates="drop").astype(int)
    lat_bin = pd.qcut(df["lat"], q=n_lat_bins, labels=False, duplicates="drop").astype(int)
    return (lat_bin * (int(lon_bin.max()) + 1) + lon_bin).to_numpy()


def spatial_group_folds(groups: np.ndarray, y: np.ndarray, min_pos_test: int = 5):
    folds = []
    for g in np.unique(groups):
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if y[test_idx].sum() < min_pos_test:
            continue
        if y[train_idx].sum() == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def eval_params(X, y, folds, params):
    aucs = []
    for tr, te in folds:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], pred)))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def run_search(df: pd.DataFrame, dataset_name: str, n_iter: int) -> dict:
    drop_cols = {"label", "lon", "lat"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Remove constant/near-constant columns to improve robustness.
    keep = []
    for c in feature_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique > 1:
            keep.append(c)
    feature_cols = keep

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / max(pos, 1)

    random_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y))
    groups = make_spatial_groups(df)
    spatial_folds = spatial_group_folds(groups, y, min_pos_test=5)
    if len(spatial_folds) < 3:
        raise RuntimeError(f"{dataset_name}: insufficient spatial folds ({len(spatial_folds)})")

    param_space = {
        "n_estimators": [150, 200, 300, 500, 700],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 2, 4, 6, 8],
        "gamma": [0.0, 0.1, 0.3, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "scale_pos_weight": [spw],
    }

    candidates = list(ParameterSampler(param_space, n_iter=n_iter, random_state=RANDOM_STATE))

    best = None
    total = len(candidates)
    print(f"[{dataset_name}] starting search with {total} candidates...", flush=True)
    for i, p in enumerate(candidates, start=1):
        rand_mean, rand_std, _ = eval_params(X, y, random_folds, p)
        sp_mean, sp_std, sp_aucs = eval_params(X, y, spatial_folds, p)
        rec = {
            "params": p,
            "random_cv_auc_mean": rand_mean,
            "random_cv_auc_std": rand_std,
            "spatial_cv_auc_mean": sp_mean,
            "spatial_cv_auc_std": sp_std,
            "spatial_fold_aucs": sp_aucs,
        }
        if (best is None) or (sp_mean > best["spatial_cv_auc_mean"]):
            best = rec
            print(
                f"[{dataset_name}] {i}/{total} new best spatial AUC={sp_mean:.4f} "
                f"(random={rand_mean:.4f})",
                flush=True,
            )
        elif i % 5 == 0:
            print(f"[{dataset_name}] {i}/{total} checked...", flush=True)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best["params"],
    )
    model.fit(X, y)
    train_pred = model.predict_proba(X)[:, 1]
    train_auc = float(roc_auc_score(y, train_pred))

    return {
        "dataset_name": dataset_name,
        "df": df,
        "feature_cols": feature_cols,
        "model": model,
        "train_pred": train_pred,
        "train_auc": train_auc,
        "n_samples": int(len(df)),
        "n_presence": int(pos),
        "n_background": int(neg),
        "n_features": int(len(feature_cols)),
        "spatial_folds": int(len(spatial_folds)),
        **best,
    }


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=40, help="hyperparameter samples per dataset")
    args = parser.parse_args()

    if not BASELINE_PATH.exists():
        raise FileNotFoundError("training_dataset.csv not found.")

    baseline_df = pd.read_csv(BASELINE_PATH)
    results = [run_search(baseline_df, "baseline", args.n_iter)]

    if AUG_PATH.exists():
        aug_df = pd.read_csv(AUG_PATH)
        results.append(run_search(aug_df, "augmented", args.n_iter))

    best_result = max(results, key=lambda r: r["spatial_cv_auc_mean"])

    best_model = best_result["model"]
    best_df = best_result["df"]
    best_features = best_result["feature_cols"]

    best_model.save_model(BEST_MODEL_PATH)

    imp = pd.DataFrame(
        {"feature": best_features, "importance_gain": best_model.feature_importances_}
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(BEST_IMPORTANCE_PATH, index=False)

    pred_df = best_df[["lon", "lat", "label"]].copy()
    pred_df["pred_proba"] = best_result["train_pred"]
    pred_df.to_csv(BEST_PRED_PATH, index=False)

    summary = {
        "selected_dataset": best_result["dataset_name"],
        "selected_spatial_cv_auc_mean": best_result["spatial_cv_auc_mean"],
        "selected_spatial_cv_auc_std": best_result["spatial_cv_auc_std"],
        "selected_random_cv_auc_mean": best_result["random_cv_auc_mean"],
        "selected_random_cv_auc_std": best_result["random_cv_auc_std"],
        "selected_train_auc_full_data": best_result["train_auc"],
        "selected_params": best_result["params"],
        "selected_top5_features": imp.head(5).to_dict(orient="records"),
        "all_results": [
            {
                "dataset_name": r["dataset_name"],
                "spatial_cv_auc_mean": r["spatial_cv_auc_mean"],
                "spatial_cv_auc_std": r["spatial_cv_auc_std"],
                "random_cv_auc_mean": r["random_cv_auc_mean"],
                "random_cv_auc_std": r["random_cv_auc_std"],
                "train_auc_full_data": r["train_auc"],
                "n_features": r["n_features"],
            }
            for r in results
        ],
    }

    with open(BEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {BEST_MODEL_PATH}")
    print(f"Saved: {BEST_METRICS_PATH}")
    print(f"Saved: {BEST_IMPORTANCE_PATH}")
    print(f"Saved: {BEST_PRED_PATH}")
    print("Selected dataset:", summary["selected_dataset"])
    print(
        "Selected spatial CV AUC: {:.4f} +/- {:.4f}".format(
            summary["selected_spatial_cv_auc_mean"],
            summary["selected_spatial_cv_auc_std"],
        )
    )
    print(
        "Selected random CV AUC: {:.4f} +/- {:.4f}".format(
            summary["selected_random_cv_auc_mean"],
            summary["selected_random_cv_auc_std"],
        )
    )
    print(
        "Selected train AUC (full data): {:.4f}".format(
            summary["selected_train_auc_full_data"]
        )
    )


if __name__ == "__main__":
    main()
