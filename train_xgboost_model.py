import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from xgboost import XGBClassifier
from project_paths import TABULAR_DIR, REPORTS_DIR, DIAGNOSTICS_DIR, with_legacy, ensure_dirs

DATA_PATH = with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv")
MODEL_PATH = REPORTS_DIR / "xgboost_model.json"
METRICS_PATH = REPORTS_DIR / "xgboost_metrics.json"
IMPORTANCE_PATH = DIAGNOSTICS_DIR / "xgboost_feature_importance.csv"
PRED_PATH = DIAGNOSTICS_DIR / "training_predictions.csv"


RANDOM_STATE = 42


def make_spatial_groups(df: pd.DataFrame, n_lon_bins: int = 3, n_lat_bins: int = 2) -> np.ndarray:
    lon_bin = pd.qcut(df["lon"], q=n_lon_bins, labels=False, duplicates="drop")
    lat_bin = pd.qcut(df["lat"], q=n_lat_bins, labels=False, duplicates="drop")
    lon_bin = lon_bin.astype(int)
    lat_bin = lat_bin.astype(int)
    groups = (lat_bin * (int(lon_bin.max()) + 1) + lon_bin).to_numpy()
    return groups


def spatial_group_folds(groups: np.ndarray, y: np.ndarray, min_pos_test: int = 5):
    folds = []
    unique_groups = np.unique(groups)
    for g in unique_groups:
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if y[test_idx].sum() < min_pos_test:
            continue
        if y[train_idx].sum() == 0 or (len(y[train_idx]) - y[train_idx].sum()) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def evaluate_params_cv(X, y, folds, params):
    aucs = []
    for train_idx, test_idx in folds:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], pred)
        aucs.append(float(auc))

    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c not in ["label", "lon", "lat"]]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = neg / max(pos, 1)

    # Random CV (optimistic baseline)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    random_folds = list(skf.split(X, y))

    # Spatial CV (defensible)
    groups = make_spatial_groups(df, n_lon_bins=3, n_lat_bins=2)
    spatial_folds = spatial_group_folds(groups, y, min_pos_test=5)

    if len(spatial_folds) < 3:
        raise RuntimeError(f"Insufficient spatial folds with positives: {len(spatial_folds)}")

    param_space = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 2, 4, 6],
        "gamma": [0.0, 0.1, 0.3, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "scale_pos_weight": [scale_pos_weight],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    candidates = list(ParameterSampler(param_space, n_iter=40, random_state=rng))

    best = None
    leaderboard = []

    for p in candidates:
        rand_mean, rand_std, _ = evaluate_params_cv(X, y, random_folds, p)
        sp_mean, sp_std, sp_fold_aucs = evaluate_params_cv(X, y, spatial_folds, p)

        rec = {
            "params": p,
            "random_cv_auc_mean": rand_mean,
            "random_cv_auc_std": rand_std,
            "spatial_cv_auc_mean": sp_mean,
            "spatial_cv_auc_std": sp_std,
            "spatial_fold_aucs": sp_fold_aucs,
        }
        leaderboard.append(rec)

        if (best is None) or (sp_mean > best["spatial_cv_auc_mean"]):
            best = rec

    best_params = best["params"]

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X, y)

    train_pred = final_model.predict_proba(X)[:, 1]
    train_auc = float(roc_auc_score(y, train_pred))

    # Save outputs
    final_model.save_model(MODEL_PATH)

    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": final_model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(IMPORTANCE_PATH, index=False)

    pred_df = df[["lon", "lat", "label"]].copy()
    pred_df["pred_proba"] = train_pred
    pred_df.to_csv(PRED_PATH, index=False)

    top5 = imp.head(5).to_dict(orient="records")

    metrics = {
        "n_samples": int(len(df)),
        "n_presence": pos,
        "n_background": neg,
        "n_features": int(len(feature_cols)),
        "spatial_folds": int(len(spatial_folds)),
        "best_params": best_params,
        "best_spatial_cv_auc_mean": best["spatial_cv_auc_mean"],
        "best_spatial_cv_auc_std": best["spatial_cv_auc_std"],
        "best_random_cv_auc_mean": best["random_cv_auc_mean"],
        "best_random_cv_auc_std": best["random_cv_auc_std"],
        "best_spatial_fold_aucs": best["spatial_fold_aucs"],
        "train_auc_full_data": train_auc,
        "top5_features_by_gain": top5,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved feature importance:", IMPORTANCE_PATH)
    print("Saved training predictions:", PRED_PATH)
    print("Best spatial CV AUC: {:.4f} +/- {:.4f}".format(best["spatial_cv_auc_mean"], best["spatial_cv_auc_std"]))
    print("Best random CV AUC: {:.4f} +/- {:.4f}".format(best["random_cv_auc_mean"], best["random_cv_auc_std"]))
    print("Train AUC (full data): {:.4f}".format(train_auc))


if __name__ == "__main__":
    main()
