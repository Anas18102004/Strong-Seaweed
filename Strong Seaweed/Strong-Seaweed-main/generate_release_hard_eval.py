import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate candidate-specific hard50 and non-overlap evaluation artifacts.")
    p.add_argument("--release_tag", type=str, required=True)
    p.add_argument(
        "--hard_test_csv",
        type=Path,
        default=Path("artifacts/experiments/v1_1_hard_test_50_web_with_features.csv"),
    )
    p.add_argument("--training_csv", type=Path, default=None, help="Defaults to data/tabular/training_dataset_<release>.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.release_tag.strip()
    base = Path("releases") / tag
    models_dir = base / "models"
    reports_dir = base / "reports"
    feat_json = models_dir / f"xgboost_realtime_features_{tag}.json"
    ens_pkl = models_dir / f"xgboost_realtime_ensemble_{tag}.pkl"
    cal_pkl = models_dir / f"xgboost_realtime_calibrator_{tag}.pkl"
    rep_json = reports_dir / f"xgboost_realtime_report_{tag}.json"
    if not (feat_json.exists() and ens_pkl.exists() and cal_pkl.exists() and rep_json.exists()):
        raise FileNotFoundError(f"Missing release artifacts for {tag}")
    if not args.hard_test_csv.exists():
        raise FileNotFoundError(f"Missing hard test csv: {args.hard_test_csv}")

    training_csv = args.training_csv or Path(f"data/tabular/training_dataset_{tag}.csv")
    if not training_csv.exists():
        raise FileNotFoundError(f"Missing training csv: {training_csv}")

    feat = json.loads(feat_json.read_text(encoding="utf-8"))
    rep = json.loads(rep_json.read_text(encoding="utf-8"))
    thr = float(rep["deployment_policy"]["recommended_threshold"])
    models = joblib.load(ens_pkl)
    cal = joblib.load(cal_pkl)

    hard_df = pd.read_csv(args.hard_test_csv)
    X = hard_df[feat].to_numpy(dtype=np.float32)
    raw = np.mean(np.vstack([m.predict_proba(X)[:, 1] for m in models]), axis=0)
    p = np.clip(cal.predict(raw), 0.0, 1.0)
    y = hard_df["expected"].astype(int).to_numpy()
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    hard_out = {
        "release": tag,
        "n": int(len(hard_df)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "threshold": thr,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    hard_df = hard_df.copy()
    hard_df["p_raw"] = raw
    hard_df["p_calibrated"] = p
    hard_df["pred_label"] = pred
    eval_csv = Path(f"artifacts/experiments/{tag}_hard50_eval.csv")
    eval_json = Path(f"artifacts/reports/{tag}_hard50_eval.json")
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    hard_df.to_csv(eval_csv, index=False)
    eval_json.write_text(json.dumps(hard_out, indent=2), encoding="utf-8")

    tr = pd.read_csv(training_csv)
    train_cells = set(zip(tr["lon"].round(8), tr["lat"].round(8)))
    hard_df["k"] = list(zip(hard_df["snap_lon"].round(8), hard_df["snap_lat"].round(8)))
    non = hard_df[~hard_df["k"].isin(train_cells)].copy()
    if len(non) > 0 and non["expected"].nunique() == 2:
        yy = non["expected"].astype(int).to_numpy()
        pp = np.clip(non["p_calibrated"].astype(float).to_numpy(), 0.0, 1.0)
        pr = (pp >= thr).astype(int)
        ncm = confusion_matrix(yy, pr, labels=[0, 1])
        non_out = {
            "release": tag,
            "n": int(len(non)),
            "positives": int((yy == 1).sum()),
            "negatives": int((yy == 0).sum()),
            "threshold": thr,
            "accuracy": float(accuracy_score(yy, pr)),
            "precision": float(precision_score(yy, pr, zero_division=0)),
            "recall": float(recall_score(yy, pr, zero_division=0)),
            "f1": float(f1_score(yy, pr, zero_division=0)),
            "roc_auc": float(roc_auc_score(yy, pp)),
            "avg_precision": float(average_precision_score(yy, pp)),
            "confusion_matrix": {
                "tn": int(ncm[0, 0]),
                "fp": int(ncm[0, 1]),
                "fn": int(ncm[1, 0]),
                "tp": int(ncm[1, 1]),
            },
        }
    else:
        non_out = {
            "release": tag,
            "n": int(len(non)),
            "positives": int((non["expected"] == 1).sum()) if len(non) else 0,
            "negatives": int((non["expected"] == 0).sum()) if len(non) else 0,
            "note": "insufficient class balance",
        }

    non_csv = Path(f"artifacts/experiments/{tag}_hard50_eval_nonoverlap.csv")
    non_json = Path(f"artifacts/reports/{tag}_hard50_eval_nonoverlap.json")
    non_csv.parent.mkdir(parents=True, exist_ok=True)
    non_json.parent.mkdir(parents=True, exist_ok=True)
    non.to_csv(non_csv, index=False)
    non_json.write_text(json.dumps(non_out, indent=2), encoding="utf-8")

    print(json.dumps({"hard50": hard_out, "nonoverlap": non_out}, indent=2))


if __name__ == "__main__":
    main()

