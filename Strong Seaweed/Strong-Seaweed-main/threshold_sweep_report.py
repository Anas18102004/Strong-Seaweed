import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import REPORTS_DIR, EXPERIMENTS_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep thresholds on OOF and hard-test sets.")
    p.add_argument(
        "--oof_csv",
        type=Path,
        default=Path("releases/v1_1/reports/xgboost_realtime_oof_predictions_v1_1.csv"),
    )
    p.add_argument(
        "--hard_csv",
        type=Path,
        default=EXPERIMENTS_DIR / "v1_1_hard_test_30pts_web_in_domain_scored.csv",
    )
    p.add_argument("--min_precision", type=float, default=0.80)
    p.add_argument("--min_recall", type=float, default=0.70)
    p.add_argument(
        "--out_json",
        type=Path,
        default=REPORTS_DIR / "v1_1_threshold_sweep_report.json",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=REPORTS_DIR / "v1_1_threshold_sweep_table.csv",
    )
    return p.parse_args()


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "threshold": float(thr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def select_recommended(df: pd.DataFrame, min_precision: float, min_recall: float) -> pd.Series:
    feasible = df[(df["hard_precision"] >= min_precision) & (df["hard_recall"] >= min_recall)]
    if not feasible.empty:
        return feasible.sort_values(
            ["hard_f1", "oof_f1", "hard_recall", "hard_precision"],
            ascending=False,
        ).iloc[0]

    # Fallback: maximize hard-test recall under precision floor; if none, maximize hard-test F1.
    floor = df[df["hard_precision"] >= min_precision]
    if not floor.empty:
        return floor.sort_values(["hard_recall", "hard_f1", "oof_f1"], ascending=False).iloc[0]
    return df.sort_values(["hard_f1", "oof_f1"], ascending=False).iloc[0]


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.oof_csv.exists():
        raise FileNotFoundError(f"Missing OOF CSV: {args.oof_csv}")
    if not args.hard_csv.exists():
        raise FileNotFoundError(f"Missing hard-test CSV: {args.hard_csv}")

    oof = pd.read_csv(args.oof_csv)
    hard = pd.read_csv(args.hard_csv)

    oof = oof.dropna(subset=["label", "oof_calibrated"]).copy()
    if "expected" not in hard.columns or "p_calibrated" not in hard.columns:
        raise ValueError("hard_csv must contain expected and p_calibrated columns.")

    y_oof = oof["label"].to_numpy(dtype=int)
    s_oof = oof["oof_calibrated"].to_numpy(dtype=float)
    y_h = hard["expected"].to_numpy(dtype=int)
    s_h = hard["p_calibrated"].to_numpy(dtype=float)

    # Candidate thresholds: fixed grid + unique score values.
    thrs = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 201),
                np.round(s_oof, 6),
                np.round(s_h, 6),
            ]
        )
    )
    thrs = thrs[(thrs >= 0.0) & (thrs <= 1.0)]
    rows = []
    for t in thrs:
        mo = metrics_at_threshold(y_oof, s_oof, float(t))
        mh = metrics_at_threshold(y_h, s_h, float(t))
        rows.append(
            {
                "threshold": float(t),
                "oof_precision": mo["precision"],
                "oof_recall": mo["recall"],
                "oof_f1": mo["f1"],
                "oof_specificity": mo["specificity"],
                "hard_precision": mh["precision"],
                "hard_recall": mh["recall"],
                "hard_f1": mh["f1"],
                "hard_specificity": mh["specificity"],
                "hard_accuracy": mh["accuracy"],
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    rec = select_recommended(table, args.min_precision, args.min_recall)

    report = {
        "inputs": {
            "oof_csv": str(args.oof_csv),
            "hard_csv": str(args.hard_csv),
            "n_oof": int(len(oof)),
            "n_hard": int(len(hard)),
            "min_precision_target": float(args.min_precision),
            "min_recall_target": float(args.min_recall),
        },
        "recommended_threshold": float(rec["threshold"]),
        "recommended_metrics": {
            "oof_precision": float(rec["oof_precision"]),
            "oof_recall": float(rec["oof_recall"]),
            "oof_f1": float(rec["oof_f1"]),
            "hard_precision": float(rec["hard_precision"]),
            "hard_recall": float(rec["hard_recall"]),
            "hard_f1": float(rec["hard_f1"]),
            "hard_specificity": float(rec["hard_specificity"]),
            "hard_accuracy": float(rec["hard_accuracy"]),
        },
        "best_hard_f1": table.sort_values("hard_f1", ascending=False).iloc[0].to_dict(),
        "best_oof_f1": table.sort_values("oof_f1", ascending=False).iloc[0].to_dict(),
        "note": "Recommendation prioritizes hard-test feasibility under precision/recall targets, then F1.",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table.to_csv(args.out_csv, index=False)

    print(f"Saved sweep report: {args.out_json}")
    print(f"Saved sweep table:  {args.out_csv}")
    print(
        "Recommended threshold {:.3f} | hard P/R/F1 {:.3f}/{:.3f}/{:.3f} | OOF P/R/F1 {:.3f}/{:.3f}/{:.3f}".format(
            report["recommended_threshold"],
            report["recommended_metrics"]["hard_precision"],
            report["recommended_metrics"]["hard_recall"],
            report["recommended_metrics"]["hard_f1"],
            report["recommended_metrics"]["oof_precision"],
            report["recommended_metrics"]["oof_recall"],
            report["recommended_metrics"]["oof_f1"],
        )
    )


if __name__ == "__main__":
    main()

