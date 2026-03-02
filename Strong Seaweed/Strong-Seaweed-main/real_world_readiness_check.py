import argparse
import json
from pathlib import Path

import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command real-world readiness check for dataset + model + external evaluation."
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "xgboost_realtime_report.json",
        help="Training report JSON (active or release report).",
    )
    p.add_argument(
        "--training_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_plus11web_plus35provisional.csv",
        help="Training dataset CSV used by the selected model.",
    )
    p.add_argument(
        "--hard50_json",
        type=Path,
        default=REPORTS_DIR / "v1.1r_plus35provisional_hard50_eval.json",
        help="Hard benchmark JSON (balanced stress test).",
    )
    p.add_argument(
        "--independent_json",
        type=Path,
        default=REPORTS_DIR / "v1.1r_plus35provisional_hard50_eval_nonoverlap.json",
        help="Independent non-overlap benchmark JSON.",
    )
    p.add_argument(
        "--oof_csv",
        type=Path,
        default=Path("releases/v1.1r_plus35provisional/reports/xgboost_realtime_oof_predictions_v1.1r_plus35provisional.csv"),
        help="OOF predictions CSV for missing-OOF sanity checks.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=REPORTS_DIR / "real_world_readiness_check.json",
        help="Output decision JSON.",
    )
    p.add_argument(
        "--output_md",
        type=Path,
        default=Path("docs/REAL_WORLD_READINESS.md"),
        help="Human-readable markdown summary.",
    )
    p.add_argument("--min_train_spatial_auc", type=float, default=0.70)
    p.add_argument("--min_train_oof_ap", type=float, default=0.45)
    p.add_argument("--min_hard50_auc", type=float, default=0.85)
    p.add_argument("--min_hard50_precision", type=float, default=0.80)
    p.add_argument("--min_hard50_recall", type=float, default=0.40)
    p.add_argument("--min_independent_n", type=int, default=40)
    p.add_argument("--min_independent_pos", type=int, default=20)
    p.add_argument("--min_independent_neg", type=int, default=20)
    p.add_argument("--min_independent_auc", type=float, default=0.75)
    p.add_argument("--min_independent_precision", type=float, default=0.70)
    p.add_argument("--min_independent_recall", type=float, default=0.60)
    p.add_argument("--max_label_conflict_ratio", type=float, default=0.02)
    p.add_argument("--max_oof_missing_ratio", type=float, default=0.15)
    return p.parse_args()


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def dataset_checks(training_csv: Path) -> dict:
    df = pd.read_csv(training_csv)
    n = int(len(df))
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    bad_label = int(n - pos - neg)
    missing_cells = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())
    dup_lon_lat_label = int(df.duplicated(subset=["lon", "lat", "label"]).sum())
    by_cell = df.groupby(["lon", "lat"])["label"].nunique(dropna=True).reset_index(name="n_labels")
    conflict_points = int((by_cell["n_labels"] > 1).sum())
    conflict_ratio = float(conflict_points / n) if n else 1.0
    return {
        "n_rows": n,
        "n_pos": pos,
        "n_neg": neg,
        "bad_label_rows": bad_label,
        "missing_cells": missing_cells,
        "rows_with_missing": rows_with_missing,
        "duplicate_lon_lat_label_rows": dup_lon_lat_label,
        "conflict_lon_lat_points": conflict_points,
        "conflict_ratio": conflict_ratio,
    }


def oof_checks(oof_csv: Path) -> dict:
    if not oof_csv.exists():
        return {
            "oof_csv_found": False,
            "oof_rows": 0,
            "oof_missing_rows": 0,
            "oof_missing_ratio": 1.0,
        }
    df = pd.read_csv(oof_csv)
    miss = int(df["oof_calibrated"].isna().sum())
    n = int(len(df))
    return {
        "oof_csv_found": True,
        "oof_rows": n,
        "oof_missing_rows": miss,
        "oof_missing_ratio": float(miss / n) if n else 1.0,
    }


def as_float(d: dict, key: str) -> float:
    return float(d.get(key, 0.0))


def as_int(d: dict, key: str) -> int:
    return int(d.get(key, 0))


def main() -> None:
    ensure_dirs()
    args = parse_args()
    require_file(args.report_json, "report_json")
    require_file(args.training_csv, "training_csv")
    require_file(args.hard50_json, "hard50_json")
    require_file(args.independent_json, "independent_json")

    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    hard = json.loads(args.hard50_json.read_text(encoding="utf-8"))
    indep = json.loads(args.independent_json.read_text(encoding="utf-8"))
    ds = dataset_checks(args.training_csv)
    oof = oof_checks(args.oof_csv)
    m = report.get("selected_metrics", {})

    checks = [
        ("train_spatial_auc", as_float(m, "spatial_auc_mean") >= args.min_train_spatial_auc),
        ("train_oof_ap_calibrated", as_float(m, "oof_ap_calibrated") >= args.min_train_oof_ap),
        ("hard50_auc", as_float(hard, "roc_auc") >= args.min_hard50_auc),
        ("hard50_precision", as_float(hard, "precision") >= args.min_hard50_precision),
        ("hard50_recall", as_float(hard, "recall") >= args.min_hard50_recall),
        ("independent_n", as_int(indep, "n") >= args.min_independent_n),
        ("independent_pos", as_int(indep, "positives") >= args.min_independent_pos),
        ("independent_neg", as_int(indep, "negatives") >= args.min_independent_neg),
        ("independent_auc", as_float(indep, "roc_auc") >= args.min_independent_auc),
        ("independent_precision", as_float(indep, "precision") >= args.min_independent_precision),
        ("independent_recall", as_float(indep, "recall") >= args.min_independent_recall),
        ("dataset_bad_labels", ds["bad_label_rows"] == 0),
        ("dataset_missing_cells", ds["missing_cells"] == 0),
        ("dataset_no_dup_lon_lat_label", ds["duplicate_lon_lat_label_rows"] == 0),
        ("dataset_conflict_ratio", ds["conflict_ratio"] <= args.max_label_conflict_ratio),
        ("oof_missing_ratio", oof["oof_missing_ratio"] <= args.max_oof_missing_ratio),
    ]
    failed = [name for name, ok in checks if not ok]
    hard_checks_ok = all(ok for _, ok in checks[:5])
    independent_ok = all(ok for _, ok in checks[5:11])
    data_ok = all(ok for _, ok in checks[11:])

    if hard_checks_ok and independent_ok and data_ok:
        decision = "PASS"
    elif hard_checks_ok and data_ok:
        decision = "WARN"
    else:
        decision = "FAIL"

    out = {
        "decision": decision,
        "model_version": report.get("model_version", ""),
        "inputs": {
            "report_json": str(args.report_json),
            "training_csv": str(args.training_csv),
            "hard50_json": str(args.hard50_json),
            "independent_json": str(args.independent_json),
            "oof_csv": str(args.oof_csv),
        },
        "thresholds": {
            "min_train_spatial_auc": args.min_train_spatial_auc,
            "min_train_oof_ap": args.min_train_oof_ap,
            "min_hard50_auc": args.min_hard50_auc,
            "min_hard50_precision": args.min_hard50_precision,
            "min_hard50_recall": args.min_hard50_recall,
            "min_independent_n": args.min_independent_n,
            "min_independent_pos": args.min_independent_pos,
            "min_independent_neg": args.min_independent_neg,
            "min_independent_auc": args.min_independent_auc,
            "min_independent_precision": args.min_independent_precision,
            "min_independent_recall": args.min_independent_recall,
            "max_label_conflict_ratio": args.max_label_conflict_ratio,
            "max_oof_missing_ratio": args.max_oof_missing_ratio,
        },
        "metrics": {
            "train_spatial_auc": as_float(m, "spatial_auc_mean"),
            "train_oof_ap_calibrated": as_float(m, "oof_ap_calibrated"),
            "hard50_auc": as_float(hard, "roc_auc"),
            "hard50_precision": as_float(hard, "precision"),
            "hard50_recall": as_float(hard, "recall"),
            "independent_n": as_int(indep, "n"),
            "independent_pos": as_int(indep, "positives"),
            "independent_neg": as_int(indep, "negatives"),
            "independent_auc": as_float(indep, "roc_auc"),
            "independent_precision": as_float(indep, "precision"),
            "independent_recall": as_float(indep, "recall"),
            "dataset": ds,
            "oof": oof,
        },
        "checks": [{"name": name, "pass": ok} for name, ok in checks],
        "failed_checks": failed,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Real-World Readiness Check",
        "",
        f"- Decision: **{decision}**",
        f"- Model version: `{out['model_version']}`",
        "",
        "## Key metrics",
        f"- Train spatial AUC: {out['metrics']['train_spatial_auc']:.4f}",
        f"- Train OOF AP (cal): {out['metrics']['train_oof_ap_calibrated']:.4f}",
        f"- Hard50 AUC / P / R: {out['metrics']['hard50_auc']:.4f} / {out['metrics']['hard50_precision']:.4f} / {out['metrics']['hard50_recall']:.4f}",
        f"- Independent n / pos / neg: {out['metrics']['independent_n']} / {out['metrics']['independent_pos']} / {out['metrics']['independent_neg']}",
        f"- Independent AUC / P / R: {out['metrics']['independent_auc']:.4f} / {out['metrics']['independent_precision']:.4f} / {out['metrics']['independent_recall']:.4f}",
        "",
        "## Dataset QA",
        f"- Rows: {ds['n_rows']} | Pos: {ds['n_pos']} | Neg: {ds['n_neg']}",
        f"- Label conflicts (same lon/lat, mixed labels): {ds['conflict_lon_lat_points']} (ratio {ds['conflict_ratio']:.4f})",
        f"- Missing cells: {ds['missing_cells']} | Rows with missing: {ds['rows_with_missing']}",
        f"- Duplicate lon/lat/label rows: {ds['duplicate_lon_lat_label_rows']}",
        "",
        "## OOF QA",
        f"- OOF file found: {oof['oof_csv_found']}",
        f"- OOF missing rows: {oof['oof_missing_rows']} / {oof['oof_rows']} (ratio {oof['oof_missing_ratio']:.4f})",
        "",
        "## Checks",
    ]
    for item in out["checks"]:
        lines.append(f"- {'PASS' if item['pass'] else 'FAIL'} | {item['name']}")
    if failed:
        lines += ["", "## Failed checks"]
        for name in failed:
            lines.append(f"- {name}")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.output_md}")
    print(f"Decision: {decision}")
    if failed:
        print("Failed checks:", ", ".join(failed))

    if decision == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
