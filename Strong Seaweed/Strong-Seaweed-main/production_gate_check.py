import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fail/pass release based on production readiness gates."
    )
    p.add_argument(
        "--report_json",
        type=Path,
        required=True,
        help="Release training report JSON (xgboost_realtime_report_*.json).",
    )
    p.add_argument(
        "--benchmark_json",
        type=Path,
        required=True,
        help="External benchmark summary JSON.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=Path("artifacts/reports/production_gate_check.json"),
        help="Where to write gate decision details.",
    )
    p.add_argument("--min_spatial_auc", type=float, default=0.75)
    p.add_argument("--max_spatial_auc_std", type=float, default=0.08)
    p.add_argument("--min_oof_ap_calibrated", type=float, default=0.50)
    p.add_argument("--min_benchmark_precision", type=float, default=0.80)
    p.add_argument("--min_benchmark_recall", type=float, default=0.80)
    p.add_argument("--max_benchmark_gt5km_snaps", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.report_json.exists():
        raise FileNotFoundError(f"Missing report_json: {args.report_json}")
    if not args.benchmark_json.exists():
        raise FileNotFoundError(f"Missing benchmark_json: {args.benchmark_json}")

    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    bench = json.loads(args.benchmark_json.read_text(encoding="utf-8"))

    m = report["selected_metrics"]
    checks = {
        "spatial_auc_mean": float(m["spatial_auc_mean"]) >= float(args.min_spatial_auc),
        "spatial_auc_std": float(m["spatial_auc_std"]) <= float(args.max_spatial_auc_std),
        "oof_ap_calibrated": float(m["oof_ap_calibrated"]) >= float(args.min_oof_ap_calibrated),
        "benchmark_precision": float(bench.get("precision", 0.0)) >= float(args.min_benchmark_precision),
        "benchmark_recall": float(bench.get("recall", 0.0)) >= float(args.min_benchmark_recall),
        "benchmark_gt5km_snaps": int(bench.get("gt5km_snaps", 0)) <= int(args.max_benchmark_gt5km_snaps),
    }
    passed = all(checks.values())
    failed = [k for k, ok in checks.items() if not ok]

    decision = {
        "passed": passed,
        "failed_checks": failed,
        "inputs": {
            "report_json": str(args.report_json),
            "benchmark_json": str(args.benchmark_json),
        },
        "thresholds": {
            "min_spatial_auc": float(args.min_spatial_auc),
            "max_spatial_auc_std": float(args.max_spatial_auc_std),
            "min_oof_ap_calibrated": float(args.min_oof_ap_calibrated),
            "min_benchmark_precision": float(args.min_benchmark_precision),
            "min_benchmark_recall": float(args.min_benchmark_recall),
            "max_benchmark_gt5km_snaps": int(args.max_benchmark_gt5km_snaps),
        },
        "metrics": {
            "spatial_auc_mean": float(m["spatial_auc_mean"]),
            "spatial_auc_std": float(m["spatial_auc_std"]),
            "oof_ap_calibrated": float(m["oof_ap_calibrated"]),
            "benchmark_precision": float(bench.get("precision", 0.0)),
            "benchmark_recall": float(bench.get("recall", 0.0)),
            "benchmark_gt5km_snaps": int(bench.get("gt5km_snaps", 0)),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Saved gate decision: {args.output_json}")
    print(f"PASSED: {passed}")
    if failed:
        print("FAILED CHECKS:", ", ".join(failed))

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
