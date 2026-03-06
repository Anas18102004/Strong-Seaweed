import argparse
import json
from pathlib import Path

from project_paths import BASE, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote kappa release only if candidate beats incumbent quality gates.")
    p.add_argument("--candidate_tag", type=str, required=True)
    p.add_argument("--incumbent_tag", type=str, default="v1.1r_base46_cmp")
    p.add_argument("--require_readiness", action="store_true", help="Require candidate readiness decision != FAIL.")
    p.add_argument(
        "--readiness_json",
        type=Path,
        default=None,
        help="Optional readiness JSON path. If omitted, auto-resolves artifacts/reports/real_world_readiness_check_<candidate>.json",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=REPORTS_DIR / "active_kappa_release.json",
        help="Active release marker consumed by serve_species_api.py",
    )
    p.add_argument("--min_improvement", type=float, default=0.005, help="Minimum weighted score improvement to promote.")
    return p.parse_args()


def load_report(tag: str) -> dict:
    p = BASE / "releases" / tag / "reports" / f"xgboost_realtime_report_{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing report for tag={tag}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def selected_metrics(report: dict) -> dict:
    m = report.get("selected_metrics", {})
    return {
        "spatial_auc_mean": float(m.get("spatial_auc_mean", 0.0)),
        "oof_auc_calibrated": float(m.get("oof_auc_calibrated", 0.0)),
        "oof_ap_calibrated": float(m.get("oof_ap_calibrated", 0.0)),
        "oof_brier_calibrated": float(m.get("oof_brier_calibrated", 1.0)),
    }


def score(m: dict) -> float:
    # Higher is better: reward discrimination + calibration quality, penalize brier.
    return (
        0.40 * m["spatial_auc_mean"]
        + 0.30 * m["oof_auc_calibrated"]
        + 0.25 * m["oof_ap_calibrated"]
        - 0.05 * m["oof_brier_calibrated"]
    )


def main() -> None:
    args = parse_args()
    cand_report = load_report(args.candidate_tag)
    inc_report = load_report(args.incumbent_tag)
    cand = selected_metrics(cand_report)
    inc = selected_metrics(inc_report)
    cand_score = score(cand)
    inc_score = score(inc)

    readiness_decision = "UNKNOWN"
    readiness_path = args.readiness_json or (REPORTS_DIR / f"real_world_readiness_check_{args.candidate_tag}.json")
    if readiness_path.exists():
        try:
            readiness_decision = str(json.loads(readiness_path.read_text(encoding="utf-8")).get("decision", "UNKNOWN"))
        except Exception:
            readiness_decision = "UNKNOWN"

    promote = (cand_score - inc_score) >= float(args.min_improvement)
    if args.require_readiness and readiness_decision == "FAIL":
        promote = False

    result = {
        "candidate_tag": args.candidate_tag,
        "incumbent_tag": args.incumbent_tag,
        "candidate_metrics": cand,
        "incumbent_metrics": inc,
        "candidate_score": cand_score,
        "incumbent_score": inc_score,
        "score_delta": cand_score - inc_score,
        "readiness_decision": readiness_decision,
        "promoted": bool(promote),
        "release_tag": args.candidate_tag if promote else args.incumbent_tag,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

