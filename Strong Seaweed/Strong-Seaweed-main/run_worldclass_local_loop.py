import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from project_paths import BASE, REPORTS_DIR, TABULAR_DIR, ensure_dirs


def run(cmd: list[str], check: bool = True, env: dict[str, str] | None = None) -> int:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, cwd=BASE, env=env)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local iterative improvement loop (weak-supervision + retrain + strict gates)."
    )
    p.add_argument("--release_tag", type=str, default="worldclass_loop_v1")
    p.add_argument("--download_copernicus", action="store_true")
    p.add_argument("--cmems_username", type=str, default=os.getenv("CMEMS_USERNAME", ""))
    p.add_argument("--cmems_password", type=str, default=os.getenv("CMEMS_PASSWORD", ""))
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2026-03-01")
    p.add_argument("--bg_ratio", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kappa_max_snap_m", type=float, default=1500.0)
    p.add_argument("--kappa_bg_ratio", type=int, default=6)
    p.add_argument("--skip_kappa_refresh", action="store_true")
    p.add_argument("--skip_multispecies_refresh", action="store_true")
    p.add_argument("--incumbent_kappa_tag", type=str, default="v1.1r_base46_cmp")
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.release_tag}_{ts}"

    summary: dict[str, object] = {
        "run_tag": run_tag,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "steps": [],
    }

    try:
        # 1) Optional Copernicus refresh.
        if args.download_copernicus:
            if not args.cmems_username or not args.cmems_password:
                raise ValueError(
                    "download_copernicus requested but CMEMS credentials are missing."
                )
            dl_env = os.environ.copy()
            dl_env["CMEMS_USERNAME"] = args.cmems_username
            dl_env["CMEMS_PASSWORD"] = args.cmems_password
            run(
                [
                    "python",
                    "download_copernicus_data.py",
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                ],
                env=dl_env,
            )
            summary["steps"].append("copernicus_download")

        # 2) Kappaphycus weak-supervision refresh (external labels + retrain).
        if not args.skip_kappa_refresh:
            run(
                [
                    "python",
                    "run_realtime_refresh.py",
                    "--fetch_external_labels",
                    "--max_snap_m",
                    str(args.kappa_max_snap_m),
                    "--bg_ratio",
                    str(args.kappa_bg_ratio),
                    "--release_tag",
                    run_tag,
                ]
            )
            summary["steps"].append("kappa_refresh_retrain")

            # Generate candidate-specific hard50 and non-overlap eval artifacts.
            run(
                [
                    "python",
                    "generate_release_hard_eval.py",
                    "--release_tag",
                    run_tag,
                    "--training_csv",
                    str(TABULAR_DIR / f"training_dataset_{run_tag}.csv"),
                ]
            )
            summary["steps"].append("kappa_candidate_eval")

            readiness_json = REPORTS_DIR / f"real_world_readiness_check_{run_tag}.json"
            rc = run(
                [
                    "python",
                    "real_world_readiness_check.py",
                    "--report_json",
                    str(Path("releases") / run_tag / "reports" / f"xgboost_realtime_report_{run_tag}.json"),
                    "--training_csv",
                    str(TABULAR_DIR / f"training_dataset_{run_tag}.csv"),
                    "--hard50_json",
                    str(REPORTS_DIR / f"{run_tag}_hard50_eval.json"),
                    "--independent_json",
                    str(REPORTS_DIR / f"{run_tag}_hard50_eval_nonoverlap.json"),
                    "--oof_csv",
                    str(Path("releases") / run_tag / "reports" / f"xgboost_realtime_oof_predictions_{run_tag}.csv"),
                    "--output_json",
                    str(readiness_json),
                    "--output_md",
                    str(Path("docs") / f"REAL_WORLD_READINESS_{run_tag}.md"),
                ]
                ,
                check=False,
            )
            decision = "UNKNOWN"
            if readiness_json.exists():
                try:
                    decision = json.loads(readiness_json.read_text(encoding="utf-8")).get("decision", "UNKNOWN")
                except Exception:
                    decision = "UNKNOWN"
            summary["kappa_readiness"] = {"decision": decision, "exit_code": rc}
            summary["steps"].append("kappa_readiness_gate")

            # Promote only when candidate score improves incumbent; optional readiness guard.
            run(
                [
                    "python",
                    "promote_kappa_release.py",
                    "--candidate_tag",
                    run_tag,
                    "--incumbent_tag",
                    args.incumbent_kappa_tag,
                    "--require_readiness",
                    "--readiness_json",
                    str(readiness_json),
                ],
                check=False,
            )
            summary["steps"].append("kappa_promotion_check")

        # 3) Multispecies weak-supervision refresh (rich Copernicus features + strict geographic gate).
        if not args.skip_multispecies_refresh:
            out_prefix = f"{run_tag}_cop_india_rich"
            release_multi = f"{run_tag}_multi"

            run(
                [
                    "python",
                    "build_multispecies_copernicus_rich_point_datasets.py",
                    "--out_prefix",
                    out_prefix,
                    "--bg_ratio",
                    str(args.bg_ratio),
                    "--seed",
                    str(args.seed),
                ]
            )
            summary["steps"].append("multispecies_dataset_refresh")

            run(
                [
                    "python",
                    "strict_geographic_holdout_gate_multispecies.py",
                    "--dataset_glob",
                    f"training_dataset_*_{out_prefix}.csv",
                    "--out_tag",
                    out_prefix,
                ]
            )
            summary["steps"].append("multispecies_strict_gate")

            run(
                [
                    "python",
                    "train_multispecies_models.py",
                    "--dataset_glob",
                    f"training_dataset_*_{out_prefix}.csv",
                    "--release_tag",
                    release_multi,
                ]
            )
            summary["steps"].append("multispecies_train")

        summary["status"] = "ok"
    except Exception as e:
        summary["status"] = "failed"
        summary["error"] = str(e)
        raise
    finally:
        summary["finished_utc"] = datetime.utcnow().isoformat() + "Z"
        out = REPORTS_DIR / f"worldclass_local_loop_{run_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SUMMARY] {out}")


if __name__ == "__main__":
    main()
