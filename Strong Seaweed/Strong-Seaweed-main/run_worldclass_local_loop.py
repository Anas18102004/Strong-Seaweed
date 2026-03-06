import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from project_paths import BASE, REPORTS_DIR, TABULAR_DIR, ensure_dirs


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=BASE)


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
            run(
                [
                    "python",
                    "download_copernicus_data.py",
                    "--username",
                    args.cmems_username,
                    "--password",
                    args.cmems_password,
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                ]
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

            run(
                [
                    "python",
                    "real_world_readiness_check.py",
                    "--report_json",
                    str(Path("releases") / run_tag / "reports" / f"xgboost_realtime_report_{run_tag}.json"),
                    "--training_csv",
                    str(TABULAR_DIR / f"training_dataset_{run_tag}.csv"),
                    "--output_json",
                    str(REPORTS_DIR / f"real_world_readiness_check_{run_tag}.json"),
                    "--output_md",
                    str(Path("docs") / f"REAL_WORLD_READINESS_{run_tag}.md"),
                ]
            )
            summary["steps"].append("kappa_readiness_gate")

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

