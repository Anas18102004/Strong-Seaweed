import argparse
import os
import subprocess
from pathlib import Path
from project_paths import BASE, TABULAR_DIR, OUTPUTS_DIR, ensure_dirs


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=BASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command refresh for real-time seaweed model.")
    p.add_argument(
        "--presence_inputs",
        nargs="*",
        default=[],
        help="Optional raw presence CSV files to ingest before training.",
    )
    p.add_argument("--download_copernicus", action="store_true")
    p.add_argument("--fetch_external_labels", action="store_true")
    p.add_argument("--cmems_username", type=str, default=os.getenv("CMEMS_USERNAME", ""))
    p.add_argument("--cmems_password", type=str, default=os.getenv("CMEMS_PASSWORD", ""))
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--max_snap_m", type=float, default=1000.0)
    p.add_argument("--bg_ratio", type=int, default=5)
    p.add_argument(
        "--data_plan_csv",
        type=str,
        default="",
        help="Optional v1.1 data-plan CSV to ingest as verified positives.",
    )
    p.add_argument(
        "--strict_data_plan",
        action="store_true",
        help="Apply strict v1.1 acceptance rules when ingesting --data_plan_csv.",
    )
    p.add_argument(
        "--make_data_plan_template",
        action="store_true",
        help="Create v1_1_data_plan.csv template and exit.",
    )
    p.add_argument(
        "--release_tag",
        type=str,
        default="",
        help="Optional release tag (e.g., v1.1) for separate train/score artifacts.",
    )
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""
    master_csv = TABULAR_DIR / (f"master_feature_matrix{suffix}.csv" if suffix else "master_feature_matrix.csv")
    training_csv = TABULAR_DIR / (f"training_dataset{suffix}.csv" if suffix else "training_dataset.csv")
    presence_csv = TABULAR_DIR / (f"kappaphycus_presence_snapped_clean{suffix}.csv" if suffix else "kappaphycus_presence_snapped_clean.csv")
    presence_report = TABULAR_DIR.parent / "reports" / (f"presence_ingestion_report{suffix}.json" if suffix else "presence_ingestion_report.json")
    if args.make_data_plan_template:
        run(["python", "ingest_presence_records.py", "--make_template"])
        print(f"Template created. Fill it and rerun with --data_plan_csv {TABULAR_DIR / 'v1_1_data_plan.csv'}")
        return

    if args.download_copernicus:
        if not args.cmems_username or not args.cmems_password:
            raise ValueError("Set --cmems_username/--cmems_password or env CMEMS_USERNAME/CMEMS_PASSWORD.")
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
        run(["python", "process_copernicus_ocean_features.py"])
        run(["python", "build_feature_matrix.py"])

    if args.presence_inputs:
        run(
            [
                "python",
                "ingest_presence_records.py",
                "--inputs",
                *args.presence_inputs,
                "--max_snap_m",
                str(args.max_snap_m),
            ]
        )

    if args.data_plan_csv:
        cmd = [
            "python",
            "ingest_presence_records.py",
            "--inputs",
            args.data_plan_csv,
            "--max_snap_m",
            str(args.max_snap_m),
            "--require_verified",
            "--species_filter",
            "kappaphycus",
            "--master_csv",
            str(master_csv),
            "--training_csv",
            str(training_csv),
            "--out_csv",
            str(presence_csv),
            "--out_report",
            str(presence_report),
        ]
        if args.strict_data_plan:
            cmd.append("--strict_acceptance")
        run(cmd)

    if args.fetch_external_labels:
        run(["python", "fetch_external_presence_labels.py"])
        run(
            [
                "python",
                "ingest_presence_records.py",
                "--inputs",
                str(TABULAR_DIR / "kappaphycus_presence_snapped_clean.csv"),
                str(TABULAR_DIR / "external_presence_candidates.csv"),
                "--max_snap_m",
                str(args.max_snap_m),
            ]
        )

    run(
        [
            "python",
            "build_training_dataset.py",
            "--max_snap_m",
            str(args.max_snap_m),
            "--bg_ratio",
            str(args.bg_ratio),
            "--feature_csv",
            str(master_csv),
            "--presence_csv",
            str(presence_csv),
            "--output",
            str(training_csv),
        ]
    )
    run(["python", "build_site_prior_policy_layers.py"])
    train_cmd = ["python", "train_realtime_production.py"]
    if args.release_tag.strip():
        train_cmd.extend(["--release_tag", args.release_tag.strip()])
    run(train_cmd)

    score_cmd = [
        "python",
        "score_realtime_production.py",
        "--input",
        str(TABULAR_DIR / "master_feature_matrix_augmented.csv"),
        "--output",
        str(OUTPUTS_DIR / "realtime_ranked_candidates.csv"),
    ]
    if args.release_tag.strip():
        score_cmd.extend(["--release_tag", args.release_tag.strip(), "--snapshot-tag", f"{args.release_tag.strip().replace('.', '_')}_baseline"])
    run(score_cmd)
    print("Refresh complete.")


if __name__ == "__main__":
    main()
