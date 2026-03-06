import argparse
import os
import subprocess
import shutil
from pathlib import Path
from project_paths import BASE, TABULAR_DIR, OUTPUTS_DIR, ensure_dirs


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=BASE, env=env)


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
    p.add_argument(
        "--no_production",
        action="store_true",
        help="Disable production threshold constraints (not recommended).",
    )
    p.add_argument(
        "--min_precision_at_threshold",
        type=float,
        default=0.80,
        help="Minimum precision target used by train_realtime_production threshold selection.",
    )
    p.add_argument(
        "--min_recall_at_threshold",
        type=float,
        default=0.40,
        help="Minimum recall target used by train_realtime_production threshold selection.",
    )
    p.add_argument(
        "--min_threshold",
        type=float,
        default=0.50,
        help="Minimum deployment threshold in production mode.",
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
    base_master_csv = TABULAR_DIR / "master_feature_matrix.csv"
    base_presence_csv = TABULAR_DIR / "kappaphycus_presence_snapped_clean.csv"
    if not master_csv.exists():
        master_csv = base_master_csv
    if not presence_csv.exists():
        presence_csv = base_presence_csv
    if args.make_data_plan_template:
        run(["python", "ingest_presence_records.py", "--make_template"])
        print(f"Template created. Fill it and rerun with --data_plan_csv {TABULAR_DIR / 'v1_1_data_plan.csv'}")
        return

    if args.download_copernicus:
        if not args.cmems_username or not args.cmems_password:
            raise ValueError("Set --cmems_username/--cmems_password or env CMEMS_USERNAME/CMEMS_PASSWORD.")
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
        # Keep release-tagged naming stable for downstream steps if requested.
        if args.release_tag.strip():
            tagged_presence = TABULAR_DIR / f"kappaphycus_presence_snapped_clean_{args.release_tag.strip()}.csv"
            if base_presence_csv.exists() and not tagged_presence.exists():
                shutil.copyfile(base_presence_csv, tagged_presence)
                presence_csv = tagged_presence

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
    run(
        [
            "python",
            "build_site_prior_policy_layers.py",
            "--master_in",
            str(master_csv),
            "--train_in",
            str(training_csv),
            "--master_out",
            str(TABULAR_DIR / (f"master_feature_matrix{suffix}_augmented.csv" if suffix else "master_feature_matrix_augmented.csv")),
            "--train_out",
            str(TABULAR_DIR / (f"training_dataset{suffix}_augmented.csv" if suffix else "training_dataset_augmented.csv")),
            "--meta_out",
            str(TABULAR_DIR.parent / "reports" / (f"site_prior_policy_metadata{suffix}.json" if suffix else "site_prior_policy_metadata.json")),
            "--seasonal_out",
            str(TABULAR_DIR / (f"seasonal_operational_weights{suffix}.csv" if suffix else "seasonal_operational_weights.csv")),
        ]
    )
    training_aug_csv = TABULAR_DIR / (
        f"training_dataset_{args.release_tag.strip()}_augmented.csv"
        if args.release_tag.strip()
        else "training_dataset_augmented.csv"
    )
    inference_feature_source = TABULAR_DIR / (
        f"master_feature_matrix_{args.release_tag.strip()}_augmented.csv"
        if args.release_tag.strip()
        else "master_feature_matrix_augmented.csv"
    )
    base_training_aug_csv = TABULAR_DIR / "training_dataset_augmented.csv"
    base_inference_feature_source = TABULAR_DIR / "master_feature_matrix_augmented.csv"
    if args.release_tag.strip():
        # Backward-compatibility fallback for very old runs; should rarely trigger now.
        if not training_aug_csv.exists() and base_training_aug_csv.exists():
            shutil.copyfile(base_training_aug_csv, training_aug_csv)
        if not inference_feature_source.exists() and base_inference_feature_source.exists():
            shutil.copyfile(base_inference_feature_source, inference_feature_source)

    train_cmd = [
        "python",
        "train_realtime_production.py",
        "--min_precision_at_threshold",
        str(float(args.min_precision_at_threshold)),
        "--min_recall_at_threshold",
        str(float(args.min_recall_at_threshold)),
    ]
    if not args.no_production:
        train_cmd.extend(["--production", "--min_threshold", str(float(args.min_threshold))])
    if args.release_tag.strip():
        train_cmd.extend(
            [
                "--release_tag",
                args.release_tag.strip(),
                "--dataset_paths",
                str(training_csv),
                str(training_aug_csv),
                "--inference_feature_source",
                str(inference_feature_source),
            ]
        )
    run(train_cmd)

    score_input = TABULAR_DIR / "master_feature_matrix_augmented.csv"
    if args.release_tag.strip():
        # Prefer release-specific augmented matrix first, then release base matrix.
        candidate_aug = master_csv.with_name(f"{master_csv.stem}_augmented.csv")
        if inference_feature_source.exists():
            score_input = inference_feature_source
        elif candidate_aug.exists():
            score_input = candidate_aug
        elif master_csv.exists():
            score_input = master_csv

    score_cmd = [
        "python",
        "score_realtime_production.py",
        "--input",
        str(score_input),
        "--output",
        str(OUTPUTS_DIR / "realtime_ranked_candidates.csv"),
    ]
    if args.release_tag.strip():
        score_cmd.extend(["--release_tag", args.release_tag.strip(), "--snapshot-tag", f"{args.release_tag.strip().replace('.', '_')}_baseline"])
    run(score_cmd)
    print("Refresh complete.")


if __name__ == "__main__":
    main()
