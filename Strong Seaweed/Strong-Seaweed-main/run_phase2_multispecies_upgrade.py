import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from project_paths import BASE, REPORTS_DIR, ensure_dirs


PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)


def run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, cwd=BASE, env=env)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-2 multispecies accuracy upgrade: verified labels + region-balanced negatives + gates."
    )
    p.add_argument("--tag_prefix", type=str, default="multi_species_cop_india_v7_phase2")
    p.add_argument("--bg_ratio", type=int, default=8)
    p.add_argument("--min_neg_abs", type=int, default=2200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard_neg_out_suffix", type=str, default="phase2hn")
    p.add_argument("--hard_neg_max_add_per_species", type=int, default=400)
    p.add_argument("--min_precision", type=float, default=0.72)
    p.add_argument("--min_recall", type=float, default=0.35)
    p.add_argument("--min_threshold", type=float, default=0.45)
    p.add_argument("--max_threshold", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_prefix = f"{args.tag_prefix}_{ts}_rich"
    release_tag = f"{args.tag_prefix}_{ts}"

    occ_csv = Path("data/tabular") / f"multispecies_occurrences_{release_tag}.csv"
    occ_report = REPORTS_DIR / f"multispecies_occurrences_{release_tag}.json"
    ds_report = REPORTS_DIR / f"multispecies_cop_rich_point_dataset_report_{out_prefix}.json"
    strict_gate_json = REPORTS_DIR / f"strict_geo_holdout_gate_{release_tag}.json"
    local_gate_json = REPORTS_DIR / f"local_species_gate_50_{release_tag}.json"
    summary_json = REPORTS_DIR / f"phase2_multispecies_upgrade_{release_tag}.json"

    run(
        [
            PYTHON_BIN,
            "build_multispecies_occurrences_with_anchors.py",
            "--out_csv",
            str(occ_csv),
            "--report_json",
            str(occ_report),
            "--max_records_per_source",
            "5000",
            "--min_year",
            "2000",
        ]
    )

    run(
        [
            PYTHON_BIN,
            "build_multispecies_copernicus_rich_point_datasets.py",
            "--occ_csv",
            str(occ_csv),
            "--bg_ratio",
            str(args.bg_ratio),
            "--min_neg_abs",
            str(args.min_neg_abs),
            "--seed",
            str(args.seed),
            "--negative_sampling_method",
            "region_balanced",
            "--neg_grid_deg",
            "2.0",
            "--max_neg_per_cell",
            "220",
            "--out_prefix",
            out_prefix,
        ]
    )

    # Carry forward known hard negatives for better decision boundaries.
    run(
        [
            PYTHON_BIN,
            "build_hard_negatives_multispecies.py",
            "--dataset_glob",
            f"training_dataset_*_{out_prefix}.csv",
            "--max_add_per_species",
            str(args.hard_neg_max_add_per_species),
            "--out_suffix",
            args.hard_neg_out_suffix,
        ]
    )

    final_glob = f"training_dataset_*_{out_prefix}_{args.hard_neg_out_suffix}.csv"

    strict_gate_exit = run(
        [
            PYTHON_BIN,
            "strict_geographic_holdout_gate_multispecies.py",
            "--dataset_glob",
            final_glob,
            "--out_tag",
            release_tag,
        ],
        check=False,
    )

    run(
        [
            PYTHON_BIN,
            "train_multispecies_models.py",
            "--dataset_glob",
            final_glob,
            "--release_tag",
            release_tag,
            "--calibration_method",
            "sigmoid",
            "--min_precision",
            str(args.min_precision),
            "--min_recall",
            str(args.min_recall),
            "--min_threshold",
            str(args.min_threshold),
            "--max_threshold",
            str(args.max_threshold),
        ]
    )

    gate_env = os.environ.copy()
    gate_env["MULTI_RELEASE"] = release_tag
    run(
        [
            PYTHON_BIN,
            "evaluate_local_species_gate_50.py",
            "--out_json",
            str(local_gate_json),
        ],
        env=gate_env,
    )

    summary = {
        "release_tag": release_tag,
        "out_prefix": out_prefix,
        "occ_csv": str(occ_csv),
        "reports": {
            "occurrence_report": str(occ_report),
            "dataset_report": str(ds_report),
            "strict_gate_report": str(strict_gate_json),
            "local_gate_report": str(local_gate_json),
        },
        "strict_gate_exit_code": strict_gate_exit,
        "notes": [
            "Set MULTI_RELEASE to this release tag in model-api env before deployment.",
            "Use strict_geo_holdout and local_species_gate_50 metrics as deployment gates.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Summary: {summary_json}")


if __name__ == "__main__":
    main()
