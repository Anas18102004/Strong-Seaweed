import argparse
import subprocess
from pathlib import Path

from project_paths import TABULAR_DIR


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run best internal-seed retrain pipeline.")
    p.add_argument(
        "--dataset_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_internal_broad_seed.csv",
    )
    p.add_argument("--release_tag", type=str, default="v1_3_internal_broad_seed")
    p.add_argument("--max_snap_m", type=float, default=1500.0)
    p.add_argument("--bg_ratio", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_csv.exists():
        raise FileNotFoundError(f"Missing dataset_csv: {args.dataset_csv}")

    # Train release artifacts
    run(
        [
            "python",
            "train_realtime_production.py",
            "--dataset_paths",
            str(args.dataset_csv),
            "--release_tag",
            args.release_tag,
        ]
    )

    # Score baseline snapshot
    run(
        [
            "python",
            "score_realtime_production.py",
            "--release_tag",
            args.release_tag,
            "--input",
            str(TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv"),
            "--output",
            str(Path("outputs") / f"realtime_ranked_candidates_{args.release_tag}.csv"),
            "--snapshot-tag",
            f"{args.release_tag}_baseline",
        ]
    )

    # Dataset-only readiness update against newly chosen training CSV.
    run(
        [
            "python",
            "real_world_readiness_check.py",
            "--training_csv",
            str(args.dataset_csv),
            "--output_json",
            str(Path("artifacts/reports") / f"real_world_readiness_check_{args.release_tag}.json"),
            "--output_md",
            str(Path("docs") / f"REAL_WORLD_READINESS_{args.release_tag}.md"),
        ]
    )


if __name__ == "__main__":
    main()
