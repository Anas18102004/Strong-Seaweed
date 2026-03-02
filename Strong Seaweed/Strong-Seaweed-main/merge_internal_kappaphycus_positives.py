import argparse
import json
from pathlib import Path

import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs, EXPERIMENTS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge internal Kappaphycus positive sources into deduplicated ingestion files."
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=[
            str(TABULAR_DIR / "v1_1_data_plan.csv"),
            str(EXPERIMENTS_DIR / "v1_1_websweep_candidates_strict_ready.csv"),
            str(TABULAR_DIR / "v1.2_from_pack_auto_verified_from_pack.csv"),
        ],
    )
    p.add_argument(
        "--out_strict_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_internal_merged_strict.csv",
    )
    p.add_argument(
        "--out_broad_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_internal_merged_broad.csv",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "kappaphycus_internal_merge_report.json",
    )
    p.add_argument("--dedup_decimals", type=int, default=8)
    return p.parse_args()


def norm_bool(v) -> bool:
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def main() -> None:
    ensure_dirs()
    args = parse_args()

    frames = []
    source_stats = {}
    for s in args.sources:
        p = Path(s)
        if not p.exists():
            source_stats[str(p)] = {"exists": False, "rows": 0}
            continue
        df = pd.read_csv(p)
        source_stats[str(p)] = {"exists": True, "rows": int(len(df))}
        df["__source_file"] = p.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid source files found.")

    all_df = pd.concat(frames, ignore_index=True)

    # Standardize expected fields.
    for c in [
        "record_id",
        "source_type",
        "source_name",
        "source_reference",
        "citation_url",
        "species",
        "eventDate",
        "year",
        "lon",
        "lat",
        "label",
        "coordinate_precision_km",
        "species_confirmed",
        "confidence_score",
        "is_verified",
        "qa_reviewer",
        "qa_status",
        "rationale",
        "notes",
    ]:
        if c not in all_df.columns:
            all_df[c] = ""

    all_df["lon"] = pd.to_numeric(all_df["lon"], errors="coerce")
    all_df["lat"] = pd.to_numeric(all_df["lat"], errors="coerce")
    all_df["label"] = pd.to_numeric(all_df["label"], errors="coerce")
    all_df["year"] = pd.to_numeric(all_df["year"], errors="coerce")
    all_df["coordinate_precision_km"] = pd.to_numeric(all_df["coordinate_precision_km"], errors="coerce")
    all_df["confidence_score"] = pd.to_numeric(all_df["confidence_score"], errors="coerce")
    all_df["qa_status"] = all_df["qa_status"].astype(str).str.strip().str.lower()

    base = all_df.dropna(subset=["lon", "lat"]).copy()
    base = base[base["label"] == 1].copy()
    base = base[base["species"].astype(str).str.lower().str.contains("kappaphycus", na=False)].copy()

    # Broad: keep all kappaphycus positives.
    broad = base.copy()
    broad["lon_r"] = broad["lon"].round(args.dedup_decimals)
    broad["lat_r"] = broad["lat"].round(args.dedup_decimals)
    broad = broad.sort_values(
        ["confidence_score", "is_verified", "qa_status"], ascending=[False, False, True]
    )
    broad = broad.drop_duplicates(subset=["lon_r", "lat_r", "label"]).drop(columns=["lon_r", "lat_r"]).reset_index(drop=True)

    # Strict: approved + verified + species_confirmed.
    strict = base.copy()
    strict = strict[
        strict["qa_status"].isin({"approved", "verified", "accepted"})
        & strict["is_verified"].map(norm_bool)
        & strict["species_confirmed"].map(norm_bool)
    ].copy()
    strict["lon_r"] = strict["lon"].round(args.dedup_decimals)
    strict["lat_r"] = strict["lat"].round(args.dedup_decimals)
    strict = strict.sort_values(["confidence_score"], ascending=False)
    strict = strict.drop_duplicates(subset=["lon_r", "lat_r", "label"]).drop(columns=["lon_r", "lat_r"]).reset_index(drop=True)

    cols = [
        "record_id",
        "source_type",
        "source_name",
        "source_reference",
        "citation_url",
        "species",
        "eventDate",
        "year",
        "lon",
        "lat",
        "label",
        "coordinate_precision_km",
        "species_confirmed",
        "confidence_score",
        "is_verified",
        "qa_reviewer",
        "qa_status",
        "rationale",
        "notes",
    ]
    broad = broad[cols]
    strict = strict[cols]

    args.out_strict_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_broad_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    strict.to_csv(args.out_strict_csv, index=False)
    broad.to_csv(args.out_broad_csv, index=False)

    report = {
        "sources": source_stats,
        "rows_all_concat": int(len(all_df)),
        "rows_base_kappaphycus_positive": int(len(base)),
        "rows_broad_dedup": int(len(broad)),
        "rows_strict_dedup": int(len(strict)),
        "out_strict_csv": str(args.out_strict_csv),
        "out_broad_csv": str(args.out_broad_csv),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
