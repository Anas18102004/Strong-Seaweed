import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Kappaphycus seed web data into project training schema + reproducible splits."
    )
    p.add_argument(
        "--input_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_complex_global_seed_dataset_v1.csv",
    )
    p.add_argument(
        "--output_schema_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_seed_training_schema_v1.csv",
    )
    p.add_argument(
        "--output_split_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_seed_training_schema_v1_splits.csv",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "kappaphycus_seed_training_schema_v1_report.json",
    )
    p.add_argument("--train_pct", type=float, default=0.70)
    p.add_argument("--val_pct", type=float, default=0.15)
    p.add_argument("--test_pct", type=float, default=0.15)
    p.add_argument(
        "--region_filter_master",
        type=Path,
        default=TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv",
        help="Optional: if exists, adds in_region_current_model flag using this bbox.",
    )
    return p.parse_args()


def choose_split_from_cell(cell_key: str, train_pct: float, val_pct: float) -> str:
    h = hashlib.md5(cell_key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < train_pct:
        return "train"
    if v < train_pct + val_pct:
        return "val"
    return "test"


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")

    if not np.isclose(args.train_pct + args.val_pct + args.test_pct, 1.0, atol=1e-6):
        raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

    raw = pd.read_csv(args.input_csv)
    needed = {"lon", "lat", "label", "scientificName"}
    miss = sorted(list(needed - set(raw.columns)))
    if miss:
        raise ValueError(f"Input missing required columns: {miss}")

    df = raw.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    df = df.drop_duplicates(subset=["lon", "lat", "label", "scientificName"]).reset_index(drop=True)

    out = pd.DataFrame()
    out["record_id"] = [f"kseed_{i+1:05d}" for i in range(len(df))]
    out["source_type"] = "web_api_occurrence"
    out["source_name"] = df.get("source", "OBIS").astype(str).replace({"": "OBIS"})
    out["source_reference"] = df["scientificName"].astype(str)
    out["citation_url"] = df.get("source_url", "").astype(str)
    out["species"] = df["scientificName"].astype(str)
    out["eventDate"] = df.get("eventDate", "").astype(str)
    out["year"] = pd.to_numeric(df.get("year", np.nan), errors="coerce")
    out["lon"] = df["lon"].astype(float)
    out["lat"] = df["lat"].astype(float)
    out["label"] = df["label"].astype(int)
    out["coordinate_precision_km"] = 1.0
    out["species_confirmed"] = np.where(
        out["label"] == 1,
        out["species"].str.lower().str.contains("kappaphycus", na=False),
        False,
    )
    out["confidence_score"] = np.where(out["label"] == 1, 0.80, 0.70)
    out["is_verified"] = False
    out["qa_reviewer"] = "auto_seed_pipeline"
    out["qa_status"] = "pending"
    out["rationale"] = np.where(
        out["label"] == 1,
        "kappaphycus_web_occurrence_positive_seed",
        "non_target_macroalgae_negative_candidate_seed",
    )
    out["notes"] = "seed_dataset=v1; requires manual QA before strict production use"

    # Reproducible spatial split by coarse 2-degree cells.
    lon_cell = np.floor(out["lon"] / 2.0).astype(int)
    lat_cell = np.floor(out["lat"] / 2.0).astype(int)
    cell_key = lon_cell.astype(str) + "_" + lat_cell.astype(str)
    out["split"] = [
        choose_split_from_cell(k, float(args.train_pct), float(args.val_pct))
        for k in cell_key
    ]

    # Ensure each class has enough rows in each split when possible.
    for lab in [0, 1]:
        sub = out[out["label"] == lab]
        n_lab = len(sub)
        if n_lab < 3:
            continue
        # For minority classes, enforce minimally useful val/test support.
        min_val = max(1, int(round(0.10 * n_lab)))
        min_test = max(1, int(round(0.10 * n_lab)))

        def _count(split_name: str) -> int:
            return int(((out["label"] == lab) & (out["split"] == split_name)).sum())

        # Fill val from train first.
        while _count("val") < min_val:
            donor = out[(out["label"] == lab) & (out["split"] == "train")]
            if donor.empty:
                donor = out[(out["label"] == lab) & (out["split"] == "test")]
            if donor.empty:
                break
            out.loc[donor.index[0], "split"] = "val"

        # Fill test from train first.
        while _count("test") < min_test:
            donor = out[(out["label"] == lab) & (out["split"] == "train")]
            if donor.empty:
                donor = out[(out["label"] == lab) & (out["split"] == "val")]
            if donor.empty:
                break
            out.loc[donor.index[0], "split"] = "test"

    out = out.sort_values(["label", "split", "record_id"], ascending=[False, True, True]).reset_index(drop=True)

    in_region_flag = np.zeros(len(out), dtype=bool)
    if args.region_filter_master.exists():
        master = pd.read_csv(args.region_filter_master, usecols=["lon", "lat"])
        lo_min, lo_max = float(master["lon"].min()), float(master["lon"].max())
        la_min, la_max = float(master["lat"].min()), float(master["lat"].max())
        in_region_flag = (
            (out["lon"] >= lo_min)
            & (out["lon"] <= lo_max)
            & (out["lat"] >= la_min)
            & (out["lat"] <= la_max)
        ).to_numpy()
        out["in_region_current_model"] = in_region_flag
    else:
        out["in_region_current_model"] = False

    args.output_schema_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_split_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output_schema_csv, index=False)
    out.to_csv(args.output_split_csv, index=False)

    split_counts = (
        out.groupby(["split", "label"]).size().rename("n").reset_index()
    )
    split_counts_map = {}
    for _, r in split_counts.iterrows():
        split_counts_map[f"{r['split']}_label{int(r['label'])}"] = int(r["n"])

    report = {
        "input_csv": str(args.input_csv),
        "output_schema_csv": str(args.output_schema_csv),
        "output_split_csv": str(args.output_split_csv),
        "n_rows": int(len(out)),
        "n_pos": int((out["label"] == 1).sum()),
        "n_neg": int((out["label"] == 0).sum()),
        "split_counts": split_counts_map,
        "in_region_current_model_rows": int(in_region_flag.sum()),
        "in_region_current_model_pos": int(((out["label"] == 1) & (out["in_region_current_model"])).sum()),
        "in_region_current_model_neg": int(((out["label"] == 0) & (out["in_region_current_model"])).sum()),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved schema CSV: {args.output_schema_csv}")
    print(f"Saved split CSV: {args.output_split_csv}")
    print(f"Saved report JSON: {args.report_json}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
