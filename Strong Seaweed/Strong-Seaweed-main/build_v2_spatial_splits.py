import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import TABULAR_DIR, REPORTS_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v2 train/val/test splits using country holdout or spatial hashing.")
    p.add_argument(
        "--input_csv",
        type=Path,
        default=TABULAR_DIR / "v2_unified_dataset_template.csv",
    )
    p.add_argument(
        "--species_config_json",
        type=Path,
        default=Path("data/config/v2_species_configs.json"),
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=TABULAR_DIR / "v2_unified_dataset_with_splits.csv",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "v2_spatial_split_report.json",
    )
    p.add_argument("--grid_deg", type=float, default=1.0)
    p.add_argument("--fallback_train_pct", type=float, default=0.70)
    p.add_argument("--fallback_val_pct", type=float, default=0.15)
    p.add_argument("--fallback_test_pct", type=float, default=0.15)
    return p.parse_args()


def hash_to_unit(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_spatial_hash_split(df: pd.DataFrame, grid_deg: float, tr: float, va: float) -> pd.Series:
    lon_cell = np.floor(df["lon"] / grid_deg).astype(int)
    lat_cell = np.floor(df["lat"] / grid_deg).astype(int)
    split = []
    for a, b in zip(lon_cell, lat_cell):
        u = hash_to_unit(f"{a}_{b}")
        if u < tr:
            split.append("train")
        elif u < tr + va:
            split.append("val")
        else:
            split.append("test")
    return pd.Series(split, index=df.index, dtype="object")


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")
    if not args.species_config_json.exists():
        raise FileNotFoundError(f"Missing species_config_json: {args.species_config_json}")

    cfg = json.loads(args.species_config_json.read_text(encoding="utf-8"))
    split_cfg = cfg.get("split_strategy", {})
    train_countries = set(split_cfg.get("train_countries", []))
    val_countries = set(split_cfg.get("val_countries", []))
    test_countries = set(split_cfg.get("test_countries", []))
    recommended_mode = split_cfg.get("recommended_mode", "country_holdout")

    df = pd.read_csv(args.input_csv)
    for c in ["lon", "lat", "species", "label"]:
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c}")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "country" not in df.columns:
        df["country"] = ""
    df["country"] = df["country"].fillna("").astype(str).str.strip()

    out = df.copy()
    out["split_mode"] = recommended_mode
    out["split"] = ""

    if recommended_mode == "country_holdout" and out["country"].str.len().gt(0).any():
        out.loc[out["country"].isin(test_countries), "split"] = "test"
        out.loc[out["country"].isin(val_countries), "split"] = "val"
        out.loc[out["country"].isin(train_countries), "split"] = "train"

        # Any country not configured falls back to spatial hashing.
        rem = out["split"] == ""
        out.loc[rem, "split"] = assign_spatial_hash_split(
            out.loc[rem],
            args.grid_deg,
            args.fallback_train_pct,
            args.fallback_val_pct,
        ).values
    else:
        out["split_mode"] = "spatial_block_hash"
        out["split"] = assign_spatial_hash_split(
            out,
            args.grid_deg,
            args.fallback_train_pct,
            args.fallback_val_pct,
        ).values

    # Minimal per-species guardrail: each species should appear in val/test when enough rows exist.
    for sp, sub in out.groupby("species"):
        idx = sub.index
        if len(sub) < 10:
            continue
        for need in ["val", "test"]:
            if (sub["split"] == need).any():
                continue
            donor = sub[sub["split"] == "train"]
            if donor.empty:
                continue
            out.loc[donor.index[0], "split"] = need

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    split_counts = out.groupby(["species", "split", "label"]).size().rename("n").reset_index()
    report = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "mode": out["split_mode"].iloc[0] if len(out) else recommended_mode,
        "n_rows": int(len(out)),
        "grid_deg": float(args.grid_deg),
        "split_counts": split_counts.to_dict(orient="records"),
        "country_counts": out["country"].value_counts(dropna=False).to_dict(),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.report_json}")
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
