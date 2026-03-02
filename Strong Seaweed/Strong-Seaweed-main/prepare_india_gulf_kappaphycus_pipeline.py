import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare India/Gulf-focused Kappaphycus seed data and optionally run Copernicus feature pipeline."
    )
    p.add_argument(
        "--seed_schema_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_seed_training_schema_v1_splits.csv",
    )
    p.add_argument(
        "--master_csv",
        type=Path,
        default=TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv",
    )
    p.add_argument("--india_min_lon", type=float, default=68.0)
    p.add_argument("--india_max_lon", type=float, default=98.5)
    p.add_argument("--india_min_lat", type=float, default=5.0)
    p.add_argument("--india_max_lat", type=float, default=24.5)
    p.add_argument("--gulf_buffer_deg", type=float, default=0.20)
    p.add_argument(
        "--out_india_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_seed_india_focus_v1.csv",
    )
    p.add_argument(
        "--out_gulf_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_seed_gulf_focus_v1.csv",
    )
    p.add_argument(
        "--out_gulf_presence_plan_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_gulf_presence_plan_v1.csv",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "kappaphycus_india_gulf_prepare_report.json",
    )
    p.add_argument(
        "--run_copernicus_flow",
        action="store_true",
        help="If set, run download/process/build-feature/ingest/build-training sequentially.",
    )
    p.add_argument(
        "--skip_copernicus_download",
        action="store_true",
        help="If set with --run_copernicus_flow, skip CMEMS download and use existing local NetCDF files.",
    )
    p.add_argument("--cmems_username", type=str, default="")
    p.add_argument("--cmems_password", type=str, default="")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--release_tag", type=str, default="kappa_india_gulf_v1")
    p.add_argument("--max_snap_m", type=float, default=1500.0)
    p.add_argument("--bg_ratio", type=int, default=5)
    p.add_argument(
        "--existing_presence_csv",
        type=Path,
        default=TABULAR_DIR / "kappaphycus_presence_snapped_clean_internal_broad.csv",
        help="Optional existing snapped presence CSV. If present, points inside Gulf bbox are used directly and ingestion is skipped.",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401

        return True
    except Exception:
        return False


def build_presence_plan(gulf_df: pd.DataFrame) -> pd.DataFrame:
    pos = gulf_df[gulf_df["label"] == 1].copy().reset_index(drop=True)
    out = pd.DataFrame()
    out["record_id"] = [f"kseed_gulf_{i+1:04d}" for i in range(len(pos))]
    out["source_type"] = "web_api_occurrence"
    out["source_name"] = pos["source_name"].astype(str)
    out["source_reference"] = pos["source_reference"].astype(str)
    out["citation_url"] = pos["citation_url"].astype(str)
    out["species"] = pos["species"].astype(str)
    out["eventDate"] = pos["eventDate"].astype(str)
    out["year"] = pd.to_numeric(pos["year"], errors="coerce")
    out["lon"] = pd.to_numeric(pos["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(pos["lat"], errors="coerce")
    out["label"] = 1
    out["coordinate_precision_km"] = pd.to_numeric(
        pos.get("coordinate_precision_km", 1.0), errors="coerce"
    ).fillna(1.0)
    out["species_confirmed"] = (
        out["species"].astype(str).str.lower().str.contains("kappaphycus", na=False)
    )
    out["confidence_score"] = pd.to_numeric(
        pos.get("confidence_score", 0.8), errors="coerce"
    ).fillna(0.8)
    out["is_verified"] = False
    out["qa_reviewer"] = "auto_seed_pipeline"
    out["qa_status"] = "pending"
    out["rationale"] = "india_gulf_filtered_seed_positive"
    out["notes"] = "from_kappaphycus_seed_training_schema_v1; split=" + pos["split"].astype(str)
    out = out.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return out


def build_presence_from_existing(
    existing_presence_csv: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    if not existing_presence_csv.exists():
        return pd.DataFrame(columns=["lon", "lat"])
    df = pd.read_csv(existing_presence_csv)
    if not {"lon", "lat"}.issubset(df.columns):
        return pd.DataFrame(columns=["lon", "lat"])
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"]).copy()
    df = df[
        (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
        & (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
    ].copy()
    keep_cols = [c for c in ["lon", "lat", "year", "source_file"] if c in df.columns]
    if "lon" not in keep_cols:
        keep_cols.append("lon")
    if "lat" not in keep_cols:
        keep_cols.append("lat")
    return df[keep_cols].drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.seed_schema_csv.exists():
        raise FileNotFoundError(f"Missing seed_schema_csv: {args.seed_schema_csv}")
    if not args.master_csv.exists():
        raise FileNotFoundError(f"Missing master_csv: {args.master_csv}")

    seed = pd.read_csv(args.seed_schema_csv)
    for c in ["lon", "lat", "label"]:
        if c not in seed.columns:
            raise ValueError(f"seed_schema_csv missing required column: {c}")

    seed["lon"] = pd.to_numeric(seed["lon"], errors="coerce")
    seed["lat"] = pd.to_numeric(seed["lat"], errors="coerce")
    seed = seed.dropna(subset=["lon", "lat"]).copy()

    india = seed[
        (seed["lon"] >= float(args.india_min_lon))
        & (seed["lon"] <= float(args.india_max_lon))
        & (seed["lat"] >= float(args.india_min_lat))
        & (seed["lat"] <= float(args.india_max_lat))
    ].copy()

    master = pd.read_csv(args.master_csv, usecols=["lon", "lat"])
    g_lon_min = float(master["lon"].min()) - float(args.gulf_buffer_deg)
    g_lon_max = float(master["lon"].max()) + float(args.gulf_buffer_deg)
    g_lat_min = float(master["lat"].min()) - float(args.gulf_buffer_deg)
    g_lat_max = float(master["lat"].max()) + float(args.gulf_buffer_deg)

    gulf = india[
        (india["lon"] >= g_lon_min)
        & (india["lon"] <= g_lon_max)
        & (india["lat"] >= g_lat_min)
        & (india["lat"] <= g_lat_max)
    ].copy()

    presence_plan = build_presence_plan(gulf)

    args.out_india_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_gulf_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_gulf_presence_plan_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    india.to_csv(args.out_india_csv, index=False)
    gulf.to_csv(args.out_gulf_csv, index=False)
    presence_plan.to_csv(args.out_gulf_presence_plan_csv, index=False)

    suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""
    master_tagged = TABULAR_DIR / (f"master_feature_matrix{suffix}.csv" if suffix else "master_feature_matrix.csv")
    presence_tagged = TABULAR_DIR / (f"kappaphycus_presence_snapped_clean{suffix}.csv" if suffix else "kappaphycus_presence_snapped_clean.csv")
    training_tagged = TABULAR_DIR / (f"training_dataset{suffix}.csv" if suffix else "training_dataset.csv")
    presence_report = REPORTS_DIR / (f"presence_ingestion_report{suffix}.json" if suffix else "presence_ingestion_report.json")

    report = {
        "inputs": {
            "seed_schema_csv": str(args.seed_schema_csv),
            "master_csv": str(args.master_csv),
        },
        "filters": {
            "india_bbox": {
                "lon_min": float(args.india_min_lon),
                "lon_max": float(args.india_max_lon),
                "lat_min": float(args.india_min_lat),
                "lat_max": float(args.india_max_lat),
            },
            "gulf_bbox_with_buffer": {
                "lon_min": g_lon_min,
                "lon_max": g_lon_max,
                "lat_min": g_lat_min,
                "lat_max": g_lat_max,
                "buffer_deg": float(args.gulf_buffer_deg),
            },
        },
        "counts": {
            "seed_rows": int(len(seed)),
            "india_rows": int(len(india)),
            "india_pos": int((india["label"] == 1).sum()),
            "india_neg": int((india["label"] == 0).sum()),
            "gulf_rows": int(len(gulf)),
            "gulf_pos": int((gulf["label"] == 1).sum()),
            "gulf_neg": int((gulf["label"] == 0).sum()),
            "gulf_presence_plan_rows": int(len(presence_plan)),
        },
        "outputs": {
            "india_csv": str(args.out_india_csv),
            "gulf_csv": str(args.out_gulf_csv),
            "gulf_presence_plan_csv": str(args.out_gulf_presence_plan_csv),
            "master_tagged_csv": str(master_tagged),
            "presence_tagged_csv": str(presence_tagged),
            "training_tagged_csv": str(training_tagged),
            "presence_report_json": str(presence_report),
        },
        "copernicus_flow_ran": bool(args.run_copernicus_flow),
        "existing_presence_csv": str(args.existing_presence_csv),
    }

    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_india_csv}")
    print(f"Saved: {args.out_gulf_csv}")
    print(f"Saved: {args.out_gulf_presence_plan_csv}")
    print(f"Saved: {args.report_json}")
    print(json.dumps(report["counts"], indent=2))

    if not args.run_copernicus_flow:
        return

    if args.skip_copernicus_download:
        print("Skipping Copernicus download and using existing local NetCDF files.")
    else:
        if not args.cmems_username or not args.cmems_password:
            raise ValueError(
                "run_copernicus_flow requires --cmems_username and --cmems_password unless --skip_copernicus_download is set"
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
                "--min_lon",
                str(g_lon_min),
                "--max_lon",
                str(g_lon_max),
                "--min_lat",
                str(g_lat_min),
                "--max_lat",
                str(g_lat_max),
            ]
        )
    if has_rasterio():
        run(
            [
                "python",
                "process_copernicus_ocean_features.py",
                "--release_tag",
                args.release_tag,
                "--depth_tif",
                "data/rasters/Depth_v1_1.tif",
            ]
        )
        run(
            [
                "python",
                "build_feature_matrix.py",
                "--release_tag",
                args.release_tag,
                "--depth_raster",
                "Depth_v1_1.tif",
                "--slope_raster",
                "Slope_v1_1.tif",
                "--distance_raster",
                "DistanceToShore_v1_1.tif",
                "--shallow_raster",
                "ShallowSuitabilityMask_v1_1.tif",
            ]
        )
    else:
        print(
            "rasterio not available; skipping Copernicus raster processing and reusing master_csv as tagged master."
        )
        if Path(args.master_csv).resolve() != master_tagged.resolve():
            shutil.copyfile(args.master_csv, master_tagged)
            print(f"Copied master matrix: {master_tagged}")
    existing_presence = build_presence_from_existing(
        args.existing_presence_csv,
        g_lon_min,
        g_lon_max,
        g_lat_min,
        g_lat_max,
    )
    if len(existing_presence) > 0:
        existing_presence.to_csv(presence_tagged, index=False)
        quick_presence_report = {
            "mode": "existing_snapped_presence",
            "existing_presence_csv": str(args.existing_presence_csv),
            "selected_rows": int(len(existing_presence)),
            "gulf_bbox_with_buffer": {
                "lon_min": g_lon_min,
                "lon_max": g_lon_max,
                "lat_min": g_lat_min,
                "lat_max": g_lat_max,
            },
            "output_csv": str(presence_tagged),
        }
        presence_report.write_text(json.dumps(quick_presence_report, indent=2), encoding="utf-8")
        print(f"Using existing snapped presence rows: {len(existing_presence)}")
    else:
        run(
            [
                "python",
                "ingest_presence_records.py",
                "--inputs",
                str(args.out_gulf_presence_plan_csv),
                "--max_snap_m",
                str(args.max_snap_m),
                "--species_filter",
                "kappaphycus",
                "--master_csv",
                str(master_tagged),
                "--training_csv",
                str(training_tagged),
                "--out_csv",
                str(presence_tagged),
                "--out_report",
                str(presence_report),
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
            str(master_tagged),
            "--presence_csv",
            str(presence_tagged),
            "--output",
            str(training_tagged),
        ]
    )

    print("Copernicus flow complete.")
    print(f"Tagged master: {master_tagged}")
    print(f"Tagged presence: {presence_tagged}")
    print(f"Tagged training dataset: {training_tagged}")


if __name__ == "__main__":
    main()
