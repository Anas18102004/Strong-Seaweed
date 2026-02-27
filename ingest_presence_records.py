import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from project_paths import BASE, TABULAR_DIR, REPORTS_DIR, ensure_dirs, with_legacy


MASTER = with_legacy(TABULAR_DIR / "master_feature_matrix.csv", "master_feature_matrix.csv")
OUT_CSV = TABULAR_DIR / "kappaphycus_presence_snapped_clean.csv"
OUT_REPORT = REPORTS_DIR / "presence_ingestion_report.json"
V11_TEMPLATE = TABULAR_DIR / "v1_1_data_plan.csv"
TRAINING_DATASET = with_legacy(TABULAR_DIR / "training_dataset.csv", "training_dataset.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest + quality-filter Kappaphycus presence points.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=False,
        help="Input CSV files containing lon/lat (and optional date/year).",
    )
    p.add_argument(
        "--master_csv",
        type=Path,
        default=MASTER,
        help="Master feature matrix CSV to snap against.",
    )
    p.add_argument(
        "--training_csv",
        type=Path,
        default=TRAINING_DATASET,
        help="Training dataset CSV used for strict acceptance duplicate checks.",
    )
    p.add_argument("--out_csv", type=Path, default=OUT_CSV)
    p.add_argument("--out_report", type=Path, default=OUT_REPORT)
    p.add_argument(
        "--min_year",
        type=int,
        default=2000,
        help="Drop records older than this year when year/eventDate exists.",
    )
    p.add_argument(
        "--max_snap_m",
        type=float,
        default=1200.0,
        help="Max distance for snapping points to nearest valid master-grid pixel.",
    )
    p.add_argument(
        "--dedup_decimals",
        type=int,
        default=5,
        help="Decimal precision for geographic deduplication.",
    )
    p.add_argument(
        "--species_filter",
        type=str,
        default="",
        help="Optional species name contains filter (e.g., 'kappaphycus').",
    )
    p.add_argument(
        "--require_verified",
        action="store_true",
        help="Keep only records marked as verified/approved in input metadata when available.",
    )
    p.add_argument(
        "--make_template",
        action="store_true",
        help="Write a v1.1 data plan template CSV and exit.",
    )
    p.add_argument(
        "--strict_acceptance",
        action="store_true",
        help="Apply v1.1 QA acceptance rules for positive ingestion.",
    )
    p.add_argument(
        "--strict_min_confidence",
        type=float,
        default=0.85,
        help="Minimum confidence_score required in strict mode.",
    )
    p.add_argument(
        "--strict_min_verified_dates",
        type=int,
        default=2,
        help="Minimum distinct verified dates required from notes in strict mode.",
    )
    p.add_argument(
        "--strict_depth_max_m",
        type=float,
        default=8.0,
        help="Maximum snapped-cell depth allowed in strict mode.",
    )
    p.add_argument(
        "--strict_distance_to_shore_max_m",
        type=float,
        default=2000.0,
        help="Maximum snapped-cell distance_to_shore allowed in strict mode.",
    )
    return p.parse_args()


def find_lon_lat_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    lon_candidates = ["lon", "longitude", "decimal_longitude", "decimalLongitude"]
    lat_candidates = ["lat", "latitude", "decimal_latitude", "decimalLatitude"]
    lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
    lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
    if not lon_col or not lat_col:
        raise ValueError("Could not detect lon/lat columns.")
    return lon_col, lat_col


def extract_year(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    if "eventDate" in df.columns:
        return pd.to_datetime(df["eventDate"], errors="coerce").dt.year
    return pd.Series(np.nan, index=df.index)


def load_and_standardize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lon_col, lat_col = find_lon_lat_columns(df)

    out = pd.DataFrame()
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["year"] = extract_year(df)
    out["source_file"] = path.name
    out["source_rows"] = len(df)

    for c in ["species", "scientificName"]:
        if c in df.columns:
            out["species_text"] = df[c].astype(str)
            break
    if "species_text" not in out.columns:
        out["species_text"] = ""

    # Optional governance / provenance columns used by v1.1 data plan.
    if "label" in df.columns:
        out["label_value"] = pd.to_numeric(df["label"], errors="coerce")
    else:
        out["label_value"] = np.nan
    if "is_verified" in df.columns:
        out["is_verified"] = df["is_verified"]
    else:
        out["is_verified"] = ""
    if "qa_status" in df.columns:
        out["qa_status"] = df["qa_status"].astype(str)
    else:
        out["qa_status"] = ""
    for c in [
        "source_type",
        "source_name",
        "source_reference",
        "citation_url",
        "record_id",
        "notes",
        "rationale",
        "qa_reviewer",
    ]:
        out[c] = df[c] if c in df.columns else ""
    for c in ["coordinate_precision_km", "confidence_score"]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if "species_confirmed" in df.columns:
        out["species_confirmed"] = df["species_confirmed"]
    else:
        out["species_confirmed"] = ""
    return out


def write_v11_template(path: Path) -> None:
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
    sample = {
        "record_id": "example_001",
        "source_type": "literature",
        "source_name": "example_paper",
        "source_reference": "doi:10.xxxx/xxxxx",
        "citation_url": "https://example.org/paper",
        "species": "Kappaphycus alvarezii",
        "eventDate": "2022-06-15",
        "year": 2022,
        "lon": 79.1324556,
        "lat": 9.2895528,
        "label": 1,
        "coordinate_precision_km": 0.5,
        "species_confirmed": True,
        "confidence_score": 0.9,
        "is_verified": True,
        "qa_reviewer": "reviewer_name",
        "qa_status": "approved",
        "rationale": "literature coordinate with confirmed species",
        "notes": "verified_dates=2023-01-10;2023-03-20;2023-06-05",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sample], columns=cols).to_csv(path, index=False)


def _extract_verified_dates_count(notes: str) -> int:
    if not isinstance(notes, str):
        return 0
    m = re.search(r"verified_dates\s*=\s*([^|]+)", notes, flags=re.IGNORECASE)
    if not m:
        return 0
    raw = m.group(1).strip()
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    # Expect ISO-like date tokens; keep unique normalized tokens.
    uniq = {p for p in parts}
    return len(uniq)


def apply_strict_acceptance_rules(
    q: pd.DataFrame,
    master: pd.DataFrame,
    training_csv: Path,
    max_snap_m: float,
    min_confidence: float,
    min_verified_dates: int,
    depth_max_m: float,
    distance_to_shore_max_m: float,
) -> pd.DataFrame:
    required_cols = [
        "source_type",
        "species_text",
        "coordinate_precision_km",
        "species_confirmed",
        "is_verified",
        "qa_status",
    ]
    missing = [c for c in required_cols if c not in q.columns]
    if missing:
        raise ValueError(f"Strict acceptance requires columns: {missing}")

    # 1) species must be kappaphycus
    species_ok = q["species_text"].astype(str).str.lower().str.contains("kappaphycus", na=False)
    q = q[species_ok].copy()

    # 2) coordinate precision <= 1 km
    q = q[q["coordinate_precision_km"].notna() & (q["coordinate_precision_km"] <= 1.0)].copy()

    # 3) source type in accepted list
    ok_sources = {"literature", "government", "satellite_digitized"}
    q = q[q["source_type"].astype(str).str.strip().str.lower().isin(ok_sources)].copy()

    # 4) species confirmed + verification + approved QA
    true_set = {"1", "true", "t", "yes", "y"}
    q = q[q["species_confirmed"].astype(str).str.strip().str.lower().isin(true_set)].copy()
    q = q[q["is_verified"].astype(str).str.strip().str.lower().isin(true_set)].copy()
    q = q[q["qa_status"].astype(str).str.strip().str.lower().isin({"approved", "verified", "accepted"})].copy()

    # 4b) confidence floor
    q = q[q["confidence_score"].notna() & (q["confidence_score"] >= float(min_confidence))].copy()

    # 4c) minimum verified dates evidence from notes.
    if "notes" in q.columns:
        q["_verified_dates_count"] = q["notes"].astype(str).apply(_extract_verified_dates_count)
        q = q[q["_verified_dates_count"] >= int(min_verified_dates)].copy()
    else:
        q = q.iloc[0:0].copy()

    # 5) not within 1km of existing positives (when training data available)
    if training_csv.exists():
        tr = pd.read_csv(training_csv)
        pos = tr[tr["label"] == 1][["lon", "lat"]].drop_duplicates()
        if not pos.empty and not q.empty:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
            px, py = transformer.transform(pos["lon"].to_numpy(), pos["lat"].to_numpy())
            qx, qy = transformer.transform(q["lon"].to_numpy(), q["lat"].to_numpy())
            tree = cKDTree(np.c_[px, py])
            d_m, _ = tree.query(np.c_[qx, qy], k=1)
            q = q[d_m > 1000.0].copy()

    # 6) snapped-cell environmental gates + avoid same positive cell
    if training_csv.exists() and not q.empty:
        tr = pd.read_csv(training_csv)
        pos = tr[tr["label"] == 1][["lon", "lat"]].drop_duplicates()
        pos_set = set(zip(pos["lon"].round(8), pos["lat"].round(8)))
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
        mx, my = transformer.transform(master["lon"].to_numpy(), master["lat"].to_numpy())
        qx, qy = transformer.transform(q["lon"].to_numpy(), q["lat"].to_numpy())
        tree = cKDTree(np.c_[mx, my])
        d_m, idx = tree.query(np.c_[qx, qy], k=1)
        snapped = master.iloc[idx][["lon", "lat"]].reset_index(drop=True)
        # Also ensure strict snapping distance.
        snap_ok = d_m <= float(max_snap_m)
        env_ok = np.ones(len(snapped), dtype=bool)
        if "depth" in master.columns:
            env_ok &= master.iloc[idx]["depth"].to_numpy(dtype=float) <= float(depth_max_m)
        if "distance_to_shore" in master.columns:
            env_ok &= master.iloc[idx]["distance_to_shore"].to_numpy(dtype=float) <= float(
                distance_to_shore_max_m
            )
        if "shallow_mask" in master.columns:
            env_ok &= master.iloc[idx]["shallow_mask"].to_numpy(dtype=float) >= 1.0
        keep = [
            bool(snap_ok[i])
            and bool(env_ok[i])
            and ((round(lon, 8), round(lat, 8)) not in pos_set)
            for i, (lon, lat) in enumerate(zip(snapped["lon"], snapped["lat"]))
        ]
        q = q[np.asarray(keep, dtype=bool)].copy()

    return q


def snap_to_master_grid(points: pd.DataFrame, master: pd.DataFrame, max_snap_m: float) -> tuple[pd.DataFrame, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    mx, my = transformer.transform(master["lon"].to_numpy(), master["lat"].to_numpy())
    px, py = transformer.transform(points["lon"].to_numpy(), points["lat"].to_numpy())

    tree = cKDTree(np.c_[mx, my])
    dist_m, idx = tree.query(np.c_[px, py], k=1)
    keep = dist_m <= max_snap_m
    snapped = master.iloc[idx[keep]][["lon", "lat"]].copy().reset_index(drop=True)
    return snapped, dist_m


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.make_template:
        write_v11_template(V11_TEMPLATE)
        print(f"Saved template: {V11_TEMPLATE}")
        return
    if not args.inputs:
        raise ValueError("Provide --inputs files, or run with --make_template.")
    if not args.master_csv.exists():
        raise FileNotFoundError(f"Missing master feature matrix: {args.master_csv}")

    master = pd.read_csv(args.master_csv)
    bounds = {
        "lon_min": float(master["lon"].min()),
        "lon_max": float(master["lon"].max()),
        "lat_min": float(master["lat"].min()),
        "lat_max": float(master["lat"].max()),
    }

    raw_parts = []
    file_stats = []
    for p in [Path(x) for x in args.inputs]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        part = load_and_standardize(p)
        raw_parts.append(part)
        file_stats.append({"file": p.name, "rows": int(len(part))})

    raw = pd.concat(raw_parts, ignore_index=True)
    n_raw = len(raw)

    # Basic geospatial validity.
    q = raw.dropna(subset=["lon", "lat"]).copy()
    q = q[(q["lon"] >= -180) & (q["lon"] <= 180) & (q["lat"] >= -90) & (q["lat"] <= 90)]

    # Domain bounding-box filter.
    q = q[
        (q["lon"] >= bounds["lon_min"])
        & (q["lon"] <= bounds["lon_max"])
        & (q["lat"] >= bounds["lat_min"])
        & (q["lat"] <= bounds["lat_max"])
    ].copy()

    # Optional species filter.
    if args.species_filter.strip():
        mask = q["species_text"].str.lower().str.contains(args.species_filter.strip().lower(), na=False)
        q = q[mask].copy()

    # Keep only presence labels if a label column exists.
    if q["label_value"].notna().any():
        q = q[(q["label_value"].isna()) | (q["label_value"] == 1)].copy()

    # Year quality filter when available.
    q = q[(q["year"].isna()) | (q["year"] >= int(args.min_year))].copy()

    # Optional strict verification filter for governance-driven ingestion.
    if args.require_verified:
        if q["is_verified"].astype(str).str.len().gt(0).any():
            ok_true = {"1", "true", "t", "yes", "y"}
            verified_mask = q["is_verified"].astype(str).str.strip().str.lower().isin(ok_true)
            q = q[verified_mask].copy()
        if q["qa_status"].astype(str).str.len().gt(0).any():
            ok_status = {"approved", "verified", "accepted"}
            status_mask = q["qa_status"].astype(str).str.strip().str.lower().isin(ok_status)
            q = q[status_mask].copy()

    if args.strict_acceptance:
        q = apply_strict_acceptance_rules(
            q=q,
            master=master,
            training_csv=args.training_csv,
            max_snap_m=float(args.max_snap_m),
            min_confidence=float(args.strict_min_confidence),
            min_verified_dates=int(args.strict_min_verified_dates),
            depth_max_m=float(args.strict_depth_max_m),
            distance_to_shore_max_m=float(args.strict_distance_to_shore_max_m),
        )

    # Pre-snap dedup.
    q["lon_r"] = q["lon"].round(args.dedup_decimals)
    q["lat_r"] = q["lat"].round(args.dedup_decimals)
    q = q.drop_duplicates(subset=["lon_r", "lat_r"]).copy()

    snapped, dist_m = snap_to_master_grid(q, master, max_snap_m=float(args.max_snap_m))
    snapped["lon_r"] = snapped["lon"].round(args.dedup_decimals)
    snapped["lat_r"] = snapped["lat"].round(args.dedup_decimals)
    snapped = snapped.drop_duplicates(subset=["lon_r", "lat_r"])[["lon", "lat"]].reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    snapped.to_csv(args.out_csv, index=False)

    report = {
        "inputs": file_stats,
        "raw_rows": int(n_raw),
        "after_quality_filters": int(len(q)),
        "snapped_rows": int(len(snapped)),
        "max_snap_m": float(args.max_snap_m),
        "min_year": int(args.min_year),
        "bounds": bounds,
        "median_snap_m_before_threshold": float(np.nanmedian(dist_m)) if len(dist_m) else None,
        "output_csv": str(args.out_csv),
        "master_csv": str(args.master_csv),
        "training_csv": str(args.training_csv),
        "require_verified": bool(args.require_verified),
        "strict_acceptance": bool(args.strict_acceptance),
        "strict_min_confidence": float(args.strict_min_confidence),
        "strict_min_verified_dates": int(args.strict_min_verified_dates),
        "strict_depth_max_m": float(args.strict_depth_max_m),
        "strict_distance_to_shore_max_m": float(args.strict_distance_to_shore_max_m),
        "sources_after_filter": q["source_file"].value_counts().to_dict(),
    }
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved cleaned presence: {args.out_csv}")
    print(f"Saved report: {args.out_report}")
    print(f"Raw rows: {n_raw} -> filtered: {len(q)} -> snapped unique: {len(snapped)}")


if __name__ == "__main__":
    main()
