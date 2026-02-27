import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

from project_paths import TABULAR_DIR, EXPERIMENTS_DIR, REPORTS_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch web Kappaphycus occurrences (OBIS/GBIF), snap to grid, and export new positive cells."
    )
    p.add_argument(
        "--master_csv",
        type=Path,
        default=TABULAR_DIR / "master_feature_matrix_v1_1_augmented.csv",
    )
    p.add_argument(
        "--training_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_merged46_plus_hn30_augmented.csv",
    )
    p.add_argument("--max_snap_m", type=float, default=1500.0)
    p.add_argument(
        "--output_raw_csv",
        type=Path,
        default=EXPERIMENTS_DIR / "web_kappaphycus_raw_occurrences.csv",
    )
    p.add_argument(
        "--output_snapped_csv",
        type=Path,
        default=EXPERIMENTS_DIR / "web_kappaphycus_snapped_new_cells.csv",
    )
    p.add_argument(
        "--output_report",
        type=Path,
        default=REPORTS_DIR / "web_kappaphycus_fetch_report.json",
    )
    return p.parse_args()


def fetch_obis(bbox: dict) -> list[dict]:
    geom = (
        f"POLYGON(({bbox['lon_min']} {bbox['lat_min']},"
        f"{bbox['lon_max']} {bbox['lat_min']},"
        f"{bbox['lon_max']} {bbox['lat_max']},"
        f"{bbox['lon_min']} {bbox['lat_max']},"
        f"{bbox['lon_min']} {bbox['lat_min']}))"
    )
    names = ["Kappaphycus", "Kappaphycus alvarezii", "Kappaphycus cottonii"]
    rows: list[dict] = []
    for name in names:
        r = requests.get(
            "https://api.obis.org/v3/occurrence",
            params={"scientificname": name, "geometry": geom, "size": 1000},
            timeout=60,
        )
        if r.status_code != 200:
            continue
        j = r.json()
        for rec in j.get("results", []):
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            rows.append(
                {
                    "source_api": "obis",
                    "record_id": f"obis-{rec.get('id', '')}",
                    "species": rec.get("scientificName", name),
                    "lat": float(lat),
                    "lon": float(lon),
                    "source_url": "https://api.obis.org/v3/occurrence",
                    "source_ref": f"OBIS occurrence ({name})",
                }
            )
    return rows


def gbif_species_key(name: str) -> int | None:
    r = requests.get(
        "https://api.gbif.org/v1/species/match", params={"name": name}, timeout=30
    )
    if r.status_code != 200:
        return None
    return r.json().get("usageKey")


def fetch_gbif(bbox: dict) -> list[dict]:
    rows: list[dict] = []
    key = gbif_species_key("Kappaphycus alvarezii")
    if not key:
        return rows

    offset = 0
    limit = 300
    for _ in range(8):  # up to 2400 records window
        r = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "taxonKey": key,
                "hasCoordinate": "true",
                "country": "IN",
                "limit": limit,
                "offset": offset,
            },
            timeout=60,
        )
        if r.status_code != 200:
            break
        j = r.json()
        res = j.get("results", [])
        if not res:
            break
        for rec in res:
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            lat = float(lat)
            lon = float(lon)
            if not (
                bbox["lon_min"] <= lon <= bbox["lon_max"]
                and bbox["lat_min"] <= lat <= bbox["lat_max"]
            ):
                continue
            occ = rec.get("key")
            rows.append(
                {
                    "source_api": "gbif",
                    "record_id": f"gbif-{occ}",
                    "species": rec.get("species", "Kappaphycus alvarezii"),
                    "lat": lat,
                    "lon": lon,
                    "source_url": f"https://www.gbif.org/occurrence/{occ}"
                    if occ
                    else "https://api.gbif.org/v1/occurrence/search",
                    "source_ref": "GBIF occurrence",
                }
            )
        offset += limit
        if offset >= j.get("count", 0):
            break
    return rows


def snap_to_master(points: pd.DataFrame, master: pd.DataFrame, max_snap_m: float) -> pd.DataFrame:
    t = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    mx, my = t.transform(master["lon"].to_numpy(), master["lat"].to_numpy())
    px, py = t.transform(points["lon"].to_numpy(), points["lat"].to_numpy())

    tree = cKDTree(np.c_[mx, my])
    dist_m, idx = tree.query(np.c_[px, py], k=1)
    keep = dist_m <= max_snap_m
    out = points.loc[keep].copy().reset_index(drop=True)
    out["snap_km"] = (dist_m[keep] / 1000.0).astype(float)
    snapped = master.iloc[idx[keep]].reset_index(drop=True)
    out["snap_lon"] = snapped["lon"].to_numpy()
    out["snap_lat"] = snapped["lat"].to_numpy()
    return out


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not args.master_csv.exists():
        raise FileNotFoundError(f"Missing master_csv: {args.master_csv}")
    if not args.training_csv.exists():
        raise FileNotFoundError(f"Missing training_csv: {args.training_csv}")

    master = pd.read_csv(args.master_csv)
    train = pd.read_csv(args.training_csv)
    bbox = {
        "lon_min": float(master["lon"].min()),
        "lon_max": float(master["lon"].max()),
        "lat_min": float(master["lat"].min()),
        "lat_max": float(master["lat"].max()),
    }

    rows = []
    rows.extend(fetch_obis(bbox))
    rows.extend(fetch_gbif(bbox))
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No web occurrences fetched in bbox.")

    # Kappaphycus only.
    raw = raw[
        raw["species"].astype(str).str.lower().str.contains("kappaphycus", na=False)
    ].copy()
    raw = raw.drop_duplicates(subset=["record_id"])
    raw["coord6"] = raw["lat"].round(6).astype(str) + "_" + raw["lon"].round(6).astype(str)
    raw = raw.drop_duplicates(subset=["coord6"]).drop(columns=["coord6"])

    args.output_raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_raw_csv, index=False)

    snapped = snap_to_master(raw, master, max_snap_m=float(args.max_snap_m))
    if snapped.empty:
        raise RuntimeError("No occurrences snapped within max_snap_m.")

    # Keep only new positive cells not already in training positives.
    pos = train[train["label"] == 1][["lon", "lat"]].drop_duplicates()
    pos_keys = set(zip(pos["lon"].round(8), pos["lat"].round(8)))
    snapped["is_existing_positive_cell"] = [
        (round(float(lo), 8), round(float(la), 8)) in pos_keys
        for lo, la in zip(snapped["snap_lon"], snapped["snap_lat"])
    ]
    new_cells = snapped[~snapped["is_existing_positive_cell"]].copy()

    # Attach full feature vector from snapped master cell.
    feat = master.merge(
        new_cells[["snap_lon", "snap_lat"]].rename(columns={"snap_lon": "lon", "snap_lat": "lat"}),
        on=["lon", "lat"],
        how="inner",
    ).drop_duplicates(subset=["lon", "lat"])
    feat["label"] = 1

    # Add provenance columns for ingestion planning.
    if not feat.empty:
        feat = feat.rename(columns={"lon": "snap_lon", "lat": "snap_lat"})
        enriched = new_cells.merge(feat, on=["snap_lon", "snap_lat"], how="inner")
    else:
        enriched = new_cells.copy()

    args.output_snapped_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_snapped_csv, index=False)

    report = {
        "bbox": bbox,
        "raw_total": int(len(rows)),
        "raw_kappaphycus_unique": int(len(raw)),
        "snapped_within_max_snap_m": int(len(snapped)),
        "new_unique_positive_cells": int(
            len(new_cells.drop_duplicates(subset=["snap_lon", "snap_lat"]))
        ),
        "max_snap_m": float(args.max_snap_m),
        "median_snap_km": float(snapped["snap_km"].median()) if len(snapped) else None,
        "sources": raw["source_api"].value_counts().to_dict(),
        "output_raw_csv": str(args.output_raw_csv),
        "output_snapped_csv": str(args.output_snapped_csv),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved raw: {args.output_raw_csv}")
    print(f"Saved snapped new cells: {args.output_snapped_csv}")
    print(f"Saved report: {args.output_report}")
    print(
        "raw_unique={} snapped={} new_cells={}".format(
            report["raw_kappaphycus_unique"],
            report["snapped_within_max_snap_m"],
            report["new_unique_positive_cells"],
        )
    )


if __name__ == "__main__":
    main()
