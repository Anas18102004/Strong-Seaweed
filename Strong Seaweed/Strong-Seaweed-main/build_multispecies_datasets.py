import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from geo_utils import make_metric_transformer
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


DEFAULT_MASTER = TABULAR_DIR / "master_feature_matrix_kappa_india_gulf_v2_hardmerge4_augmented.csv"

SPECIES_CONFIG = {
    "kappaphycus_alvarezii": {
        "display_name": "Kappaphycus alvarezii",
        "queries": ["Kappaphycus alvarezii", "Kappaphycus"],
    },
    "gracilaria_spp": {
        "display_name": "Gracilaria spp.",
        "queries": ["Gracilaria", "Gracilaria edulis"],
    },
    "ulva_spp": {
        "display_name": "Ulva spp.",
        "queries": ["Ulva", "Ulva lactuca"],
    },
    "sargassum_spp": {
        "display_name": "Sargassum spp.",
        "queries": ["Sargassum", "Sargassum wightii"],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-species training datasets from GBIF/OBIS + master feature grid.")
    p.add_argument("--master_csv", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--min_lon", type=float, default=68.0)
    p.add_argument("--max_lon", type=float, default=98.5)
    p.add_argument("--min_lat", type=float, default=5.0)
    p.add_argument("--max_lat", type=float, default=24.5)
    p.add_argument("--max_records_per_source", type=int, default=2000)
    p.add_argument("--max_snap_m", type=float, default=12000.0)
    p.add_argument("--bg_ratio", type=int, default=5)
    p.add_argument("--min_neg_distance_m", type=float, default=4000.0)
    p.add_argument("--min_neg_abs", type=int, default=300)
    p.add_argument("--out_prefix", type=str, default="multispecies_india_v1")
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args()


def _polygon(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> str:
    return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"


def fetch_obis(query: str, bbox: tuple[float, float, float, float], timeout: int, max_records: int) -> list[dict]:
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = _polygon(min_lon, max_lon, min_lat, max_lat)
    rows = []
    size = 500
    start = 0
    while len(rows) < max_records:
        r = requests.get(
            "https://api.obis.org/v3/occurrence",
            params={"scientificname": query, "geometry": poly, "size": size, "start": start},
            timeout=timeout,
        )
        r.raise_for_status()
        js = r.json()
        chunk = js.get("results", [])
        if not chunk:
            break
        for x in chunk:
            rows.append(
                {
                    "query": query,
                    "source": "OBIS",
                    "record_id": x.get("id"),
                    "species_reported": x.get("scientificName") or x.get("species"),
                    "lon": x.get("decimalLongitude"),
                    "lat": x.get("decimalLatitude"),
                    "event_date": x.get("eventDate"),
                    "year": x.get("year"),
                }
            )
            if len(rows) >= max_records:
                break
        start += size
        if start >= int(js.get("total", 0)):
            break
    return rows


def fetch_gbif(query: str, bbox: tuple[float, float, float, float], timeout: int, max_records: int) -> list[dict]:
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = _polygon(min_lon, max_lon, min_lat, max_lat)
    rows = []
    limit = 300
    offset = 0
    while len(rows) < max_records:
        r = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "scientificName": query,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "geometry": poly,
                "limit": limit,
                "offset": offset,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        js = r.json()
        chunk = js.get("results", [])
        if not chunk:
            break
        for x in chunk:
            rows.append(
                {
                    "query": query,
                    "source": "GBIF",
                    "record_id": x.get("key"),
                    "species_reported": x.get("scientificName") or x.get("species"),
                    "lon": x.get("decimalLongitude"),
                    "lat": x.get("decimalLatitude"),
                    "event_date": x.get("eventDate"),
                    "year": x.get("year"),
                }
            )
            if len(rows) >= max_records:
                break
        offset += limit
        if offset >= int(js.get("count", 0)):
            break
    return rows


def snap_points_to_master(points: pd.DataFrame, master: pd.DataFrame, max_snap_m: float) -> tuple[pd.DataFrame, np.ndarray]:
    if points.empty:
        return master.iloc[0:0].copy(), np.array([], dtype=float)
    tf = make_metric_transformer()
    mx, my = tf.transform(master["lon"].to_numpy(dtype=np.float64), master["lat"].to_numpy(dtype=np.float64))
    px, py = tf.transform(points["lon"].to_numpy(dtype=np.float64), points["lat"].to_numpy(dtype=np.float64))
    tree = cKDTree(np.c_[mx, my])
    dist_m, idx = tree.query(np.c_[px, py], k=1)
    keep = dist_m <= max_snap_m
    snapped = master.iloc[idx[keep]].copy().reset_index(drop=True)
    snapped = snapped.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return snapped, dist_m


def sample_negatives(master: pd.DataFrame, positives: pd.DataFrame, n_target: int, min_neg_distance_m: float) -> pd.DataFrame:
    if n_target <= 0:
        return master.iloc[0:0].copy()
    tf = make_metric_transformer()
    mx, my = tf.transform(master["lon"].to_numpy(dtype=np.float64), master["lat"].to_numpy(dtype=np.float64))
    if positives.empty:
        pool = master.copy()
    else:
        px, py = tf.transform(positives["lon"].to_numpy(dtype=np.float64), positives["lat"].to_numpy(dtype=np.float64))
        tree = cKDTree(np.c_[px, py])
        d_m, _ = tree.query(np.c_[mx, my], k=1)
        pool = master[d_m >= float(min_neg_distance_m)].copy()
    if pool.empty:
        return pool
    n_take = min(n_target, len(pool))
    take_idx = np.random.default_rng(42).choice(pool.index.to_numpy(), size=n_take, replace=False)
    return pool.loc[take_idx].copy().reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.master_csv.exists():
        raise FileNotFoundError(f"Missing master_csv: {args.master_csv}")

    master = pd.read_csv(args.master_csv)
    master = master.dropna().drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    # Practical ecological domain for nearshore farming candidates.
    domain = master[
        (master["depth"] < 20.0)
        & (master["distance_to_shore"] <= 10000.0)
        & (master["shallow_mask"] > 0)
    ].copy()
    bbox = (float(args.min_lon), float(args.max_lon), float(args.min_lat), float(args.max_lat))

    all_occ = []
    dataset_reports = []

    for species_id, cfg in SPECIES_CONFIG.items():
        occ_rows = []
        for q in cfg["queries"]:
            try:
                occ_rows.extend(fetch_obis(q, bbox, args.timeout, args.max_records_per_source))
            except Exception as e:
                print(f"[WARN] OBIS failed for {q}: {e}")
            try:
                occ_rows.extend(fetch_gbif(q, bbox, args.timeout, args.max_records_per_source))
            except Exception as e:
                print(f"[WARN] GBIF failed for {q}: {e}")

        occ = pd.DataFrame(occ_rows)
        if occ.empty:
            occ = pd.DataFrame(columns=["query", "source", "record_id", "species_reported", "lon", "lat", "event_date", "year"])
        else:
            occ["lon"] = pd.to_numeric(occ["lon"], errors="coerce")
            occ["lat"] = pd.to_numeric(occ["lat"], errors="coerce")
            occ = occ.dropna(subset=["lon", "lat"])
            occ = occ[
                (occ["lon"] >= bbox[0]) & (occ["lon"] <= bbox[1]) &
                (occ["lat"] >= bbox[2]) & (occ["lat"] <= bbox[3])
            ]
            occ = occ.drop_duplicates(subset=["source", "record_id", "lon", "lat"]).reset_index(drop=True)

        occ["species_id"] = species_id
        occ["species_target"] = cfg["display_name"]
        all_occ.append(occ)

        snapped_pos, dist_m = snap_points_to_master(occ[["lon", "lat"]], domain, float(args.max_snap_m))
        n_pos = len(snapped_pos)
        n_neg_target = max(int(n_pos * int(args.bg_ratio)), int(args.min_neg_abs))
        negatives = sample_negatives(domain, snapped_pos[["lon", "lat"]], n_neg_target, float(args.min_neg_distance_m))

        pos_ds = snapped_pos.copy()
        pos_ds["label"] = 1
        pos_ds["label_weight"] = 1.0
        pos_ds["species_id"] = species_id
        pos_ds["species_target"] = cfg["display_name"]
        pos_ds["sample_type"] = "occurrence_positive"

        neg_ds = negatives.copy()
        neg_ds["label"] = 0
        neg_ds["label_weight"] = 1.0
        neg_ds["species_id"] = species_id
        neg_ds["species_target"] = cfg["display_name"]
        neg_ds["sample_type"] = "stratified_negative"

        ds = pd.concat([pos_ds, neg_ds], ignore_index=True)
        ds = ds.drop_duplicates(subset=["lon", "lat", "label"]).sample(frac=1, random_state=42).reset_index(drop=True)

        out_csv = TABULAR_DIR / f"training_dataset_{species_id}_{args.out_prefix}.csv"
        ds.to_csv(out_csv, index=False)

        dataset_reports.append(
            {
                "species_id": species_id,
                "species_target": cfg["display_name"],
                "queries": cfg["queries"],
                "raw_occurrences": int(len(occ)),
                "snapped_positives": int(n_pos),
                "negatives": int(len(neg_ds)),
                "total_rows": int(len(ds)),
                "median_snap_m": float(np.nanmedian(dist_m)) if len(dist_m) else None,
                "output_csv": str(out_csv),
            }
        )
        print(f"[OK] {species_id}: raw={len(occ)} snapped_pos={n_pos} neg={len(neg_ds)} total={len(ds)}")

    occ_all = pd.concat(all_occ, ignore_index=True) if all_occ else pd.DataFrame()
    occ_out = TABULAR_DIR / f"multispecies_occurrences_{args.out_prefix}.csv"
    occ_all.to_csv(occ_out, index=False)

    report = {
        "bbox": {"min_lon": bbox[0], "max_lon": bbox[1], "min_lat": bbox[2], "max_lat": bbox[3]},
        "master_csv": str(args.master_csv),
        "domain_rows": int(len(domain)),
        "occurrences_csv": str(occ_out),
        "species_reports": dataset_reports,
    }
    report_out = REPORTS_DIR / f"multispecies_dataset_build_report_{args.out_prefix}.json"
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Saved occurrences: {occ_out}")
    print(f"[DONE] Saved report: {report_out}")


if __name__ == "__main__":
    main()
