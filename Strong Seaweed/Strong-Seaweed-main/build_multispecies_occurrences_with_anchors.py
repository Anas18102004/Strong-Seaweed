import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


SPECIES_CONFIG = {
    "gracilaria_spp": {
        "display_name": "Gracilaria spp.",
        "queries": ["Hydropuntia edulis", "Gracilaria edulis", "Gracilaria"],
    },
    "ulva_spp": {
        "display_name": "Ulva spp.",
        "queries": ["Ulva lactuca", "Ulva"],
    },
    "sargassum_spp": {
        "display_name": "Sargassum spp.",
        "queries": ["Sargassum swartzii", "Sargassum wightii", "Sargassum"],
    },
}


# Curated anchors for known west-coast cultivation/collection zones.
# These are treated as high-confidence weak labels to improve region coverage.
CURATED_ANCHORS = [
    {"species_id": "sargassum_spp", "name": "Okha", "lat": 22.4678, "lon": 69.0700, "weight": 1.35},
    {"species_id": "sargassum_spp", "name": "Dwarka", "lat": 22.2442, "lon": 68.9685, "weight": 1.30},
    {"species_id": "sargassum_spp", "name": "Bet_Dwarka", "lat": 22.4707, "lon": 69.1370, "weight": 1.30},
    {"species_id": "sargassum_spp", "name": "Porbandar", "lat": 21.6417, "lon": 69.6293, "weight": 1.25},
    {"species_id": "sargassum_spp", "name": "Veraval", "lat": 20.9077, "lon": 70.3673, "weight": 1.25},
    {"species_id": "sargassum_spp", "name": "Jafrabad", "lat": 20.8685, "lon": 71.3707, "weight": 1.20},
    {"species_id": "gracilaria_spp", "name": "Mithapur", "lat": 22.4118, "lon": 69.0089, "weight": 1.20},
    {"species_id": "gracilaria_spp", "name": "Okha", "lat": 22.4678, "lon": 69.0700, "weight": 1.18},
    {"species_id": "ulva_spp", "name": "Mithapur", "lat": 22.4118, "lon": 69.0089, "weight": 1.15},
    {"species_id": "ulva_spp", "name": "Porbandar", "lat": 21.6417, "lon": 69.6293, "weight": 1.12},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build multispecies occurrence CSV from OBIS/GBIF plus curated Gujarat anchors."
    )
    p.add_argument("--out_csv", type=Path, default=TABULAR_DIR / "multispecies_occurrences_multispecies_india_v6_gujarat.csv")
    p.add_argument("--report_json", type=Path, default=REPORTS_DIR / "multispecies_occurrences_v6_gujarat_report.json")
    p.add_argument("--min_lon", type=float, default=68.0)
    p.add_argument("--max_lon", type=float, default=98.5)
    p.add_argument("--min_lat", type=float, default=5.0)
    p.add_argument("--max_lat", type=float, default=24.5)
    p.add_argument("--timeout", type=int, default=35)
    p.add_argument("--max_records_per_source", type=int, default=3000)
    p.add_argument("--min_year", type=int, default=2005)
    p.add_argument("--anchor_weight", type=float, default=1.25)
    p.add_argument(
        "--verified_labels_csv",
        type=Path,
        default=TABULAR_DIR / "verified_farm_labels_multispecies_india_v1.csv",
    )
    return p.parse_args()


def _polygon(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> str:
    return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"


def fetch_obis(query: str, bbox: tuple[float, float, float, float], timeout: int, max_records: int) -> list[dict]:
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = _polygon(min_lon, max_lon, min_lat, max_lat)
    rows: list[dict] = []
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
                    "source": "OBIS",
                    "record_id": str(x.get("id") or ""),
                    "species_reported": x.get("scientificName") or x.get("species") or query,
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
    rows: list[dict] = []
    offset = 0
    limit = 300
    while len(rows) < max_records:
        r = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "scientificName": query,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "geometry": poly,
                "offset": offset,
                "limit": limit,
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
                    "source": "GBIF",
                    "record_id": str(x.get("key") or ""),
                    "species_reported": x.get("scientificName") or x.get("species") or query,
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


def normalize_occ(df: pd.DataFrame, bbox: tuple[float, float, float, float], min_year: int) -> pd.DataFrame:
    if df.empty:
        return df
    min_lon, max_lon, min_lat, max_lat = bbox
    df = df.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])
    df = df[
        (df["lon"] >= min_lon)
        & (df["lon"] <= max_lon)
        & (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
    ].copy()
    if min_year > 0:
        df = df[(df["year"].isna()) | (df["year"] >= int(min_year))]
    df["lon_r5"] = df["lon"].round(5)
    df["lat_r5"] = df["lat"].round(5)
    df = df.drop_duplicates(subset=["source", "record_id", "lon_r5", "lat_r5"])
    return df.drop(columns=["lon_r5", "lat_r5"], errors="ignore").reset_index(drop=True)


def build_anchor_rows(anchor_weight: float) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(CURATED_ANCHORS, start=1):
        rows.append(
            {
                "species_id": str(a["species_id"]),
                "source": "CURATED_ANCHOR",
                "record_id": f"anchor-{i:03d}",
                "species_reported": str(a["name"]),
                "lon": float(a["lon"]),
                "lat": float(a["lat"]),
                "event_date": "",
                "year": None,
                "label_weight": float(a.get("weight", anchor_weight)),
                "source_name": "gujarat_cultivation_anchor",
            }
        )
    return pd.DataFrame(rows)


def build_verified_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "species_id",
                "source",
                "record_id",
                "species_reported",
                "lon",
                "lat",
                "event_date",
                "year",
                "label_weight",
                "source_name",
            ]
        )
    df = pd.read_csv(path)
    required = {"species_id", "lon", "lat"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"verified_labels_csv missing columns: {missing}")
    out = pd.DataFrame()
    out["species_id"] = df["species_id"].astype(str)
    out["source"] = "VERIFIED_FARM"
    if "record_id" in df.columns:
        out["record_id"] = df["record_id"].astype(str)
    else:
        out["record_id"] = [f"verified-{i:04d}" for i in range(1, len(df) + 1)]
    out["species_reported"] = (
        df["note"].astype(str)
        if "note" in df.columns
        else df["species_id"].astype(str)
    )
    out["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    out["event_date"] = ""
    out["year"] = pd.NA
    out["label_weight"] = (
        pd.to_numeric(df.get("label_weight", 1.35), errors="coerce")
        .fillna(1.35)
        .clip(lower=0.2, upper=2.0)
    )
    out["source_name"] = (
        df["source_name"].astype(str)
        if "source_name" in df.columns
        else "verified_farm_label"
    )
    out = out.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["species_id", "lon", "lat"]).reset_index(drop=True)
    return out


def main() -> None:
    ensure_dirs()
    args = parse_args()
    bbox = (float(args.min_lon), float(args.max_lon), float(args.min_lat), float(args.max_lat))

    species_reports = []
    all_rows = []

    for species_id, cfg in SPECIES_CONFIG.items():
        fetched = []
        for q in cfg["queries"]:
            try:
                fetched.extend(fetch_obis(q, bbox, args.timeout, args.max_records_per_source))
            except Exception as e:
                print(f"[WARN] OBIS failed for {q}: {e}")
            try:
                fetched.extend(fetch_gbif(q, bbox, args.timeout, args.max_records_per_source))
            except Exception as e:
                print(f"[WARN] GBIF failed for {q}: {e}")
        df = pd.DataFrame(fetched)
        if df.empty:
            df = pd.DataFrame(columns=["source", "record_id", "species_reported", "lon", "lat", "event_date", "year"])
        df = normalize_occ(df, bbox, args.min_year)
        if not df.empty:
            df["species_id"] = species_id
            df["label_weight"] = 1.0
            df["source_name"] = "web_occurrence"
            all_rows.append(df)
        species_reports.append(
            {
                "species_id": species_id,
                "queries": cfg["queries"],
                "rows": int(len(df)),
                "sources": df["source"].value_counts().to_dict() if not df.empty else {},
            }
        )
        print(f"[OK] {species_id}: {len(df)} rows")

    anchors = build_anchor_rows(anchor_weight=float(args.anchor_weight))
    all_rows.append(anchors)
    verified = build_verified_rows(args.verified_labels_csv)
    if not verified.empty:
        all_rows.append(verified)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if out.empty:
        raise RuntimeError("No occurrence rows fetched/built.")

    out["species_id"] = out["species_id"].astype(str)
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["label_weight"] = pd.to_numeric(out["label_weight"], errors="coerce").fillna(1.0).clip(lower=0.2, upper=2.0)
    out = out.dropna(subset=["lon", "lat"])
    out["lon_r5"] = out["lon"].round(5)
    out["lat_r5"] = out["lat"].round(5)
    out = out.drop_duplicates(subset=["species_id", "lon_r5", "lat_r5"], keep="first").drop(columns=["lon_r5", "lat_r5"])
    out = out.sort_values(["species_id", "source", "year"], na_position="last").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    report = {
        "bbox": {"min_lon": bbox[0], "max_lon": bbox[1], "min_lat": bbox[2], "max_lat": bbox[3]},
        "rows_total": int(len(out)),
        "species_counts": out["species_id"].value_counts().to_dict(),
        "source_counts": out["source"].value_counts().to_dict(),
        "species_reports": species_reports,
        "anchor_rows_added": int(len(anchors)),
        "verified_rows_added": int(len(verified)),
        "verified_labels_csv": str(args.verified_labels_csv),
        "output_csv": str(args.out_csv),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Saved: {args.out_csv}")
    print(f"[DONE] Report: {args.report_json}")


if __name__ == "__main__":
    main()
