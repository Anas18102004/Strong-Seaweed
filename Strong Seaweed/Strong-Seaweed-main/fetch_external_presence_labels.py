import argparse
from pathlib import Path

import pandas as pd
import requests
from project_paths import TABULAR_DIR, ensure_dirs, with_legacy

MASTER = with_legacy(TABULAR_DIR / "master_feature_matrix.csv", "master_feature_matrix.csv")

SPECIES_CANDIDATES = [
    "Kappaphycus",
    "Kappaphycus alvarezii",
    "Kappaphycus striatus",
    "Kappaphycus cottonii",
    "Eucheuma denticulatum",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch external Kappaphycus presence labels from OBIS/GBIF.")
    p.add_argument("--out", type=Path, default=TABULAR_DIR / "external_presence_candidates.csv")
    p.add_argument("--min_lon", type=float, default=None)
    p.add_argument("--max_lon", type=float, default=None)
    p.add_argument("--min_lat", type=float, default=None)
    p.add_argument("--max_lat", type=float, default=None)
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args()


def get_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if None not in (args.min_lon, args.max_lon, args.min_lat, args.max_lat):
        return float(args.min_lon), float(args.max_lon), float(args.min_lat), float(args.max_lat)
    if not MASTER.exists():
        raise FileNotFoundError(f"Missing {MASTER}. Provide bbox explicitly.")
    df = pd.read_csv(MASTER, usecols=["lon", "lat"])
    return float(df.lon.min()), float(df.lon.max()), float(df.lat.min()), float(df.lat.max())


def fetch_obis(species: str, bbox: tuple[float, float, float, float], timeout: int) -> list[dict]:
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"

    rows = []
    size = 1000
    start = 0
    while True:
        url = "https://api.obis.org/v3/occurrence"
        params = {"scientificname": species, "geometry": poly, "size": size, "start": start}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        res = js.get("results", [])
        if not res:
            break
        for x in res:
            rows.append(
                {
                    "lon": x.get("decimalLongitude"),
                    "lat": x.get("decimalLatitude"),
                    "year": x.get("year"),
                    "eventDate": x.get("eventDate"),
                    "species": species,
                    "source": "OBIS",
                    "record_id": x.get("id"),
                }
            )
        start += size
        if start >= int(js.get("total", 0)):
            break
    return rows


def fetch_gbif(species: str, bbox: tuple[float, float, float, float], timeout: int) -> list[dict]:
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    rows = []
    offset = 0
    limit = 300
    while True:
        url = "https://api.gbif.org/v1/occurrence/search"
        params = {
            "scientificName": species,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "geometry": poly,
            "offset": offset,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        res = js.get("results", [])
        if not res:
            break
        for x in res:
            rows.append(
                {
                    "lon": x.get("decimalLongitude"),
                    "lat": x.get("decimalLatitude"),
                    "year": x.get("year"),
                    "eventDate": x.get("eventDate"),
                    "species": species,
                    "source": "GBIF",
                    "record_id": x.get("key"),
                }
            )
        offset += limit
        if offset >= int(js.get("count", 0)):
            break
    return rows


def main() -> None:
    ensure_dirs()
    args = parse_args()
    bbox = get_bbox(args)
    print("Using bbox:", bbox)

    all_rows = []
    for sp in SPECIES_CANDIDATES:
        try:
            obis_rows = fetch_obis(sp, bbox, args.timeout)
        except Exception as e:
            print(f"OBIS failed for {sp}: {e}")
            obis_rows = []
        try:
            gbif_rows = fetch_gbif(sp, bbox, args.timeout)
        except Exception as e:
            print(f"GBIF failed for {sp}: {e}")
            gbif_rows = []
        print(f"{sp}: OBIS={len(obis_rows)} GBIF={len(gbif_rows)}")
        all_rows.extend(obis_rows)
        all_rows.extend(gbif_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=["lon", "lat", "year", "eventDate", "species", "source", "record_id"])
    else:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df = df.dropna(subset=["lon", "lat"])
        df = df.drop_duplicates(subset=["source", "record_id", "lon", "lat"])
        df = df.sort_values(["source", "species", "year"], na_position="last").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} | rows={len(df)}")


if __name__ == "__main__":
    main()
