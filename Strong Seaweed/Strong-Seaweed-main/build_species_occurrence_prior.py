import argparse
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


DEFAULT_SPECIES = {
    "kappaphycus_alvarezii": "Kappaphycus alvarezii",
    "gracilaria_edulis": "Hydropuntia edulis",
    "ulva_lactuca": "Ulva lactuca",
    "sargassum_wightii": "Sargassum swartzii",
}


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def fetch_json(url: str, timeout_s: int = 30) -> dict:
    with urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_obis_points(scientific_name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float, max_records: int) -> pd.DataFrame:
    rows = []
    size = 1000
    start = 0
    while start < max_records:
        query = urlencode(
            {
                "scientificname": scientific_name,
                "decimalLatitude": f"{lat_min},{lat_max}",
                "decimalLongitude": f"{lon_min},{lon_max}",
                "size": size,
                "start": start,
            }
        )
        url = f"https://api.obis.org/v3/occurrence?{query}"
        try:
            data = fetch_json(url)
        except Exception:
            break
        results = data.get("results") or []
        if not results:
            break
        for r in results:
            la = r.get("decimalLatitude")
            lo = r.get("decimalLongitude")
            if la is None or lo is None:
                continue
            try:
                la_f = float(la)
                lo_f = float(lo)
            except Exception:
                continue
            if not (lat_min <= la_f <= lat_max and lon_min <= lo_f <= lon_max):
                continue
            rows.append({"lat": la_f, "lon": lo_f})
        if len(results) < size:
            break
        start += size
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def support_from_points(lat: float, lon: float, points: pd.DataFrame) -> tuple[float, int, float | None]:
    if points.empty:
        return 0.0, 0, None
    d = haversine_km(lat, lon, points["lat"].to_numpy(dtype=np.float64), points["lon"].to_numpy(dtype=np.float64))
    nearest = float(np.min(d))
    local_count = int(np.sum(d <= 75.0))
    nearest_term = math.exp(-nearest / 150.0)
    count_term = min(1.0, local_count / 10.0)
    support = float(np.clip(0.55 * nearest_term + 0.45 * count_term, 0.0, 1.0))
    return support, local_count, nearest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OBIS occurrence prior grid for species tie-break support.")
    p.add_argument("--lat_min", type=float, default=5.0)
    p.add_argument("--lat_max", type=float, default=25.0)
    p.add_argument("--lon_min", type=float, default=68.0)
    p.add_argument("--lon_max", type=float, default=94.0)
    p.add_argument("--grid_step", type=float, default=0.5)
    p.add_argument("--max_records_per_species", type=int, default=5000)
    p.add_argument(
        "--out_parquet",
        type=Path,
        default=Path("artifacts/priors/species_occurrence_prior_india.parquet"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lats = np.arange(args.lat_min, args.lat_max + 1e-9, args.grid_step, dtype=np.float64)
    lons = np.arange(args.lon_min, args.lon_max + 1e-9, args.grid_step, dtype=np.float64)

    species_points: dict[str, pd.DataFrame] = {}
    for species_id, sci_name in DEFAULT_SPECIES.items():
        print(f"[FETCH] {species_id} -> {sci_name}", flush=True)
        pts = fetch_obis_points(
            sci_name,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
            args.max_records_per_species,
        )
        species_points[species_id] = pts
        print(f"  points={len(pts)}", flush=True)

    rows = []
    total = len(DEFAULT_SPECIES) * len(lats) * len(lons)
    done = 0
    for species_id, points in species_points.items():
        for lat in lats:
            for lon in lons:
                support, local_count, nearest = support_from_points(float(lat), float(lon), points)
                rows.append(
                    {
                        "species_id": species_id,
                        "lat": round(float(lat), 4),
                        "lon": round(float(lon), 4),
                        "support_score": round(float(support), 6),
                        "local_count_75km": int(local_count),
                        "nearest_km": round(float(nearest), 4) if nearest is not None else None,
                    }
                )
                done += 1
        print(f"[GRID] {species_id} complete ({done}/{total})", flush=True)

    out_df = pd.DataFrame(rows)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(args.out_parquet, index=False)
        print(f"[SAVE] {args.out_parquet}", flush=True)
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}); saving csv fallback", flush=True)
        out_df.to_csv(args.out_parquet.with_suffix(".csv"), index=False)
        print(f"[SAVE] {args.out_parquet.with_suffix('.csv')}", flush=True)
    else:
        out_df.to_csv(args.out_parquet.with_suffix(".csv"), index=False)
        print(f"[SAVE] {args.out_parquet.with_suffix('.csv')} (fallback/inspect)", flush=True)


if __name__ == "__main__":
    main()
