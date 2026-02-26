import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
import argparse
from pathlib import Path
from project_paths import RASTER_DIR, TABULAR_DIR, ensure_dirs, with_legacy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cyclone exposure index aligned to a depth grid.")
    p.add_argument("--release_tag", type=str, default="", help="Optional tag suffix for output file, e.g. v1_1")
    p.add_argument("--depth_tif", type=str, default="", help="Optional depth raster path/name to align against.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    depth_path = Path(args.depth_tif) if args.depth_tif else with_legacy(RASTER_DIR / "Depth.tif", "Depth.tif")
    csv_path = with_legacy(TABULAR_DIR / "ibtracs.NI.list.v04r01.csv", "ibtracs.NI.list.v04r01.csv")
    suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""
    output_path = RASTER_DIR / f"Cyclone_Exposure_Index{suffix}.tif"

    # 1) Load master grid
    with rasterio.open(depth_path) as ds:
        transform = ds.transform
        width = ds.width
        height = ds.height
        profile = ds.profile.copy()

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    lon_grid = np.array(xs)
    lat_grid = np.array(ys)

    lon_flat = lon_grid.ravel().astype(np.float64)
    lat_flat = lat_grid.ravel().astype(np.float64)

    # 2) Load and filter cyclone data
    df = pd.read_csv(
        csv_path,
        usecols=["SEASON", "LAT", "LON", "USA_WIND"],
        low_memory=False,
        skiprows=[1],
    )

    for col in ["SEASON", "LAT", "LON", "USA_WIND"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SEASON", "LAT", "LON", "USA_WIND"])
    df = df[(df["SEASON"] >= 2000) & (df["SEASON"] <= 2023)]
    df = df[
        (df["LAT"] >= 6)
        & (df["LAT"] <= 15)
        & (df["LON"] >= 75)
        & (df["LON"] <= 85)
    ]

    if df.empty:
        raise RuntimeError("No cyclone points remain after filtering; check filters/CSV.")

    storm_lats = df["LAT"].to_numpy(dtype=np.float64)
    storm_lons = df["LON"].to_numpy(dtype=np.float64)
    storm_winds = df["USA_WIND"].to_numpy(dtype=np.float64)

    print(f"Filtered storm points: {len(storm_lats)}")

    # 3) Compute exposure: sum(wind / (distance_km^2 + 1)) for points <= 200 km
    geod = Geod(ellps="WGS84")
    exposure = np.zeros(lon_flat.size, dtype=np.float64)

    # Reused arrays avoid per-iteration allocations
    lon_src = np.empty_like(lon_flat)
    lat_src = np.empty_like(lat_flat)

    for slat, slon, swind in zip(storm_lats, storm_lons, storm_winds):
        lon_src.fill(slon)
        lat_src.fill(slat)
        _, _, dist_m = geod.inv(lon_src, lat_src, lon_flat, lat_flat)
        dist_km = dist_m / 1000.0

        mask = dist_km <= 200.0
        if np.any(mask):
            exposure[mask] += swind / (dist_km[mask] ** 2 + 1.0)

    exposure_grid = exposure.reshape(height, width)

    # 4) Save aligned GeoTIFF
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(exposure_grid.astype(np.float32), 1)

    print(f"Saved: {output_path}")
    print(
        "Exposure stats -> min: {:.6f}, max: {:.6f}, mean: {:.6f}".format(
            float(np.nanmin(exposure_grid)),
            float(np.nanmax(exposure_grid)),
            float(np.nanmean(exposure_grid)),
        )
    )


if __name__ == "__main__":
    main()
