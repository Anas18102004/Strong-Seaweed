import argparse
from pathlib import Path

import numpy as np
import rasterio
import requests
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from rasterio.transform import from_origin

from project_paths import NETCDF_DIR, RASTER_DIR, ensure_dirs, with_legacy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build v1.1 physical base rasters (Depth/Slope/Distance/ShallowMask) from ETOPO."
    )
    p.add_argument("--min_lon", type=float, default=77.9)
    p.add_argument("--max_lon", type=float, default=79.8)
    p.add_argument("--min_lat", type=float, default=8.4)
    p.add_argument("--max_lat", type=float, default=9.8)
    p.add_argument("--release_tag", type=str, default="v1_1")
    p.add_argument(
        "--etopo_url",
        type=str,
        default="https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo360.nc",
    )
    return p.parse_args()


def download_etopo_subset(url_base: str, out_nc: Path, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> None:
    q = (
        f"{url_base}?altitude[({max_lat}):1:({min_lat})][({min_lon}):1:({max_lon})]"
    )
    r = requests.get(q, timeout=60)
    r.raise_for_status()
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out_nc.write_bytes(r.content)


def target_grid_from_depth_template(min_lon: float, max_lon: float, min_lat: float, max_lat: float):
    depth_tpl = with_legacy(RASTER_DIR / "Depth.tif", "Depth.tif")
    with rasterio.open(depth_tpl) as ds:
        res_x = float(abs(ds.transform.a))
        res_y = float(abs(ds.transform.e))
        crs = ds.crs

    width = int(round((max_lon - min_lon) / res_x))
    height = int(round((max_lat - min_lat) / res_y))
    transform = from_origin(min_lon, max_lat, res_x, res_y)

    lon = min_lon + (np.arange(width) + 0.5) * res_x
    lat = max_lat - (np.arange(height) + 0.5) * res_y
    return lon, lat, transform, width, height, crs, res_x, res_y


def compute_slope_deg(depth: np.ndarray, lat_vec: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    dy_m = res_y * 111_320.0
    dz_dy = np.gradient(depth, axis=0) / dy_m

    dx_row_m = res_x * 111_320.0 * np.cos(np.deg2rad(lat_vec))
    dz_dx = np.zeros_like(depth, dtype=np.float64)
    for i in range(depth.shape[0]):
        dx = max(float(dx_row_m[i]), 1.0)
        dz_dx[i, :] = np.gradient(depth[i, :]) / dx

    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    slope[~np.isfinite(depth)] = np.nan
    return slope.astype(np.float32)


def compute_distance_to_shore(depth: np.ndarray, lon_vec: np.ndarray, lat_vec: np.ndarray) -> np.ndarray:
    water = depth > 0
    yy, xx = np.meshgrid(lat_vec, lon_vec, indexing="ij")
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x_m, y_m = tr.transform(xx.ravel(), yy.ravel())
    pts = np.c_[x_m, y_m]

    land_mask_flat = (~water).ravel()
    water_mask_flat = water.ravel()
    dist = np.zeros(depth.size, dtype=np.float32)

    if np.any(land_mask_flat) and np.any(water_mask_flat):
        land_pts = pts[land_mask_flat]
        water_pts = pts[water_mask_flat]
        tree = cKDTree(land_pts)
        d, _ = tree.query(water_pts, k=1)
        dist[water_mask_flat] = d.astype(np.float32)

    return dist.reshape(depth.shape)


def write_raster(path: Path, arr: np.ndarray, transform, crs) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    tag = args.release_tag.strip()
    if not tag:
        raise ValueError("--release_tag is required")

    etopo_nc = NETCDF_DIR / f"etopo_subset_{tag}.nc"
    print("Downloading ETOPO subset...")
    download_etopo_subset(
        args.etopo_url, etopo_nc, args.min_lon, args.max_lon, args.min_lat, args.max_lat
    )

    lon_t, lat_t, transform, width, height, crs, res_x, res_y = target_grid_from_depth_template(
        args.min_lon, args.max_lon, args.min_lat, args.max_lat
    )

    ds = xr.open_dataset(etopo_nc)
    try:
        # ETOPO uses altitude where ocean is negative and land positive.
        alt = ds["altitude"].interp(longitude=lon_t, latitude=lat_t, method="linear")
        alt_arr = np.asarray(alt.values, dtype=np.float32)
    finally:
        ds.close()

    depth = np.maximum(-alt_arr, 0.0).astype(np.float32)
    slope = compute_slope_deg(depth, lat_t, res_x, res_y)
    distance = compute_distance_to_shore(depth, lon_t, lat_t).astype(np.float32)
    shallow = ((depth > 0) & (depth < 5.0) & (distance <= 5000.0)).astype(np.float32)

    out_depth = RASTER_DIR / f"Depth_{tag}.tif"
    out_slope = RASTER_DIR / f"Slope_{tag}.tif"
    out_dist = RASTER_DIR / f"DistanceToShore_{tag}.tif"
    out_shallow = RASTER_DIR / f"ShallowSuitabilityMask_{tag}.tif"

    write_raster(out_depth, depth, transform, crs)
    write_raster(out_slope, slope, transform, crs)
    write_raster(out_dist, distance, transform, crs)
    write_raster(out_shallow, shallow, transform, crs)

    print("Saved:")
    print(f" - {out_depth}")
    print(f" - {out_slope}")
    print(f" - {out_dist}")
    print(f" - {out_shallow}")
    print(f"Grid: width={width}, height={height}, res~({res_x:.6f}, {res_y:.6f})")
    print(
        "Depth stats: min={:.3f} max={:.3f} mean={:.3f}".format(
            float(np.nanmin(depth)), float(np.nanmax(depth)), float(np.nanmean(depth))
        )
    )


if __name__ == "__main__":
    main()

