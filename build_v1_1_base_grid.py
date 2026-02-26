import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from project_paths import RASTER_DIR, ensure_dirs, with_legacy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build west-expanded v1.1 base rasters on an isolated grid.")
    p.add_argument("--min_lon", type=float, default=77.9)
    p.add_argument("--max_lon", type=float, default=79.8)
    p.add_argument("--min_lat", type=float, default=8.4)
    p.add_argument("--max_lat", type=float, default=9.8)
    p.add_argument("--release_tag", type=str, default="v1_1")
    p.add_argument(
        "--fill_outside",
        choices=["nan", "nearest_edge"],
        default="nan",
        help="How to fill areas outside source coverage. nearest_edge is only for diagnostics and not scientifically recommended.",
    )
    return p.parse_args()


def reproject_to_target(
    src_path: Path,
    dst_path: Path,
    dst_transform,
    dst_width: int,
    dst_height: int,
    dst_crs,
    resampling: Resampling,
    fill_outside: str,
) -> float:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        out = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    nan_ratio = float(np.isnan(out).mean())
    if fill_outside == "nearest_edge" and np.isnan(out).any():
        # Conservative nearest fill for uncovered fringe; keep explicit warning via high nan_ratio in report.
        yy, xx = np.where(~np.isnan(out))
        if len(xx) > 0:
            from scipy.spatial import cKDTree  # local import to avoid hard dependency if unused

            tree = cKDTree(np.c_[yy, xx])
            y_nan, x_nan = np.where(np.isnan(out))
            _, idx = tree.query(np.c_[y_nan, x_nan], k=1)
            out[y_nan, x_nan] = out[yy[idx], xx[idx]]

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out, 1)
    return nan_ratio


def main() -> None:
    ensure_dirs()
    args = parse_args()
    tag = args.release_tag.strip()
    if not tag:
        raise ValueError("release_tag is required")

    depth_src = with_legacy(RASTER_DIR / "Depth.tif", "Depth.tif")
    slope_src = with_legacy(RASTER_DIR / "Slope (1).tif", "Slope (1).tif")
    dist_src = with_legacy(RASTER_DIR / "DistanceToShore.tif", "DistanceToShore.tif")
    shallow_src = with_legacy(RASTER_DIR / "ShallowSuitabilityMask.tif", "ShallowSuitabilityMask.tif")

    with rasterio.open(depth_src) as ds:
        xres = abs(ds.transform.a)
        yres = abs(ds.transform.e)
        crs = ds.crs

    width = int(np.ceil((args.max_lon - args.min_lon) / xres))
    height = int(np.ceil((args.max_lat - args.min_lat) / yres))
    transform = from_origin(args.min_lon, args.max_lat, xres, yres)

    out_depth = RASTER_DIR / f"Depth_{tag}.tif"
    out_slope = RASTER_DIR / f"Slope_{tag}.tif"
    out_dist = RASTER_DIR / f"DistanceToShore_{tag}.tif"
    out_shallow = RASTER_DIR / f"ShallowSuitabilityMask_{tag}.tif"

    stats = {}
    stats["depth_nan_ratio"] = reproject_to_target(
        depth_src, out_depth, transform, width, height, crs, Resampling.bilinear, args.fill_outside
    )
    stats["slope_nan_ratio"] = reproject_to_target(
        slope_src, out_slope, transform, width, height, crs, Resampling.bilinear, args.fill_outside
    )
    stats["distance_nan_ratio"] = reproject_to_target(
        dist_src, out_dist, transform, width, height, crs, Resampling.bilinear, args.fill_outside
    )
    stats["shallow_nan_ratio"] = reproject_to_target(
        shallow_src, out_shallow, transform, width, height, crs, Resampling.nearest, args.fill_outside
    )

    print(f"Saved base rasters for {tag}:")
    print(f" - {out_depth.name}")
    print(f" - {out_slope.name}")
    print(f" - {out_dist.name}")
    print(f" - {out_shallow.name}")
    print(
        f"Grid: lon [{args.min_lon}, {args.max_lon}] lat [{args.min_lat}, {args.max_lat}] "
        f"res ~({xres:.6f}, {yres:.6f}) size={width}x{height}"
    )
    print("NaN ratios:", stats)
    if any(v > 0.0 for v in stats.values()):
        print(
            "WARNING: Source coverage does not fully cover target bbox. "
            "West fringe may be unsupported unless you provide broader source rasters."
        )


if __name__ == "__main__":
    main()

