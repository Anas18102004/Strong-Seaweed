import numpy as np
import rasterio
from pathlib import Path
import argparse
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from project_paths import RASTER_DIR, ensure_dirs, with_legacy


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    if np.all(mask):
        return arr
    _, idx = ndimage.distance_transform_edt(mask, return_indices=True)
    out = arr.copy()
    out[mask] = arr[tuple(i[mask] for i in idx)]
    return out


def align_one(
    src_path: Path,
    ref_path: Path,
    dst_path: Path,
    resampling: Resampling,
    fill_nodata_nearest: bool = False,
) -> None:
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width
        dst_profile = ref.profile.copy()

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata
        dst_arr = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    if fill_nodata_nearest:
        dst_arr = fill_nan_nearest(dst_arr)

    dst_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress='lzw')
    with rasterio.open(dst_path, 'w', **dst_profile) as dst:
        dst.write(dst_arr.astype(np.float32), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align selected rasters to a depth reference grid.")
    p.add_argument("--release_tag", type=str, default="", help="Optional tag suffix for output names.")
    p.add_argument("--ref_depth", type=str, default="", help="Optional depth raster filename/path to use as reference.")
    p.add_argument(
        "--fill_nodata_nearest",
        action="store_true",
        help="Fill NaN cells after reprojection with nearest valid cell (for coverage completion).",
    )
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    ref = Path(args.ref_depth) if args.ref_depth else with_legacy(RASTER_DIR / 'Depth.tif', 'Depth.tif')
    suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""

    # source_name, output_name, resampling
    tasks = [
        ('Rain_Mean (1).tif', f'Rain_Mean_aligned{suffix}.tif' if suffix else 'Rain_Mean_aligned.tif', Resampling.bilinear),
        ('Rain_StdDev.tif', f'Rain_StdDev_aligned{suffix}.tif' if suffix else 'Rain_StdDev_aligned.tif', Resampling.bilinear),
        ('Extreme_Rain_Count.tif', f'Extreme_Rain_Count_aligned{suffix}.tif' if suffix else 'Extreme_Rain_Count_aligned.tif', Resampling.nearest),
        ('Wind_Mean.tif', f'Wind_Mean_aligned{suffix}.tif' if suffix else 'Wind_Mean_aligned.tif', Resampling.bilinear),
        ('Wind_StdDev.tif', f'Wind_StdDev_aligned{suffix}.tif' if suffix else 'Wind_StdDev_aligned.tif', Resampling.bilinear),
        ('GulfMannar_Chl_Mean_2018_2023.tif', f'Chl_Mean_aligned{suffix}.tif' if suffix else 'Chl_Mean_aligned.tif', Resampling.bilinear),
        ('GulfMannar_Chl_StdDev_2018_2023.tif', f'Chl_StdDev_aligned{suffix}.tif' if suffix else 'Chl_StdDev_aligned.tif', Resampling.bilinear),
        ('GulfMannar_Turbidity_Mean_2018_2023.tif', f'Turbidity_Mean_aligned{suffix}.tif' if suffix else 'Turbidity_Mean_aligned.tif', Resampling.bilinear),
        ('GulfMannar_Turbidity_StdDev_2018_2023.tif', f'Turbidity_StdDev_aligned{suffix}.tif' if suffix else 'Turbidity_StdDev_aligned.tif', Resampling.bilinear),
    ]

    for src_name, out_name, method in tasks:
        src = with_legacy(RASTER_DIR / src_name, src_name)
        dst = RASTER_DIR / out_name
        align_one(src, ref, dst, method, fill_nodata_nearest=args.fill_nodata_nearest)
        print(f'Aligned: {src_name} -> {out_name}')


if __name__ == '__main__':
    main()
