import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import argparse
from project_paths import RASTER_DIR, TABULAR_DIR, ensure_dirs


def load_flat(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32).ravel()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build master feature matrix from aligned rasters.")
    p.add_argument("--release_tag", type=str, default="", help="Optional tag to read tagged aligned rasters and write tagged matrix.")
    p.add_argument("--depth_raster", type=str, default="", help="Optional depth raster filename/path for tagged releases.")
    p.add_argument("--slope_raster", type=str, default="", help="Optional slope raster filename/path for tagged releases.")
    p.add_argument("--distance_raster", type=str, default="", help="Optional distance-to-shore raster filename/path for tagged releases.")
    p.add_argument("--shallow_raster", type=str, default="", help="Optional shallow mask raster filename/path for tagged releases.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""

    rasters = {
        "depth": args.depth_raster or ("Depth.tif" if not suffix else f"Depth{suffix}.tif"),
        "slope": args.slope_raster or ("Slope (1).tif" if not suffix else f"Slope{suffix}.tif"),
        "distance_to_shore": args.distance_raster or ("DistanceToShore.tif" if not suffix else f"DistanceToShore{suffix}.tif"),
        "shallow_mask": args.shallow_raster or ("ShallowSuitabilityMask.tif" if not suffix else f"ShallowSuitabilityMask{suffix}.tif"),
        "rain_mean": f"Rain_Mean_aligned{suffix}.tif" if suffix else "Rain_Mean_aligned.tif",
        "rain_std": f"Rain_StdDev_aligned{suffix}.tif" if suffix else "Rain_StdDev_aligned.tif",
        "extreme_rain": f"Extreme_Rain_Count_aligned{suffix}.tif" if suffix else "Extreme_Rain_Count_aligned.tif",
        "wind_mean": f"Wind_Mean_aligned{suffix}.tif" if suffix else "Wind_Mean_aligned.tif",
        "wind_std": f"Wind_StdDev_aligned{suffix}.tif" if suffix else "Wind_StdDev_aligned.tif",
        "chl_mean": f"Chl_Mean_aligned{suffix}.tif" if suffix else "Chl_Mean_aligned.tif",
        "chl_std": f"Chl_StdDev_aligned{suffix}.tif" if suffix else "Chl_StdDev_aligned.tif",
        "turb_mean": f"Turbidity_Mean_aligned{suffix}.tif" if suffix else "Turbidity_Mean_aligned.tif",
        "turb_std": f"Turbidity_StdDev_aligned{suffix}.tif" if suffix else "Turbidity_StdDev_aligned.tif",
        "cyclone": f"Cyclone_Exposure_Index{suffix}.tif" if suffix else "Cyclone_Exposure_Index.tif",
        "wave_mean": f"wave_mean_aligned{suffix}.tif" if suffix else "wave_mean_aligned.tif",
        "wave_std": f"wave_std_aligned{suffix}.tif" if suffix else "wave_std_aligned.tif",
        "sal_mean": f"sal_mean_aligned{suffix}.tif" if suffix else "sal_mean_aligned.tif",
        "sal_std": f"sal_std_aligned{suffix}.tif" if suffix else "sal_std_aligned.tif",
        "sal_shock_days": f"sal_shock_days_aligned{suffix}.tif" if suffix else "sal_shock_days_aligned.tif",
        "current_mean": f"current_mean_aligned{suffix}.tif" if suffix else "current_mean_aligned.tif",
        "extreme_wave_days": f"extreme_wave_days_aligned{suffix}.tif" if suffix else "extreme_wave_days_aligned.tif",
        "sst_mean": f"sst_mean_aligned{suffix}.tif" if suffix else "sst_mean_aligned.tif",
        "sst_var": f"sst_var_aligned{suffix}.tif" if suffix else "sst_var_aligned.tif",
        "heat_stress_freq": f"heat_stress_freq_aligned{suffix}.tif" if suffix else "heat_stress_freq_aligned.tif",
    }

    missing = [name for name, f in rasters.items() if not (RASTER_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raster files for: {missing}")

    # Validate all rasters match the depth grid exactly.
    ref_path = RASTER_DIR / rasters["depth"]
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

        rows, cols = np.meshgrid(np.arange(ref.height), np.arange(ref.width), indexing="ij")
        xs, ys = rasterio.transform.xy(ref_transform, rows, cols, offset="center")
        lon = np.array(xs, dtype=np.float64).ravel()
        lat = np.array(ys, dtype=np.float64).ravel()

    data = {"lon": lon, "lat": lat}

    for name, fname in rasters.items():
        path = RASTER_DIR / fname
        with rasterio.open(path) as src:
            same = (
                src.crs == ref_crs
                and src.transform == ref_transform
                and (src.height, src.width) == ref_shape
            )
            if not same:
                raise ValueError(
                    f"Raster misalignment for {name}: {fname}. "
                    f"Expected CRS/transform/shape from {rasters['depth']}."
                )
        data[name] = load_flat(path)

    df = pd.DataFrame(data)

    # Clean invalid numbers
    df = df.replace([np.inf, -np.inf], np.nan)

    # Use shallow mask to keep intended marine suitability domain where mask > 0.
    if "shallow_mask" in df.columns:
        df = df[df["shallow_mask"] > 0]

    # Keep ocean pixels only.
    df = df[df["depth"] > 0]

    # Remove rows with any missing feature values.
    feature_cols = [c for c in df.columns if c not in ["lon", "lat"]]
    df = df.dropna(subset=feature_cols)

    out_csv = TABULAR_DIR / (f"master_feature_matrix{suffix}.csv" if suffix else "master_feature_matrix.csv")
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Final dataset shape: {df.shape}")
    print("Columns:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
