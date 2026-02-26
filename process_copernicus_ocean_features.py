from pathlib import Path
import argparse

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr
from project_paths import NETCDF_DIR, RASTER_DIR, ensure_dirs, with_legacy


PHYSICS_NC = with_legacy(NETCDF_DIR / "gulf_physics_2018_2023.nc", "gulf_physics_2018_2023.nc")
WAVES_NC = with_legacy(NETCDF_DIR / "gulf_waves_2018_2023.nc", "gulf_waves_2018_2023.nc")
SST_NC = with_legacy(NETCDF_DIR / "gulf_sst_2018_2023.nc", "gulf_sst_2018_2023.nc")
SO_ANFC_NC = with_legacy(NETCDF_DIR / "gulf_so_2024_2025_anfc.nc", "gulf_so_2024_2025_anfc.nc")
CUR_ANFC_NC = with_legacy(NETCDF_DIR / "gulf_cur_2024_2025_anfc.nc", "gulf_cur_2024_2025_anfc.nc")
SST_ANFC_NC = with_legacy(NETCDF_DIR / "gulf_thetao_2024_2025_anfc.nc", "gulf_thetao_2024_2025_anfc.nc")
WAV_ANFC_NC = with_legacy(NETCDF_DIR / "gulf_vhm0_2024_2025_anfc.nc", "gulf_vhm0_2024_2025_anfc.nc")
DEPTH_TIF = with_legacy(RASTER_DIR / "Depth.tif", "Depth.tif")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process Copernicus ocean features and align to depth grid.")
    p.add_argument("--release_tag", type=str, default="", help="Optional tag to suffix output rasters, e.g. v1_1")
    p.add_argument(
        "--depth_tif",
        type=str,
        default="",
        help="Optional depth raster path to use as alignment reference (e.g., data/rasters/Depth_v1_1.tif).",
    )
    return p.parse_args()


def choose_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def to_2d(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def concat_time_unique(arrays: list[xr.DataArray]) -> xr.DataArray:
    merged = xr.concat(arrays, dim="time", join="outer").sortby("time")
    time_vals = np.asarray(merged["time"].values)
    _, unique_idx = np.unique(time_vals, return_index=True)
    unique_idx.sort()
    return merged.isel(time=unique_idx)


def build_src_transform(lat: np.ndarray, lon: np.ndarray) -> tuple:
    dlon = float(np.mean(np.diff(lon)))
    dlat = float(np.mean(np.diff(lat)))

    lat_descending = lat[0] > lat[-1]
    top_lat = float(lat[0] if lat_descending else lat[-1])

    transform = from_origin(
        west=float(np.min(lon) - abs(dlon) / 2.0),
        north=float(top_lat + abs(dlat) / 2.0),
        xsize=float(abs(dlon)),
        ysize=float(abs(dlat)),
    )
    return transform, lat_descending


def orient_north_up(arr: np.ndarray, lat_descending: bool) -> np.ndarray:
    return arr if lat_descending else np.flipud(arr)


def reproject_to_depth_grid(
    src_arr: np.ndarray,
    src_transform,
    dst_profile: dict,
    method: Resampling,
) -> np.ndarray:
    dst_arr = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=method,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst_arr


def save_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    depth_ref = Path(args.depth_tif) if args.depth_tif else DEPTH_TIF
    with rasterio.open(depth_ref) as ref:
        dst_profile = ref.profile.copy()

    physics_path = choose_existing(with_legacy(NETCDF_DIR / "gulf_physics_2018_2025.nc", "gulf_physics_2018_2025.nc"), PHYSICS_NC)
    waves_path = choose_existing(with_legacy(NETCDF_DIR / "gulf_waves_2018_2025.nc", "gulf_waves_2018_2025.nc"), WAVES_NC)
    physics = xr.open_dataset(physics_path)
    waves = xr.open_dataset(waves_path)
    sst_ds = xr.open_dataset(SST_NC)
    so_anfc = xr.open_dataset(SO_ANFC_NC) if SO_ANFC_NC.exists() else None
    cur_anfc = xr.open_dataset(CUR_ANFC_NC) if CUR_ANFC_NC.exists() else None
    sst_anfc = xr.open_dataset(SST_ANFC_NC) if SST_ANFC_NC.exists() else None
    wav_anfc = xr.open_dataset(WAV_ANFC_NC) if WAV_ANFC_NC.exists() else None

    try:
        # Use shallow-water averaged (0-~5 m) fields over time and merge extra ANFC data if present.
        sal_series = [physics["so"].mean(dim="depth", skipna=True)]
        u_series = [physics["uo"].mean(dim="depth", skipna=True)]
        v_series = [physics["vo"].mean(dim="depth", skipna=True)]

        if so_anfc is not None:
            sal_series.append(so_anfc["so"].mean(dim="depth", skipna=True))
        if cur_anfc is not None:
            u_series.append(cur_anfc["uo"].mean(dim="depth", skipna=True))
            v_series.append(cur_anfc["vo"].mean(dim="depth", skipna=True))

        sal = concat_time_unique(sal_series)
        u = concat_time_unique(u_series)
        v = concat_time_unique(v_series)
        current_speed = np.sqrt(u**2 + v**2)

        sal_mean = sal.mean(dim="time", skipna=True)
        sal_std = sal.std(dim="time", skipna=True)
        sal_shock_days = (sal < 25.0).sum(dim="time")
        current_mean = current_speed.mean(dim="time", skipna=True)

        wave_series = [waves["VHM0"]]
        if wav_anfc is not None:
            wave_series.append(wav_anfc["VHM0"])
        wave = concat_time_unique(wave_series)
        wave_mean = wave.mean(dim="time", skipna=True)
        wave_std = wave.std(dim="time", skipna=True)
        extreme_wave_days = (wave > 2.5).sum(dim="time")

        # SST features from near-surface temperature (thetao), averaged over 0-~5 m.
        sst_series = [sst_ds["thetao"].mean(dim="depth", skipna=True)]
        if sst_anfc is not None:
            sst_series.append(sst_anfc["thetao"].mean(dim="depth", skipna=True))
        sst = concat_time_unique(sst_series)
        sst_mean = sst.mean(dim="time", skipna=True)
        sst_var = sst.var(dim="time", skipna=True)
        heat_stress_freq = (sst > 30.0).sum(dim="time")

        phys_lat = sal_mean["latitude"].values
        phys_lon = sal_mean["longitude"].values
        wave_lat = wave_mean["latitude"].values
        wave_lon = wave_mean["longitude"].values
        sst_lat = sst_mean["latitude"].values
        sst_lon = sst_mean["longitude"].values

        phys_tx, phys_desc = build_src_transform(phys_lat, phys_lon)
        wave_tx, wave_desc = build_src_transform(wave_lat, wave_lon)
        sst_tx, sst_desc = build_src_transform(sst_lat, sst_lon)

        feature_specs = [
            ("sal_mean", sal_mean, phys_tx, phys_desc, Resampling.bilinear),
            ("sal_std", sal_std, phys_tx, phys_desc, Resampling.bilinear),
            ("sal_shock_days", sal_shock_days, phys_tx, phys_desc, Resampling.nearest),
            ("current_mean", current_mean, phys_tx, phys_desc, Resampling.bilinear),
            ("wave_mean", wave_mean, wave_tx, wave_desc, Resampling.bilinear),
            ("wave_std", wave_std, wave_tx, wave_desc, Resampling.bilinear),
            ("extreme_wave_days", extreme_wave_days, wave_tx, wave_desc, Resampling.nearest),
            ("sst_mean", sst_mean, sst_tx, sst_desc, Resampling.bilinear),
            ("sst_var", sst_var, sst_tx, sst_desc, Resampling.bilinear),
            ("heat_stress_freq", heat_stress_freq, sst_tx, sst_desc, Resampling.nearest),
        ]

        for name, da, src_tx, lat_desc, method in feature_specs:
            src_arr = orient_north_up(to_2d(da), lat_desc)
            out_arr = reproject_to_depth_grid(src_arr, src_tx, dst_profile, method)
            suffix = f"_{args.release_tag.strip()}" if args.release_tag.strip() else ""
            out_path = RASTER_DIR / f"{name}_aligned{suffix}.tif"
            save_raster(out_path, out_arr, dst_profile)
            print(
                f"Saved {out_path.name} | min={float(np.nanmin(out_arr)):.4f} "
                f"max={float(np.nanmax(out_arr)):.4f} mean={float(np.nanmean(out_arr)):.4f}"
            )
    finally:
        physics.close()
        waves.close()
        sst_ds.close()
        if so_anfc is not None:
            so_anfc.close()
        if cur_anfc is not None:
            cur_anfc.close()
        if sst_anfc is not None:
            sst_anfc.close()
        if wav_anfc is not None:
            wav_anfc.close()


if __name__ == "__main__":
    main()
