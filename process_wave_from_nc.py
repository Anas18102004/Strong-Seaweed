import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr
from project_paths import NETCDF_DIR, RASTER_DIR, ensure_dirs, with_legacy


def main() -> None:
    ensure_dirs()
    nc_path = with_legacy(NETCDF_DIR / 'cmems_mod_glo_wav_my_0.2deg_PT3H-i_1771821824953.nc', 'cmems_mod_glo_wav_my_0.2deg_PT3H-i_1771821824953.nc')
    depth_path = with_legacy(RASTER_DIR / 'Depth.tif', 'Depth.tif')
    out_mean = RASTER_DIR / 'WaveHeight_Mean.tif'
    out_std = RASTER_DIR / 'WaveHeight_StdDev.tif'

    ds = xr.open_dataset(str(nc_path))
    if 'VHM0' not in ds:
        raise RuntimeError('Expected VHM0 variable in netCDF, but not found.')

    vhm0 = ds['VHM0']
    mean_da = vhm0.mean(dim='time', skipna=True)
    std_da = vhm0.std(dim='time', skipna=True)

    lat = ds['latitude'].values
    lon = ds['longitude'].values

    dlon = float(np.mean(np.diff(lon)))
    dlat = float(np.mean(np.diff(lat)))

    # Convert to north-up raster orientation (row 0 = max latitude)
    mean_arr = np.array(mean_da.values, dtype=np.float32)
    std_arr = np.array(std_da.values, dtype=np.float32)
    if lat[0] < lat[-1]:
        mean_arr = np.flipud(mean_arr)
        std_arr = np.flipud(std_arr)
        lat_for_transform_top = float(np.max(lat))
    else:
        lat_for_transform_top = float(lat[0])

    src_transform = from_origin(
        west=float(np.min(lon) - dlon / 2.0),
        north=float(lat_for_transform_top + abs(dlat) / 2.0),
        xsize=float(abs(dlon)),
        ysize=float(abs(dlat)),
    )

    with rasterio.open(str(depth_path)) as ref:
        dst_profile = ref.profile.copy()
        dst_transform = ref.transform
        dst_crs = ref.crs
        dst_h, dst_w = ref.height, ref.width

    def reproject_to_depth(src_arr: np.ndarray) -> np.ndarray:
        dst_arr = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs='EPSG:4326',
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return dst_arr

    mean_out = reproject_to_depth(mean_arr)
    std_out = reproject_to_depth(std_arr)

    out_profile = dst_profile
    out_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress='lzw')

    with rasterio.open(str(out_mean), 'w', **out_profile) as dst:
        dst.write(mean_out, 1)

    with rasterio.open(str(out_std), 'w', **out_profile) as dst:
        dst.write(std_out, 1)

    print('Saved:', out_mean)
    print('Saved:', out_std)
    print('Wave mean stats -> min {:.4f}, max {:.4f}, mean {:.4f}'.format(
        float(np.nanmin(mean_out)), float(np.nanmax(mean_out)), float(np.nanmean(mean_out))
    ))
    print('Wave std stats -> min {:.4f}, max {:.4f}, mean {:.4f}'.format(
        float(np.nanmin(std_out)), float(np.nanmax(std_out)), float(np.nanmean(std_out))
    ))


if __name__ == '__main__':
    main()
