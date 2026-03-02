import argparse
import os
from pathlib import Path

from copernicusmarine import login, subset
from project_paths import NETCDF_DIR, ensure_dirs

PHYSICS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
WAVE_DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Copernicus ocean datasets for Gulf of Mannar.")
    p.add_argument("--min_lon", type=float, default=78.5)
    p.add_argument("--max_lon", type=float, default=80.5)
    p.add_argument("--min_lat", type=float, default=8.0)
    p.add_argument("--max_lat", type=float, default=10.0)
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument(
        "--username",
        type=str,
        default=os.getenv("CMEMS_USERNAME", ""),
        help="Copernicus username (or set CMEMS_USERNAME env var).",
    )
    p.add_argument(
        "--password",
        type=str,
        default=os.getenv("CMEMS_PASSWORD", ""),
        help="Copernicus password (or set CMEMS_PASSWORD env var).",
    )
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.username or not args.password:
        raise ValueError("Provide --username/--password or set CMEMS_USERNAME/CMEMS_PASSWORD.")

    login(
        username=args.username,
        password=args.password,
        force_overwrite=True,
        check_credentials_valid=True,
    )

    # Ocean physics: salinity + currents (surface to 5 m).
    subset(
        dataset_id=PHYSICS_DATASET_ID,
        variables=["so", "uo", "vo"],
        minimum_longitude=args.min_lon,
        maximum_longitude=args.max_lon,
        minimum_latitude=args.min_lat,
        maximum_latitude=args.max_lat,
        start_datetime=args.start,
        end_datetime=args.end,
        minimum_depth=0,
        maximum_depth=5,
        output_filename=str(NETCDF_DIR / "gulf_physics_2018_2025.nc"),
    )

    # Waves: significant wave height.
    subset(
        dataset_id=WAVE_DATASET_ID,
        variables=["VHM0"],
        minimum_longitude=args.min_lon,
        maximum_longitude=args.max_lon,
        minimum_latitude=args.min_lat,
        maximum_latitude=args.max_lat,
        start_datetime=args.start,
        end_datetime=args.end,
        output_filename=str(NETCDF_DIR / "gulf_waves_2018_2025.nc"),
    )

    print("Downloaded Copernicus datasets:")
    print(" - gulf_physics_2018_2025.nc")
    print(" - gulf_waves_2018_2025.nc")
    print("Next: run process_copernicus_ocean_features.py, then build_feature_matrix.py")


if __name__ == "__main__":
    main()
