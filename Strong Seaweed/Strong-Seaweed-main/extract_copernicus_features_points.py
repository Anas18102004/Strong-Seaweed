import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr


def nearest_point_features(df: pd.DataFrame, phys_path: Path, wav_path: Path) -> pd.DataFrame:
    phys = xr.open_dataset(phys_path)
    wav = xr.open_dataset(wav_path)
    try:
        so = phys["so"].mean(dim="depth", skipna=True) if "depth" in phys["so"].dims else phys["so"]
        uo = phys["uo"].mean(dim="depth", skipna=True) if "depth" in phys["uo"].dims else phys["uo"]
        vo = phys["vo"].mean(dim="depth", skipna=True) if "depth" in phys["vo"].dims else phys["vo"]

        so_mean = so.mean(dim="time", skipna=True)
        so_std = so.std(dim="time", skipna=True)
        current_mean = np.sqrt(uo**2 + vo**2).mean(dim="time", skipna=True)

        wave = wav["VHM0"]
        wave_mean = wave.mean(dim="time", skipna=True)
        wave_std = wave.std(dim="time", skipna=True)

        out_rows = []
        for _, r in df.iterrows():
            lat = float(r["lat"])
            lon = float(r["lon"])

            so_m = float(so_mean.sel(latitude=lat, longitude=lon, method="nearest").values)
            so_s = float(so_std.sel(latitude=lat, longitude=lon, method="nearest").values)
            cur_m = float(current_mean.sel(latitude=lat, longitude=lon, method="nearest").values)
            wav_m = float(wave_mean.sel(latitude=lat, longitude=lon, method="nearest").values)
            wav_s = float(wave_std.sel(latitude=lat, longitude=lon, method="nearest").values)

            out_rows.append(
                {
                    **r.to_dict(),
                    "cop_so_mean": so_m,
                    "cop_so_std": so_s,
                    "cop_current_mean": cur_m,
                    "cop_wave_mean": wav_m,
                    "cop_wave_std": wav_s,
                }
            )

        return pd.DataFrame(out_rows)
    finally:
        phys.close()
        wav.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--physics_nc", default="data/netcdf/gulf_physics_2018_2025.nc")
    p.add_argument("--waves_nc", default="data/netcdf/gulf_waves_2018_2025.nc")
    args = p.parse_args()

    inp = Path(args.input_csv)
    out = Path(args.output_csv)
    phys = Path(args.physics_nc)
    wav = Path(args.waves_nc)

    df = pd.read_csv(inp)
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("input_csv must include lat, lon columns")

    feat = nearest_point_features(df, phys, wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
