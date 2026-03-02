import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import NETCDF_DIR, REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-species point datasets from India-wide Copernicus features.")
    p.add_argument("--occ_csv", type=Path, default=TABULAR_DIR / "multispecies_occurrences_multispecies_india_v2.csv")
    p.add_argument("--physics_nc", type=Path, default=NETCDF_DIR / "india_physics_2025w01.nc")
    p.add_argument("--waves_nc", type=Path, default=NETCDF_DIR / "india_waves_2025w01.nc")
    p.add_argument("--bg_ratio", type=int, default=5)
    p.add_argument("--min_neg_abs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", type=str, default="multispecies_cop_india_v1")
    return p.parse_args()


def build_feature_grids(physics_nc: Path, waves_nc: Path):
    phys = xr.open_dataset(physics_nc)
    wav = xr.open_dataset(waves_nc)
    try:
        so = phys["so"].mean(dim="depth", skipna=True) if "depth" in phys["so"].dims else phys["so"]
        uo = phys["uo"].mean(dim="depth", skipna=True) if "depth" in phys["uo"].dims else phys["uo"]
        vo = phys["vo"].mean(dim="depth", skipna=True) if "depth" in phys["vo"].dims else phys["vo"]
        current = np.sqrt(uo**2 + vo**2)

        so_mean = so.mean(dim="time", skipna=True)
        so_std = so.std(dim="time", skipna=True)
        current_mean = current.mean(dim="time", skipna=True)
        wave_mean_raw = wav["VHM0"].mean(dim="time", skipna=True)
        wave_std_raw = wav["VHM0"].std(dim="time", skipna=True)
        # Align wave grid to physics grid for consistent flattened feature table.
        wave_mean = wave_mean_raw.interp(
            longitude=so_mean["longitude"], latitude=so_mean["latitude"], method="nearest"
        )
        wave_std = wave_std_raw.interp(
            longitude=so_mean["longitude"], latitude=so_mean["latitude"], method="nearest"
        )

        grids = {
            "so_mean": so_mean,
            "so_std": so_std,
            "current_mean": current_mean,
            "wave_mean": wave_mean,
            "wave_std": wave_std,
        }
        return grids
    finally:
        phys.close()
        wav.close()


def extract_features_for_points(points: pd.DataFrame, grids: dict) -> pd.DataFrame:
    rows = []
    for r in points.itertuples(index=False):
        lat = float(r.lat)
        lon = float(r.lon)
        feat = {"lon": lon, "lat": lat}
        ok = True
        for k, da in grids.items():
            v = da.sel(latitude=lat, longitude=lon, method="nearest").values
            val = float(v)
            feat[k] = val
            if not np.isfinite(val):
                ok = False
        if ok:
            rows.append(feat)
    return pd.DataFrame(rows)


def grid_points_dataframe(grids: dict) -> pd.DataFrame:
    ref = grids["so_mean"]
    lon2d, lat2d = np.meshgrid(ref["longitude"].values, ref["latitude"].values)
    out = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel()})
    for k, da in grids.items():
        out[k] = np.asarray(da.values).ravel()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def sample_negatives(pool: pd.DataFrame, pos: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if n_target <= 0 or pool.empty:
        return pool.iloc[0:0].copy()
    # Exclude exact coordinate overlap with positives.
    pos_keys = set(zip(pos["lon"].round(4), pos["lat"].round(4)))
    mask = ~pool[["lon", "lat"]].round(4).apply(tuple, axis=1).isin(pos_keys)
    pool = pool[mask].copy()
    if pool.empty:
        return pool
    n = min(n_target, len(pool))
    idx = np.random.default_rng(seed).choice(pool.index.to_numpy(), size=n, replace=False)
    return pool.loc[idx].copy().reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.occ_csv.exists():
        raise FileNotFoundError(f"Missing occurrence CSV: {args.occ_csv}")
    if not args.physics_nc.exists() or not args.waves_nc.exists():
        raise FileNotFoundError(f"Missing Copernicus files: {args.physics_nc} / {args.waves_nc}")

    occ = pd.read_csv(args.occ_csv)
    if "species_id" not in occ.columns:
        raise ValueError("occ_csv missing species_id")
    occ["lon"] = pd.to_numeric(occ["lon"], errors="coerce")
    occ["lat"] = pd.to_numeric(occ["lat"], errors="coerce")
    occ = occ.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["species_id", "lon", "lat"]).reset_index(drop=True)

    grids = build_feature_grids(args.physics_nc, args.waves_nc)
    grid_pool = grid_points_dataframe(grids)

    species_reports = []
    for species_id in sorted(occ["species_id"].unique()):
        sub = occ[occ["species_id"] == species_id][["lon", "lat"]].drop_duplicates().reset_index(drop=True)
        pos_feat = extract_features_for_points(sub, grids)
        n_pos = len(pos_feat)
        n_neg_target = max(int(n_pos * int(args.bg_ratio)), int(args.min_neg_abs))
        neg_feat = sample_negatives(grid_pool, pos_feat[["lon", "lat"]] if n_pos else pos_feat, n_neg_target, args.seed)

        if n_pos:
            pos_feat["label"] = 1
            pos_feat["label_weight"] = 1.0
            pos_feat["sample_type"] = "occurrence_positive"
        if len(neg_feat):
            neg_feat["label"] = 0
            neg_feat["label_weight"] = 1.0
            neg_feat["sample_type"] = "grid_negative"

        ds = pd.concat([pos_feat, neg_feat], ignore_index=True)
        ds["species_id"] = species_id
        ds = ds.drop_duplicates(subset=["lon", "lat", "label"]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        out_csv = TABULAR_DIR / f"training_dataset_{species_id}_{args.out_prefix}.csv"
        ds.to_csv(out_csv, index=False)
        species_reports.append(
            {
                "species_id": species_id,
                "occ_points": int(len(sub)),
                "positive_rows": int(n_pos),
                "negative_rows": int(len(neg_feat)),
                "total_rows": int(len(ds)),
                "output_csv": str(out_csv),
            }
        )
        print(f"[OK] {species_id}: pos={n_pos} neg={len(neg_feat)} total={len(ds)}")

    report = {
        "occ_csv": str(args.occ_csv),
        "physics_nc": str(args.physics_nc),
        "waves_nc": str(args.waves_nc),
        "grid_pool_rows": int(len(grid_pool)),
        "species_reports": species_reports,
    }
    report_out = REPORTS_DIR / f"multispecies_cop_point_dataset_report_{args.out_prefix}.json"
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Report: {report_out}")


if __name__ == "__main__":
    main()
