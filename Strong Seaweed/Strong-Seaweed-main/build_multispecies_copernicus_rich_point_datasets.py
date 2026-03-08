import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import NETCDF_DIR, REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-species rich Copernicus point datasets (India-wide).")
    p.add_argument("--occ_csv", type=Path, default=TABULAR_DIR / "multispecies_occurrences_multispecies_india_v2_prod.csv")
    p.add_argument("--physics_nc", type=Path, default=NETCDF_DIR / "india_physics_2025w01.nc")
    p.add_argument("--waves_nc", type=Path, default=NETCDF_DIR / "india_waves_2025w01.nc")
    p.add_argument("--bg_ratio", type=int, default=8)
    p.add_argument("--min_neg_abs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_sampling_method", type=str, default="region_balanced", choices=["uniform", "region_balanced"])
    p.add_argument("--neg_grid_deg", type=float, default=2.0)
    p.add_argument("--max_neg_per_cell", type=int, default=250)
    p.add_argument("--out_prefix", type=str, default="multispecies_cop_india_v3_rich")
    return p.parse_args()


def _q(da: xr.DataArray, q: float, dim: str = "time") -> xr.DataArray:
    out = da.quantile(q, dim=dim, skipna=True)
    if "quantile" in out.dims:
        out = out.squeeze("quantile", drop=True)
    return out


def _grad_mag_2d(da: xr.DataArray) -> xr.DataArray:
    arr = np.asarray(da.values, dtype=np.float64)
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx * gx + gy * gy)
    return xr.DataArray(mag, coords=da.coords, dims=da.dims)


def build_feature_grids(physics_nc: Path, waves_nc: Path) -> dict[str, xr.DataArray]:
    phys = xr.open_dataset(physics_nc)
    wav = xr.open_dataset(waves_nc)
    try:
        so = phys["so"].mean(dim="depth", skipna=True) if "depth" in phys["so"].dims else phys["so"]
        uo = phys["uo"].mean(dim="depth", skipna=True) if "depth" in phys["uo"].dims else phys["uo"]
        vo = phys["vo"].mean(dim="depth", skipna=True) if "depth" in phys["vo"].dims else phys["vo"]
        current = np.sqrt(uo**2 + vo**2)

        wave = wav["VHM0"].interp(longitude=so["longitude"], latitude=so["latitude"], method="nearest")

        grids: dict[str, xr.DataArray] = {}

        # Salinity summaries.
        grids["so_mean"] = so.mean(dim="time", skipna=True)
        grids["so_std"] = so.std(dim="time", skipna=True)
        grids["so_min"] = so.min(dim="time", skipna=True)
        grids["so_max"] = so.max(dim="time", skipna=True)
        grids["so_p10"] = _q(so, 0.10)
        grids["so_p90"] = _q(so, 0.90)
        grids["so_range"] = grids["so_max"] - grids["so_min"]
        grids["so_cv"] = grids["so_std"] / (np.abs(grids["so_mean"]) + 1e-6)

        # Current summaries.
        grids["uo_mean"] = uo.mean(dim="time", skipna=True)
        grids["vo_mean"] = vo.mean(dim="time", skipna=True)
        grids["current_mean"] = current.mean(dim="time", skipna=True)
        grids["current_std"] = current.std(dim="time", skipna=True)
        grids["current_max"] = current.max(dim="time", skipna=True)
        grids["current_p90"] = _q(current, 0.90)
        grids["current_cv"] = grids["current_std"] / (np.abs(grids["current_mean"]) + 1e-6)

        # Wave summaries.
        grids["wave_mean"] = wave.mean(dim="time", skipna=True)
        grids["wave_std"] = wave.std(dim="time", skipna=True)
        grids["wave_min"] = wave.min(dim="time", skipna=True)
        grids["wave_max"] = wave.max(dim="time", skipna=True)
        grids["wave_p90"] = _q(wave, 0.90)
        grids["wave_p95"] = _q(wave, 0.95)
        grids["wave_range"] = grids["wave_max"] - grids["wave_min"]
        grids["wave_cv"] = grids["wave_std"] / (np.abs(grids["wave_mean"]) + 1e-6)

        # Spatial gradients (heterogeneity proxy).
        grids["so_grad"] = _grad_mag_2d(grids["so_mean"])
        grids["current_grad"] = _grad_mag_2d(grids["current_mean"])
        grids["wave_grad"] = _grad_mag_2d(grids["wave_mean"])

        return grids
    finally:
        phys.close()
        wav.close()


def extract_features_for_points(points: pd.DataFrame, grids: dict[str, xr.DataArray]) -> pd.DataFrame:
    rows = []
    for r in points.itertuples(index=False):
        lat = float(r.lat)
        lon = float(r.lon)
        feat = {"lon": lon, "lat": lat}
        for k, da in grids.items():
            v = da.sel(latitude=lat, longitude=lon, method="nearest").values
            val = float(v)
            feat[k] = val
        rows.append(feat)
    return pd.DataFrame(rows)


def grid_points_dataframe(grids: dict[str, xr.DataArray]) -> pd.DataFrame:
    ref = grids["so_mean"]
    lon2d, lat2d = np.meshgrid(ref["longitude"].values, ref["latitude"].values)
    out = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel()})
    for k, da in grids.items():
        out[k] = np.asarray(da.values).ravel()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return out


def _exclude_positive_overlap(pool: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool.iloc[0:0].copy()
    if pos.empty:
        return pool.copy()
    pos_keys = set(zip(pos["lon"].round(4), pos["lat"].round(4)))
    mask = ~pool[["lon", "lat"]].round(4).apply(tuple, axis=1).isin(pos_keys)
    return pool[mask].copy()


def sample_negatives_uniform(pool: pd.DataFrame, pos: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if n_target <= 0 or pool.empty:
        return pool.iloc[0:0].copy()
    pool = _exclude_positive_overlap(pool, pos)
    if pool.empty:
        return pool
    n = min(n_target, len(pool))
    idx = np.random.default_rng(seed).choice(pool.index.to_numpy(), size=n, replace=False)
    return pool.loc[idx].copy().reset_index(drop=True)


def sample_negatives_region_balanced(
    pool: pd.DataFrame,
    pos: pd.DataFrame,
    n_target: int,
    seed: int,
    grid_deg: float,
    max_per_cell: int,
) -> pd.DataFrame:
    if n_target <= 0 or pool.empty:
        return pool.iloc[0:0].copy()
    pool = _exclude_positive_overlap(pool, pos)
    if pool.empty:
        return pool
    rng = np.random.default_rng(seed)
    deg = float(grid_deg) if float(grid_deg) > 0 else 2.0
    pool = pool.copy()
    pool["lon_cell"] = np.floor(pool["lon"] / deg).astype(int)
    pool["lat_cell"] = np.floor(pool["lat"] / deg).astype(int)
    pool["cell_id"] = pool["lon_cell"].astype(str) + "_" + pool["lat_cell"].astype(str)
    groups = list(pool.groupby("cell_id", sort=False))
    if not groups:
        return pool.iloc[0:0].copy()

    n_cells = max(1, len(groups))
    per_cell = int(np.ceil(float(n_target) / float(n_cells)))
    per_cell = max(1, min(per_cell, int(max_per_cell)))

    picked_idx: list[int] = []
    for _, g in groups:
        k = min(len(g), per_cell)
        if k <= 0:
            continue
        chosen = rng.choice(g.index.to_numpy(), size=k, replace=False)
        picked_idx.extend(chosen.tolist())

    if not picked_idx:
        return pool.iloc[0:0].copy()

    picked = pool.loc[picked_idx].copy()
    if len(picked) > n_target:
        keep_idx = rng.choice(picked.index.to_numpy(), size=n_target, replace=False)
        picked = picked.loc[keep_idx].copy()
    elif len(picked) < n_target:
        remaining = pool.drop(index=picked.index, errors="ignore")
        needed = min(n_target - len(picked), len(remaining))
        if needed > 0:
            extra_idx = rng.choice(remaining.index.to_numpy(), size=needed, replace=False)
            picked = pd.concat([picked, remaining.loc[extra_idx]], ignore_index=False)

    return (
        picked.drop(columns=["lon_cell", "lat_cell", "cell_id"], errors="ignore")
        .reset_index(drop=True)
    )


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
        sub_cols = ["lon", "lat"]
        if "label_weight" in occ.columns:
            sub_cols.append("label_weight")
        sub = occ[occ["species_id"] == species_id][sub_cols].copy()
        if "label_weight" in sub.columns:
            sub["label_weight"] = pd.to_numeric(sub["label_weight"], errors="coerce").fillna(1.0).clip(lower=0.2, upper=2.0)
            sub = (
                sub.groupby(["lon", "lat"], as_index=False)["label_weight"]
                .max()
                .reset_index(drop=True)
            )
        else:
            sub = sub.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
        pos_feat = extract_features_for_points(sub, grids)
        n_pos = len(pos_feat)
        n_neg_target = max(int(n_pos * int(args.bg_ratio)), int(args.min_neg_abs))
        pos_coords = pos_feat[["lon", "lat"]] if n_pos else pos_feat
        if args.negative_sampling_method == "region_balanced":
            neg_feat = sample_negatives_region_balanced(
                grid_pool,
                pos_coords,
                n_neg_target,
                args.seed,
                grid_deg=float(args.neg_grid_deg),
                max_per_cell=int(args.max_neg_per_cell),
            )
        else:
            neg_feat = sample_negatives_uniform(grid_pool, pos_coords, n_neg_target, args.seed)

        if n_pos:
            if "label_weight" in sub.columns:
                pos_feat = pos_feat.merge(sub[["lon", "lat", "label_weight"]], on=["lon", "lat"], how="left")
                pos_feat["label_weight"] = pd.to_numeric(pos_feat["label_weight"], errors="coerce").fillna(1.0).clip(lower=0.2, upper=2.0)
            else:
                pos_feat["label_weight"] = 1.0
            pos_feat["label"] = 1
            pos_feat["sample_type"] = "occurrence_positive"
        if len(neg_feat):
            neg_feat["label"] = 0
            neg_feat["label_weight"] = 1.0
            neg_feat["sample_type"] = "grid_negative"

        ds = pd.concat([pos_feat, neg_feat], ignore_index=True)
        ds["species_id"] = species_id
        # Keep all samples, then robust-impute numeric feature columns.
        num_cols = [c for c in ds.columns if c not in {"label", "label_weight", "sample_type", "species_id"} and pd.api.types.is_numeric_dtype(ds[c])]
        ds[num_cols] = ds[num_cols].replace([np.inf, -np.inf], np.nan)
        for c in num_cols:
            med = float(ds[c].median(skipna=True)) if ds[c].notna().any() else 0.0
            ds[c] = ds[c].fillna(med)
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
                "n_features": int(len([c for c in ds.columns if c not in {'label','label_weight','sample_type','species_id'}])),
                "negative_sampling_method": args.negative_sampling_method,
                "output_csv": str(out_csv),
            }
        )
        print(f"[OK] {species_id}: pos={n_pos} neg={len(neg_feat)} total={len(ds)}")

    report = {
        "occ_csv": str(args.occ_csv),
        "physics_nc": str(args.physics_nc),
        "waves_nc": str(args.waves_nc),
        "grid_pool_rows": int(len(grid_pool)),
        "feature_names": sorted([k for k in grids.keys()]),
        "species_reports": species_reports,
    }
    report_out = REPORTS_DIR / f"multispecies_cop_rich_point_dataset_report_{args.out_prefix}.json"
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Report: {report_out}")


if __name__ == "__main__":
    main()
