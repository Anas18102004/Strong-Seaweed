import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import NETCDF_DIR, REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expand soft positives around hard positives using distance bands and ecological gating."
    )
    p.add_argument("--dataset_glob", type=str, default="training_dataset_*_multispecies_cop_india_v2_prod.csv")
    p.add_argument("--hard_positive_type", type=str, default="occurrence_positive")
    p.add_argument("--distance_bands_km", type=str, default="1.5:0.85,3.0:0.65,5.0:0.45")
    p.add_argument("--neg_buffer_km", type=float, default=8.0)
    p.add_argument("--q_low", type=float, default=0.10)
    p.add_argument("--q_high", type=float, default=0.90)
    p.add_argument("--min_similarity", type=float, default=0.55)
    p.add_argument("--min_soft_weight", type=float, default=0.35)
    p.add_argument("--max_soft_weight", type=float, default=0.85)
    p.add_argument("--use_copernicus_grid", action="store_true")
    p.add_argument("--physics_nc", type=Path, default=NETCDF_DIR / "india_physics_2025w01.nc")
    p.add_argument("--waves_nc", type=Path, default=NETCDF_DIR / "india_waves_2025w01.nc")
    p.add_argument("--max_candidates_per_species", type=int, default=20000)
    p.add_argument("--out_suffix", type=str, default="softpos_v1")
    return p.parse_args()


def parse_bands(spec: str) -> list[tuple[float, float]]:
    out = []
    for token in spec.split(","):
        d, w = token.strip().split(":")
        out.append((float(d), float(w)))
    out.sort(key=lambda x: x[0])
    return out


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_hp_distance_km(pool: pd.DataFrame, hp: pd.DataFrame) -> np.ndarray:
    p_lat = pool["lat"].to_numpy(dtype=np.float64)
    p_lon = pool["lon"].to_numpy(dtype=np.float64)
    h_lat = hp["lat"].to_numpy(dtype=np.float64)
    h_lon = hp["lon"].to_numpy(dtype=np.float64)
    out = np.full(len(pool), np.inf, dtype=np.float64)
    for i in range(len(hp)):
        d = haversine_km(np.full(len(pool), h_lat[i]), np.full(len(pool), h_lon[i]), p_lat, p_lon)
        out = np.minimum(out, d)
    return out


def band_weight(dist_km: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    w = np.zeros(len(dist_km), dtype=np.float64)
    for max_d, bw in bands:
        mask = (dist_km <= max_d) & (w == 0)
        w[mask] = bw
    return w


def ecological_similarity(pool: pd.DataFrame, hp: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not feature_cols:
        return np.ones(len(pool), dtype=np.float64)
    hp_feat = hp[feature_cols].to_numpy(dtype=np.float64)
    pool_feat = pool[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(hp_feat, axis=0)
    sd = np.nanstd(hp_feat, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z = (pool_feat - mu) / sd
    z2 = np.nanmean(z * z, axis=1)
    return np.exp(-0.5 * z2)


def in_quantile_box(pool: pd.DataFrame, hp: pd.DataFrame, feature_cols: list[str], q_low: float, q_high: float) -> np.ndarray:
    if not feature_cols:
        return np.ones(len(pool), dtype=bool)
    keep = np.ones(len(pool), dtype=bool)
    for c in feature_cols:
        lo = float(hp[c].quantile(q_low))
        hi = float(hp[c].quantile(q_high))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if lo > hi:
            lo, hi = hi, lo
        keep &= pool[c].to_numpy(dtype=np.float64) >= lo
        keep &= pool[c].to_numpy(dtype=np.float64) <= hi
    return keep


def build_grid_pool(physics_nc: Path, waves_nc: Path) -> pd.DataFrame:
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
        wave_mean = wav["VHM0"].mean(dim="time", skipna=True).interp(
            longitude=so_mean["longitude"], latitude=so_mean["latitude"], method="nearest"
        )
        wave_std = wav["VHM0"].std(dim="time", skipna=True).interp(
            longitude=so_mean["longitude"], latitude=so_mean["latitude"], method="nearest"
        )

        lon2d, lat2d = np.meshgrid(so_mean["longitude"].values, so_mean["latitude"].values)
        out = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel()})
        out["so_mean"] = np.asarray(so_mean.values).ravel()
        out["so_std"] = np.asarray(so_std.values).ravel()
        out["current_mean"] = np.asarray(current_mean.values).ravel()
        out["wave_mean"] = np.asarray(wave_mean.values).ravel()
        out["wave_std"] = np.asarray(wave_std.values).ravel()
        out = out.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
        return out
    finally:
        phys.close()
        wav.close()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    bands = parse_bands(args.distance_bands_km)
    paths = sorted(TABULAR_DIR.glob(args.dataset_glob))
    if not paths:
        raise FileNotFoundError(f"No datasets matched {args.dataset_glob}")

    report = {"out_suffix": args.out_suffix, "datasets": []}
    drop_cols = {"label", "label_weight", "species_id", "species_target", "sample_type", "lon", "lat"}
    grid_pool = None
    if args.use_copernicus_grid:
        if not args.physics_nc.exists() or not args.waves_nc.exists():
            raise FileNotFoundError(f"Missing NC files: {args.physics_nc} / {args.waves_nc}")
        grid_pool = build_grid_pool(args.physics_nc, args.waves_nc)
        print(f"[INFO] Loaded Copernicus grid candidates: {len(grid_pool)}")

    for path in paths:
        df = pd.read_csv(path)
        df = df.dropna(subset=["lon", "lat", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        if "label_weight" not in df.columns:
            df["label_weight"] = 1.0
        if "sample_type" not in df.columns:
            df["sample_type"] = np.where(df["label"] == 1, "positive", "negative")

        hp = df[(df["label"] == 1) & (df["sample_type"] == args.hard_positive_type)].copy()
        neg = df[df["label"] == 0].copy()
        if hp.empty or neg.empty:
            out_path = path.with_name(path.stem + f"_{args.out_suffix}.csv")
            df.to_csv(out_path, index=False)
            report["datasets"].append(
                {
                    "dataset": str(path),
                    "output": str(out_path),
                    "status": "unchanged",
                    "reason": "missing_hard_positive_or_negative",
                    "hard_positives": int(len(hp)),
                    "negatives": int(len(neg)),
                }
            )
            continue

        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
        feature_cols = [c for c in feature_cols if hp[c].notna().any()]

        candidates = neg.copy()
        n_extra_candidates = 0
        if grid_pool is not None:
            max_band_km = max(d for d, _ in bands)
            d_grid = nearest_hp_distance_km(grid_pool[["lon", "lat"]], hp[["lon", "lat"]])
            near_grid = grid_pool[d_grid <= max_band_km].copy()
            existing_keys = set(zip(df["lon"].round(6), df["lat"].round(6)))
            near_grid = near_grid[
                ~near_grid[["lon", "lat"]].round(6).apply(tuple, axis=1).isin(existing_keys)
            ].copy()
            if len(near_grid) > int(args.max_candidates_per_species):
                near_grid = near_grid.sample(n=int(args.max_candidates_per_species), random_state=42).copy()
            near_grid["label"] = 0
            near_grid["label_weight"] = 1.0
            near_grid["sample_type"] = "grid_candidate"
            if "species_id" in df.columns:
                near_grid["species_id"] = str(df["species_id"].dropna().iloc[0])
            n_extra_candidates = int(len(near_grid))
            candidates = pd.concat([candidates, near_grid], ignore_index=True, sort=False)

        d_km = nearest_hp_distance_km(candidates[["lon", "lat"]], hp[["lon", "lat"]])
        bw = band_weight(d_km, bands)
        sim = ecological_similarity(candidates, hp, feature_cols)
        q_gate = in_quantile_box(candidates, hp, feature_cols, args.q_low, args.q_high)

        to_soft = (bw > 0) & q_gate & (sim >= float(args.min_similarity))
        soft_rows = candidates.loc[to_soft].copy()
        soft_rows["__soft_w"] = np.clip(bw[to_soft] * sim[to_soft], args.min_soft_weight, args.max_soft_weight)
        soft_rows["__d_km"] = d_km[to_soft]
        soft_rows["__sim"] = sim[to_soft]
        soft_rows["__key"] = soft_rows[["lon", "lat"]].round(6).apply(tuple, axis=1)

        existing_neg_keys = {
            (round(float(x.lon), 6), round(float(x.lat), 6)): idx
            for idx, x in neg[["lon", "lat"]].iterrows()
        }
        soft_rows["__existing_idx"] = soft_rows["__key"].map(existing_neg_keys)

        existing_soft = soft_rows[soft_rows["__existing_idx"].notna()].copy()
        new_soft = soft_rows[soft_rows["__existing_idx"].isna()].copy()

        # Promote selected near-neighbor negatives to soft positives.
        if len(existing_soft):
            idxs = existing_soft["__existing_idx"].astype(int).to_numpy()
            df.loc[idxs, "label"] = 1
            df.loc[idxs, "label_weight"] = existing_soft["__soft_w"].to_numpy(dtype=np.float64)
            df.loc[idxs, "sample_type"] = "neighbor_soft_positive"
            df.loc[idxs, "nearest_hp_km"] = existing_soft["__d_km"].to_numpy(dtype=np.float64)
            df.loc[idxs, "soft_similarity"] = existing_soft["__sim"].to_numpy(dtype=np.float64)

        # Append brand-new soft positives from Copernicus grid candidates.
        n_new_soft = int(len(new_soft))
        if n_new_soft > 0:
            add = new_soft.drop(columns=["__key", "__existing_idx"]).copy()
            add["label"] = 1
            add["sample_type"] = "neighbor_soft_positive"
            add["label_weight"] = add["__soft_w"].to_numpy(dtype=np.float64)
            add["nearest_hp_km"] = add["__d_km"].to_numpy(dtype=np.float64)
            add["soft_similarity"] = add["__sim"].to_numpy(dtype=np.float64)
            add = add.drop(columns=["__soft_w", "__d_km", "__sim"])
            if "species_id" in df.columns and "species_id" not in add.columns:
                add["species_id"] = str(df["species_id"].dropna().iloc[0])
            df = pd.concat([df, add], ignore_index=True, sort=False)

        # Protect near-positive uncertain negatives with a buffer: remove from training.
        d_all = nearest_hp_distance_km(df[df["label"] == 0][["lon", "lat"]], hp[["lon", "lat"]])
        neg_remaining = df[df["label"] == 0].copy()
        uncertain_idx = neg_remaining.index[d_all < float(args.neg_buffer_km)]
        n_uncertain = int(len(uncertain_idx))
        if n_uncertain:
            df = df.drop(index=uncertain_idx).reset_index(drop=True)

        df = df.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)

        out_path = path.with_name(path.stem + f"_{args.out_suffix}.csv")
        df.to_csv(out_path, index=False)

        report["datasets"].append(
            {
                "dataset": str(path),
                "output": str(out_path),
                "hard_positives": int(len(hp)),
                "soft_positives_added": int(to_soft.sum()),
                "soft_from_existing_negatives": int(len(existing_soft)),
                "soft_from_grid_candidates": int(n_new_soft),
                "extra_candidates_considered": n_extra_candidates,
                "uncertain_negatives_removed": n_uncertain,
                "total_rows_out": int(len(df)),
                "positives_out": int((df["label"] == 1).sum()),
                "negatives_out": int((df["label"] == 0).sum()),
                "distance_bands_km": bands,
                "feature_cols_used": feature_cols,
            }
        )
        print(
            f"[OK] {path.name}: hp={len(hp)} soft_added={int(to_soft.sum())} "
            f"neg_removed={n_uncertain} rows_out={len(df)}"
        )

    report_path = REPORTS_DIR / f"neighbor_soft_positive_report_{args.out_suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Report: {report_path}")


if __name__ == "__main__":
    main()
