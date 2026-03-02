import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import NETCDF_DIR, REPORTS_DIR, TABULAR_DIR, ensure_dirs
from build_neighbor_soft_positives import nearest_hp_distance_km, ecological_similarity, in_quantile_box


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add hard negatives from Copernicus grid for multispecies datasets.")
    p.add_argument("--dataset_glob", type=str, default="training_dataset_*_multispecies_cop_india_v2_prod_softpos_v4_grid.csv")
    p.add_argument("--physics_nc", type=Path, default=NETCDF_DIR / "india_physics_2025w01.nc")
    p.add_argument("--waves_nc", type=Path, default=NETCDF_DIR / "india_waves_2025w01.nc")
    p.add_argument("--hard_positive_type", type=str, default="occurrence_positive")
    p.add_argument("--min_dist_km", type=float, default=12.0)
    p.add_argument("--max_dist_km", type=float, default=60.0)
    p.add_argument("--min_similarity", type=float, default=0.55)
    p.add_argument("--q_low", type=float, default=0.05)
    p.add_argument("--q_high", type=float, default=0.95)
    p.add_argument("--max_add_per_species", type=int, default=350)
    p.add_argument("--hard_negative_weight", type=float, default=1.25)
    p.add_argument("--out_suffix", type=str, default="hn_v1")
    return p.parse_args()


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
    paths = sorted(TABULAR_DIR.glob(args.dataset_glob))
    if not paths:
        raise FileNotFoundError(f"No dataset files matched: {args.dataset_glob}")
    if not args.physics_nc.exists() or not args.waves_nc.exists():
        raise FileNotFoundError(f"Missing NC files: {args.physics_nc} / {args.waves_nc}")

    grid = build_grid_pool(args.physics_nc, args.waves_nc)
    report = {"out_suffix": args.out_suffix, "datasets": [], "grid_rows": int(len(grid))}

    drop_cols = {"label", "label_weight", "species_id", "species_target", "sample_type", "lon", "lat", "nearest_hp_km", "soft_similarity"}
    for p in paths:
        df = pd.read_csv(p)
        df = df.dropna(subset=["lon", "lat", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        if "label_weight" not in df.columns:
            df["label_weight"] = 1.0
        if "sample_type" not in df.columns:
            df["sample_type"] = np.where(df["label"] == 1, "positive", "negative")
        sp = str(df["species_id"].dropna().iloc[0]) if "species_id" in df.columns and len(df) else p.stem
        hp = df[(df["label"] == 1) & (df["sample_type"] == args.hard_positive_type)].copy()
        if hp.empty:
            out_path = p.with_name(p.stem + f"_{args.out_suffix}.csv")
            df.to_csv(out_path, index=False)
            report["datasets"].append({"dataset": str(p), "output": str(out_path), "species_id": sp, "added_hard_negatives": 0, "reason": "no_hard_positives"})
            continue

        feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
        feat_cols = [c for c in feat_cols if hp[c].notna().any()]

        # Candidate pool: not already present in dataset.
        keys = set(zip(df["lon"].round(6), df["lat"].round(6)))
        cand = grid[~grid[["lon", "lat"]].round(6).apply(tuple, axis=1).isin(keys)].copy()
        if cand.empty:
            out_path = p.with_name(p.stem + f"_{args.out_suffix}.csv")
            df.to_csv(out_path, index=False)
            report["datasets"].append({"dataset": str(p), "output": str(out_path), "species_id": sp, "added_hard_negatives": 0, "reason": "no_candidates"})
            continue

        d = nearest_hp_distance_km(cand[["lon", "lat"]], hp[["lon", "lat"]])
        common_feat = [c for c in feat_cols if c in cand.columns and c in hp.columns]
        if not common_feat:
            out_path = p.with_name(p.stem + f"_{args.out_suffix}.csv")
            df.to_csv(out_path, index=False)
            report["datasets"].append({"dataset": str(p), "output": str(out_path), "species_id": sp, "added_hard_negatives": 0, "reason": "no_common_features"})
            continue

        sim = ecological_similarity(cand, hp, common_feat)
        q_gate = in_quantile_box(cand, hp, common_feat, float(args.q_low), float(args.q_high))
        mask = (
            (d >= float(args.min_dist_km))
            & (d <= float(args.max_dist_km))
            & (sim >= float(args.min_similarity))
            & q_gate
        )
        hn = cand.loc[mask].copy()
        if hn.empty:
            out_path = p.with_name(p.stem + f"_{args.out_suffix}.csv")
            df.to_csv(out_path, index=False)
            report["datasets"].append({"dataset": str(p), "output": str(out_path), "species_id": sp, "added_hard_negatives": 0, "reason": "no_gate_match"})
            continue

        hn["__score"] = sim[mask]
        hn = hn.sort_values("__score", ascending=False).head(int(args.max_add_per_species)).copy()
        hn["label"] = 0
        hn["label_weight"] = float(args.hard_negative_weight)
        hn["sample_type"] = "hard_negative"
        hn["nearest_hp_km"] = d[mask][: len(hn)]
        hn["soft_similarity"] = sim[mask][: len(hn)]
        if "species_id" in df.columns:
            hn["species_id"] = sp
        hn = hn.drop(columns=["__score"], errors="ignore")

        out_df = pd.concat([df, hn], ignore_index=True, sort=False)
        out_df = out_df.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
        out_path = p.with_name(p.stem + f"_{args.out_suffix}.csv")
        out_df.to_csv(out_path, index=False)

        report["datasets"].append(
            {
                "dataset": str(p),
                "output": str(out_path),
                "species_id": sp,
                "added_hard_negatives": int(len(hn)),
                "rows_out": int(len(out_df)),
                "positives_out": int((out_df["label"] == 1).sum()),
                "negatives_out": int((out_df["label"] == 0).sum()),
            }
        )
        print(f"[OK] {p.name}: added_hn={len(hn)} rows_out={len(out_df)}")

    out_report = REPORTS_DIR / f"hard_negative_build_report_{args.out_suffix}.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Report: {out_report}")


if __name__ == "__main__":
    main()
