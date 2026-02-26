import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from project_paths import OUTPUTS_DIR, EXPERIMENTS_DIR, ensure_dirs, with_legacy

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create spatially spread field survey candidates.")
    p.add_argument("--input", type=Path, default=with_legacy(OUTPUTS_DIR / "realtime_ranked_candidates.csv", "realtime_ranked_candidates.csv"))
    p.add_argument("--output", type=Path, default=EXPERIMENTS_DIR / "field_sampling_pack.csv")
    p.add_argument("--n_points", type=int, default=30)
    p.add_argument("--min_spacing_km", type=float, default=2.0)
    p.add_argument("--allow_medium", action="store_true", help="Include medium priority after high.")
    return p.parse_args()


def min_distance_km(lon: float, lat: float, chosen: list[tuple[float, float]], geod: Geod) -> float:
    if not chosen:
        return np.inf
    dmin = np.inf
    for cl, ct in chosen:
        _, _, d_m = geod.inv(cl, ct, lon, lat)
        dmin = min(dmin, d_m / 1000.0)
    return dmin


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}")

    df = pd.read_csv(args.input)
    required = {"lon", "lat", "p_calibrated", "priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.allow_medium:
        df = df[df["priority"].isin(["high", "medium"])].copy()
    else:
        df = df[df["priority"] == "high"].copy()

    df = df.sort_values("p_calibrated", ascending=False).reset_index(drop=True)
    geod = Geod(ellps="WGS84")
    chosen_rows = []
    chosen_pts: list[tuple[float, float]] = []

    for _, row in df.iterrows():
        lon = float(row["lon"])
        lat = float(row["lat"])
        d = min_distance_km(lon, lat, chosen_pts, geod)
        if d >= args.min_spacing_km:
            out = row.to_dict()
            out["min_dist_to_selected_km"] = None if np.isinf(d) else float(d)
            chosen_rows.append(out)
            chosen_pts.append((lon, lat))
        if len(chosen_rows) >= args.n_points:
            break

    out_df = pd.DataFrame(chosen_rows)
    out_df.insert(0, "site_rank", np.arange(1, len(out_df) + 1))
    out_df.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Selected points: {len(out_df)}")
    if len(out_df) > 0:
        print(
            f"p_calibrated range: {out_df['p_calibrated'].min():.3f} - {out_df['p_calibrated'].max():.3f}"
        )


if __name__ == "__main__":
    main()
