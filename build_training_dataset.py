import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree
from project_paths import TABULAR_DIR, ensure_dirs


def sample_stratified_background(
    bg_pool: pd.DataFrame,
    presence: pd.DataFrame,
    n_target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample background to roughly follow presence depth/shore strata."""
    if bg_pool.empty or n_target <= 0:
        return bg_pool.iloc[0:0].copy()

    pres = presence.copy()
    bg = bg_pool.copy()

    # Two-dimensional coarse strata reduce location shortcuts without overfitting.
    d_bins = [0, 2, 3, 4, 5]
    s_bins = [0, 1000, 2000, 3000, 4000, 5000]
    pres["d_bin"] = pd.cut(pres["depth"], bins=d_bins, include_lowest=True)
    pres["s_bin"] = pd.cut(
        pres["distance_to_shore"], bins=s_bins, include_lowest=True
    )
    bg["d_bin"] = pd.cut(bg["depth"], bins=d_bins, include_lowest=True)
    bg["s_bin"] = pd.cut(bg["distance_to_shore"], bins=s_bins, include_lowest=True)

    weights = (
        pres.groupby(["d_bin", "s_bin"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    weights["frac"] = weights["count"] / weights["count"].sum()
    weights["n_req"] = np.floor(weights["frac"] * n_target).astype(int)

    # Distribute remainder to highest-frequency strata.
    remainder = n_target - weights["n_req"].sum()
    if remainder > 0:
        order = weights.sort_values("frac", ascending=False).index.tolist()
        for i in range(remainder):
            weights.loc[order[i % len(order)], "n_req"] += 1

    sampled_parts = []
    for _, row in weights.iterrows():
        subset = bg[(bg["d_bin"] == row["d_bin"]) & (bg["s_bin"] == row["s_bin"])]
        n_req = int(row["n_req"])
        if n_req <= 0 or subset.empty:
            continue
        n_take = min(n_req, len(subset))
        idx = rng.choice(subset.index.to_numpy(), size=n_take, replace=False)
        sampled_parts.append(bg.loc[idx])

    # Keep original bg indices so top-up excludes already sampled rows correctly.
    sampled = (
        pd.concat(sampled_parts, ignore_index=False)
        if sampled_parts
        else bg.iloc[0:0].copy()
    )

    # Top up from remaining pool if strata were undersupplied.
    if len(sampled) < n_target:
        need = n_target - len(sampled)
        remaining = bg.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            take = min(need, len(remaining))
            idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
            sampled = pd.concat([sampled, remaining.loc[idx]], ignore_index=False)

    sampled = sampled.drop(columns=["d_bin", "s_bin"], errors="ignore")
    sampled = sampled.drop_duplicates(subset=["lon", "lat"])
    return sampled.reset_index(drop=True)


def snap_presence_to_domain(
    domain: pd.DataFrame,
    presence_xy: pd.DataFrame,
    max_snap_m: float = 1500.0,
) -> pd.DataFrame:
    """Snap raw presence points to nearest valid domain pixel within distance cap."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)

    dx, dy = transformer.transform(
        domain["lon"].to_numpy(dtype=np.float64),
        domain["lat"].to_numpy(dtype=np.float64),
    )
    px, py = transformer.transform(
        presence_xy["lon"].to_numpy(dtype=np.float64),
        presence_xy["lat"].to_numpy(dtype=np.float64),
    )

    domain_xy = np.c_[dx, dy]
    pres_xy = np.c_[px, py]

    tree = cKDTree(domain_xy)
    dist_m, idx = tree.query(pres_xy, k=1)
    keep = dist_m <= max_snap_m
    if not np.any(keep):
        return domain.iloc[0:0].copy()

    matched = domain.iloc[idx[keep]].copy()
    matched = matched.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return matched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build training dataset from feature matrix + presence points.")
    p.add_argument(
        "--feature_csv",
        type=str,
        default="master_feature_matrix.csv",
        help="Feature matrix CSV filename under data/tabular (or absolute path).",
    )
    p.add_argument(
        "--presence_csv",
        type=str,
        default="kappaphycus_presence_snapped_clean.csv",
        help="Presence CSV filename under data/tabular (or absolute path).",
    )
    p.add_argument(
        "--max_snap_m",
        type=float,
        default=1.0,
        help="Maximum snap distance (meters) from raw presence point to nearest valid grid pixel.",
    )
    p.add_argument(
        "--bg_ratio",
        type=int,
        default=5,
        help="Background-to-presence ratio target.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="training_dataset.csv",
        help="Output CSV filename (under workspace base path).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    feat_path = Path(args.feature_csv)
    if not feat_path.is_absolute():
        feat_path = TABULAR_DIR / feat_path
    pres_path = Path(args.presence_csv)
    if not pres_path.is_absolute():
        pres_path = TABULAR_DIR / pres_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = TABULAR_DIR / out_path

    if not feat_path.exists() or not pres_path.exists():
        raise FileNotFoundError(
            "Expected master_feature_matrix.csv and kappaphycus_presence_snapped_clean.csv"
        )

    rng = np.random.default_rng(42)
    features = pd.read_csv(feat_path)
    presence_xy = pd.read_csv(pres_path)[["lon", "lat"]].drop_duplicates()

    # Core ecological domain for both classes.
    domain = features[
        (features["depth"] < 5)
        & (features["distance_to_shore"] <= 5000)
        & (features["shallow_mask"] > 0)
    ].copy()

    presence = snap_presence_to_domain(domain, presence_xy, max_snap_m=float(args.max_snap_m))
    presence["label"] = 1

    if presence.empty:
        raise RuntimeError("No presence pixels matched the feature matrix.")

    # Exclude candidate background within 1 km of any presence pixel.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    px, py = transformer.transform(
        presence["lon"].to_numpy(), presence["lat"].to_numpy()
    )
    dx, dy = transformer.transform(domain["lon"].to_numpy(), domain["lat"].to_numpy())
    tree = cKDTree(np.c_[px, py])
    dist_m, _ = tree.query(np.c_[dx, dy], k=1)

    domain["dist_to_presence_m"] = dist_m
    bg_pool = domain[domain["dist_to_presence_m"] > 1000].copy()

    # Remove any accidental overlap with presence keys.
    pres_keys = set(zip(presence["lon"], presence["lat"]))
    bg_pool = bg_pool[
        ~bg_pool[["lon", "lat"]].apply(tuple, axis=1).isin(pres_keys)
    ].copy()

    n_bg_target = min(len(bg_pool), len(presence) * int(args.bg_ratio))
    background = sample_stratified_background(bg_pool, presence, n_bg_target, rng)
    background["label"] = 0

    training = pd.concat([presence, background], ignore_index=True)
    training = training.sample(frac=1, random_state=42).reset_index(drop=True)
    training = training.drop(columns=["dist_to_presence_m"], errors="ignore")
    training = training.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
    training.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Presence count: {len(presence)}")
    print(f"Background count: {len(background)}")
    print(f"Total rows: {len(training)}")
    print(f"max_snap_m: {float(args.max_snap_m)}")
    print(f"Columns: {', '.join(training.columns)}")


if __name__ == "__main__":
    main()
