import argparse
import json
from pathlib import Path

import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean training dataset by resolving lon/lat label conflicts and duplicates."
    )
    p.add_argument(
        "--input_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_plus11web_plus35provisional.csv",
        help="Input training dataset CSV.",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=TABULAR_DIR / "training_dataset_v1_1_plus11web_plus35provisional_clean.csv",
        help="Cleaned output CSV.",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "training_dataset_conflict_cleaning_report.json",
        help="Where to write cleaning summary JSON.",
    )
    p.add_argument(
        "--conflict_policy",
        choices=["drop_both", "keep_positive", "keep_negative", "majority"],
        default="drop_both",
        help="How to handle rows where same lon/lat has both labels.",
    )
    p.add_argument(
        "--round_coords",
        type=int,
        default=8,
        help="Decimal rounding for lon/lat conflict grouping.",
    )
    return p.parse_args()


def resolve_conflicts(df: pd.DataFrame, policy: str, decimals: int) -> tuple[pd.DataFrame, dict]:
    tmp = df.copy()
    tmp["lon_r"] = tmp["lon"].round(decimals)
    tmp["lat_r"] = tmp["lat"].round(decimals)
    key = ["lon_r", "lat_r"]

    before_n = int(len(tmp))
    dup_before = int(tmp.duplicated(subset=["lon_r", "lat_r", "label"]).sum())
    grouped = tmp.groupby(key)["label"].nunique().reset_index(name="n_labels")
    conflict_cells = grouped[grouped["n_labels"] > 1][key]
    n_conflict_cells = int(len(conflict_cells))

    if n_conflict_cells > 0:
        tmp = tmp.merge(conflict_cells.assign(_is_conflict=1), on=key, how="left")
        tmp["_is_conflict"] = tmp["_is_conflict"].fillna(0).astype(int)
    else:
        tmp["_is_conflict"] = 0

    if policy == "drop_both":
        out = tmp[tmp["_is_conflict"] == 0].copy()
    elif policy == "keep_positive":
        out = tmp[(tmp["_is_conflict"] == 0) | (tmp["label"] == 1)].copy()
    elif policy == "keep_negative":
        out = tmp[(tmp["_is_conflict"] == 0) | (tmp["label"] == 0)].copy()
    else:
        # majority per rounded lon/lat; tie defaults to dropping both.
        non_conflict = tmp[tmp["_is_conflict"] == 0].copy()
        conflict = tmp[tmp["_is_conflict"] == 1].copy()
        if conflict.empty:
            out = non_conflict
        else:
            votes = (
                conflict.groupby(key + ["label"])
                .size()
                .rename("n")
                .reset_index()
                .sort_values(key + ["n", "label"], ascending=[True, True, False, False])
            )
            top = votes.groupby(key).head(1).copy()
            ties = votes.groupby(key)["n"].nunique().reset_index(name="n_unique")
            tied_cells = ties[ties["n_unique"] == 1][key]
            # Keep only cells with a clear winner; if exact tie, drop that cell.
            if len(tied_cells) > 0:
                top = top.merge(tied_cells.assign(_ok=1), on=key, how="left")
                top = top[top["_ok"] == 1].drop(columns=["_ok"])
            keep = conflict.merge(top[key + ["label"]], on=key + ["label"], how="inner")
            out = pd.concat([non_conflict, keep], ignore_index=True)

    out = out.drop(columns=["lon_r", "lat_r", "_is_conflict"], errors="ignore")
    out = out.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)

    out_tmp = out.copy()
    out_tmp["lon_r"] = out_tmp["lon"].round(decimals)
    out_tmp["lat_r"] = out_tmp["lat"].round(decimals)
    after_conflict = (
        out_tmp.groupby(["lon_r", "lat_r"])["label"].nunique().gt(1).sum()
    )

    summary = {
        "rows_before": before_n,
        "rows_after": int(len(out)),
        "rows_removed": int(before_n - len(out)),
        "duplicates_lon_lat_label_before": dup_before,
        "duplicates_lon_lat_label_after": int(out.duplicated(subset=["lon", "lat", "label"]).sum()),
        "conflict_cells_before": n_conflict_cells,
        "conflict_cells_after": int(after_conflict),
        "labels_before": {
            "pos": int((df["label"] == 1).sum()),
            "neg": int((df["label"] == 0).sum()),
        },
        "labels_after": {
            "pos": int((out["label"] == 1).sum()),
            "neg": int((out["label"] == 0).sum()),
        },
    }
    return out, summary


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required = {"lon", "lat", "label"}
    miss = sorted(required - set(df.columns))
    if miss:
        raise ValueError(f"Input CSV missing required columns: {miss}")

    cleaned, summary = resolve_conflicts(df, args.conflict_policy, args.round_coords)
    summary["input_csv"] = str(args.input_csv)
    summary["output_csv"] = str(args.output_csv)
    summary["report_json"] = str(args.report_json)
    summary["conflict_policy"] = args.conflict_policy
    summary["round_coords"] = int(args.round_coords)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_csv, index=False)
    args.report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved cleaned CSV: {args.output_csv}")
    print(f"Saved report JSON: {args.report_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
