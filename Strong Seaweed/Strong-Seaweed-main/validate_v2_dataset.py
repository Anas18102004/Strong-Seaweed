import argparse
import json
from pathlib import Path

import pandas as pd
from project_paths import TABULAR_DIR, REPORTS_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate unified v2 seaweed dataset schema and QA gates.")
    p.add_argument(
        "--input_csv",
        type=Path,
        default=TABULAR_DIR / "v2_unified_dataset_template.csv",
    )
    p.add_argument(
        "--schema_json",
        type=Path,
        default=Path("data/config/v2_dataset_schema.json"),
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=REPORTS_DIR / "v2_dataset_validation_report.json",
    )
    return p.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")
    if not args.schema_json.exists():
        raise FileNotFoundError(f"Missing schema_json: {args.schema_json}")

    schema = json.loads(args.schema_json.read_text(encoding="utf-8"))
    req_cols = schema["required_columns"]
    allowed_types = set(schema["label_definition"]["label_type_values"])
    rec_weights = schema["label_definition"]["recommended_weights"]
    dedup_cols = schema["hard_quality_rules"]["deduplicate_on"]
    max_coord_km = float(schema["hard_quality_rules"]["coordinate_precision_km_max"])

    df = pd.read_csv(args.input_csv)
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    q = df.copy()
    q["lon"] = pd.to_numeric(q["lon"], errors="coerce")
    q["lat"] = pd.to_numeric(q["lat"], errors="coerce")
    q["label"] = pd.to_numeric(q["label"], errors="coerce")
    q["label_weight"] = pd.to_numeric(q["label_weight"], errors="coerce")
    if "coordinate_precision_km" in q.columns:
        q["coordinate_precision_km"] = pd.to_numeric(q["coordinate_precision_km"], errors="coerce")
    else:
        q["coordinate_precision_km"] = pd.NA

    report = {
        "input_csv": str(args.input_csv),
        "n_rows": int(len(q)),
        "checks": {},
        "failed_checks": [],
        "species_counts": q["species"].value_counts(dropna=False).to_dict(),
    }

    def set_check(name: str, ok: bool, value=None):
        report["checks"][name] = {"pass": bool(ok), "value": value}
        if not ok:
            report["failed_checks"].append(name)

    set_check("missing_required_columns", len(missing_cols) == 0, missing_cols)
    set_check("labels_binary", q["label"].isin([0, 1]).all(), q["label"].value_counts(dropna=False).to_dict())
    set_check("label_type_allowed", q["label_type"].astype(str).isin(allowed_types).all(), q["label_type"].value_counts(dropna=False).to_dict())
    set_check("coordinates_not_null", q["lon"].notna().all() and q["lat"].notna().all(), int(q[["lon", "lat"]].isna().any(axis=1).sum()))
    set_check("coordinates_valid_range", ((q["lon"] >= -180) & (q["lon"] <= 180) & (q["lat"] >= -90) & (q["lat"] <= 90)).all(), None)
    set_check("duplicate_on_key", int(q.duplicated(subset=dedup_cols).sum()) == 0, int(q.duplicated(subset=dedup_cols).sum()))

    conflict_cells = int(q.groupby(["species", "lon", "lat"])["label"].nunique().gt(1).sum())
    set_check("no_conflicting_labels_same_cell", conflict_cells == 0, conflict_cells)

    if q["coordinate_precision_km"].notna().any():
        set_check(
            "coordinate_precision_threshold",
            bool((q["coordinate_precision_km"].fillna(max_coord_km) <= max_coord_km).all()),
            float(q["coordinate_precision_km"].max()),
        )
    else:
        set_check("coordinate_precision_threshold", True, "all_null")

    # Weight sanity by label_type.
    weight_diffs = []
    for lt, rec in rec_weights.items():
        sub = q[q["label_type"].astype(str) == lt]
        if sub.empty:
            continue
        mean_w = float(sub["label_weight"].mean())
        if abs(mean_w - float(rec)) > 0.25:
            weight_diffs.append({"label_type": lt, "mean_weight": mean_w, "recommended": rec})
    set_check("label_weight_sanity", len(weight_diffs) == 0, weight_diffs)

    report["decision"] = "PASS" if not report["failed_checks"] else "FAIL"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output_json}")
    print(f"Decision: {report['decision']}")
    if report["failed_checks"]:
        print("Failed checks:", ", ".join(report["failed_checks"]))

    if report["decision"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
