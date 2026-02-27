import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert reviewed shortlist to strict-ingestion-ready verified positives."
    )
    p.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Reviewed shortlist CSV with verification flags.",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV for strict ingestion.",
    )
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.85,
        help="Minimum confidence score to keep.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    req_cols = [
        "verify_multidate_pass",
        "verify_raft_persistence_pass",
        "verify_not_port_shipping_pass",
        "verify_env_gate_pass",
        "verified_dates",
        "confidence_score",
        "lat",
        "lon",
        "species",
        "record_id",
    ]
    miss = [c for c in req_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    keep = (
        df["verify_multidate_pass"].astype(bool)
        & df["verify_raft_persistence_pass"].astype(bool)
        & df["verify_not_port_shipping_pass"].astype(bool)
        & df["verify_env_gate_pass"].astype(bool)
        & df["verified_dates"].astype(str).str.contains(";", na=False)
        & (pd.to_numeric(df["confidence_score"], errors="coerce") >= float(args.min_confidence))
    )
    out = df[keep].copy()
    if out.empty:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved empty verified set: {args.output_csv}")
        return

    out["label"] = 1
    out["source_type"] = out.get("source_type", "satellite_digitized")
    out["source_name"] = out.get("source_name", "regime_positive_mining_v1_1p1")
    out["source_reference"] = out.get(
        "source_reference", "verified shortlist from regime-positive mining"
    )
    out["citation_url"] = out.get("citation_url", "")
    out["eventDate"] = out.get("eventDate", "")
    out["year"] = out.get("year", pd.NA)
    out["coordinate_precision_km"] = pd.to_numeric(
        out.get("coordinate_precision_km", 1.0), errors="coerce"
    ).fillna(1.0)
    out["species_confirmed"] = True
    out["is_verified"] = True
    out["qa_status"] = "approved"
    out["notes"] = (
        out.get("notes", "").astype(str)
        + "|verified_dates="
        + out["verified_dates"].astype(str)
    )

    export_cols = [
        "record_id",
        "source_type",
        "source_name",
        "source_reference",
        "citation_url",
        "species",
        "eventDate",
        "year",
        "lon",
        "lat",
        "label",
        "coordinate_precision_km",
        "species_confirmed",
        "confidence_score",
        "is_verified",
        "qa_reviewer",
        "qa_status",
        "rationale",
        "notes",
    ]
    for c in export_cols:
        if c not in out.columns:
            out[c] = ""
    out = out[export_cols].copy()
    out = out.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved verified positives: {args.output_csv} | rows={len(out)}")


if __name__ == "__main__":
    main()
