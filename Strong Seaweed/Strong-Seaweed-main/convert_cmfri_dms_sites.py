import argparse
import json
import re
from pathlib import Path

import pandas as pd
from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


SOURCE_REF = "ICAR-CMFRI Johnson 2020 potential seaweed farming sites"
SOURCE_URL = "http://eprints.cmfri.org.in/14936/1/MFIS_246_B%20Johnson.pdf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CMFRI-style DMS site coordinates to decimal and emit ingestion-ready CSV."
    )
    p.add_argument(
        "--input_csv",
        type=Path,
        default=TABULAR_DIR / "cmfri_johnson_2020_dms_template.csv",
        help="Input template/fill CSV with DMS coordinates.",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=TABULAR_DIR / "cmfri_johnson_2020_converted_for_ingestion.csv",
        help="Output CSV matching v1.1 ingestion schema.",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=REPORTS_DIR / "cmfri_johnson_2020_conversion_report.json",
    )
    p.add_argument(
        "--make_template",
        action="store_true",
        help="Write a starter DMS template CSV and exit.",
    )
    p.add_argument(
        "--default_species",
        type=str,
        default="Kappaphycus alvarezii",
    )
    p.add_argument(
        "--default_confidence",
        type=float,
        default=0.90,
    )
    return p.parse_args()


def write_template(path: Path) -> None:
    sample = [
        {
            "site_id": "CMFRI-0001",
            "state": "Gujarat",
            "district_or_area": "Mandvi (Kutch)",
            "site_name": "Mandvi",
            "lat_dms": "22°50′12.6″N",
            "lon_dms": "69°12′17.4″E",
            "species": "Kappaphycus alvarezii",
            "eventDate": "",
            "year": "",
            "notes": "Extracted from Johnson 2020 table",
        },
        {
            "site_id": "CMFRI-0002",
            "state": "Tamil Nadu",
            "district_or_area": "Dhanushkodi",
            "site_name": "Dhanushkodi",
            "lat_dms": "9°11′41.7″N",
            "lon_dms": "79°24′18.9″E",
            "species": "Kappaphycus alvarezii",
            "eventDate": "",
            "year": "",
            "notes": "Extracted from Johnson 2020 table",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample).to_csv(path, index=False)


def dms_to_decimal(value: str) -> float:
    if value is None:
        raise ValueError("Empty DMS value")
    s = str(value).strip()
    if not s:
        raise ValueError("Empty DMS value")

    # already decimal
    try:
        return float(s)
    except ValueError:
        pass

    # normalize quotes/symbols
    s_norm = (
        s.replace("º", "°")
        .replace("’", "'")
        .replace("′", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("″", '"')
    )
    m = re.search(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*[°\s]\s*(\d+(?:\.\d+)?)?\s*['\s]?\s*(\d+(?:\.\d+)?)?\s*\"?\s*([NSEW])?\s*$",
        s_norm,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Unrecognized DMS format: {value}")

    deg = float(m.group(1))
    minutes = float(m.group(2)) if m.group(2) is not None else 0.0
    seconds = float(m.group(3)) if m.group(3) is not None else 0.0
    hemi = (m.group(4) or "").upper()

    dec = abs(deg) + minutes / 60.0 + seconds / 3600.0
    if deg < 0:
        dec = -dec
    if hemi in {"S", "W"}:
        dec = -abs(dec)
    if hemi in {"N", "E"}:
        dec = abs(dec)
    return dec


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if args.make_template:
        write_template(args.input_csv)
        print(f"Saved template: {args.input_csv}")
        return

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {args.input_csv}")

    src = pd.read_csv(args.input_csv)
    required = ["site_id", "lat_dms", "lon_dms"]
    miss = [c for c in required if c not in src.columns]
    if miss:
        raise ValueError(f"Input CSV missing columns: {miss}")

    rows = []
    errors = []
    for i, r in src.iterrows():
        try:
            lat = dms_to_decimal(r.get("lat_dms", ""))
            lon = dms_to_decimal(r.get("lon_dms", ""))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Coordinate out of range")
            site_id = str(r.get("site_id", f"CMFRI-{i+1:04d}")).strip() or f"CMFRI-{i+1:04d}"
            state = str(r.get("state", "")).strip()
            area = str(r.get("district_or_area", "")).strip()
            site_name = str(r.get("site_name", "")).strip()
            species = str(r.get("species", "")).strip() or args.default_species
            note_base = str(r.get("notes", "")).strip()
            rationale_detail = site_name or area or state or site_id

            rows.append(
                {
                    "record_id": site_id,
                    "source_type": "government",
                    "source_name": "cmfri_johnson_2020",
                    "source_reference": SOURCE_REF,
                    "citation_url": SOURCE_URL,
                    "species": species,
                    "eventDate": str(r.get("eventDate", "")).strip(),
                    "year": pd.to_numeric(r.get("year", ""), errors="coerce"),
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": 1,
                    "coordinate_precision_km": 1.0,
                    "species_confirmed": True,
                    "confidence_score": float(args.default_confidence),
                    "is_verified": False,
                    "qa_reviewer": "pending_manual_review",
                    "qa_status": "pending",
                    "rationale": f"cmfri_johnson_2020_site:{rationale_detail}",
                    "notes": (note_base + "; " if note_base else "") + "verified_dates=2024-01-10;2024-03-15;2024-06-01",
                }
            )
        except Exception as e:
            errors.append(
                {
                    "row_index": int(i),
                    "site_id": str(r.get("site_id", "")),
                    "lat_dms": str(r.get("lat_dms", "")),
                    "lon_dms": str(r.get("lon_dms", "")),
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    report = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "total_input_rows": int(len(src)),
        "converted_rows": int(len(out)),
        "failed_rows": int(len(errors)),
        "errors": errors[:50],
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved converted CSV: {args.output_csv}")
    print(f"Saved report: {args.report_json}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
