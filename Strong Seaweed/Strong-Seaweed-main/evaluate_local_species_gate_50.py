import argparse
import json
from pathlib import Path
from typing import Any

import requests

import serve_species_api as model_api
from evaluate_live_system import SPECIES_QUERY, default_cases


def obis_top_species(session: requests.Session, lat: float, lon: float) -> tuple[str | None, int]:
    species_counts: dict[str, int] = {}
    lat1, lat2 = lat - 1.0, lat + 1.0
    lon1, lon2 = lon - 1.0, lon + 1.0
    geometry = (
        f"POLYGON(({lon1} {lat1}, {lon2} {lat1}, {lon2} {lat2}, "
        f"{lon1} {lat2}, {lon1} {lat1}))"
    )
    for sid, sci_name in SPECIES_QUERY.items():
        try:
            r = session.get(
                "https://api.obis.org/v3/occurrence",
                params={"scientificname": sci_name, "geometry": geometry, "size": 1},
                timeout=20,
            )
            js = r.json() if r.ok else {}
            species_counts[sid] = int(js.get("total") or 0)
        except Exception:
            species_counts[sid] = 0
    ranked = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None, 0
    return ranked[0][0], ranked[0][1]


def run(out_json: Path) -> dict[str, Any]:
    session = requests.Session()
    cases = default_cases()
    rows = []

    coverage_hits = 0
    tie_hits = 0
    forced_pilot_hits = 0
    tie_recommended_hits = 0
    external_agreement_hits = 0
    external_valid_total = 0
    high_conf_mismatch = 0
    high_conf_total = 0
    labeled_hits = 0
    labeled_total = 0

    for case in cases:
        pred = model_api.predict_species(case.lat, case.lon, {"season": case.season, "depthM": 5})
        best = pred.get("bestSpecies") or {}
        selection = pred.get("selectionDiagnostics") or {}
        warnings = [str(w) for w in (pred.get("warnings") or [])]

        sid = str(best.get("speciesId") or "")
        action = str(best.get("actionability") or "insufficient_data")
        prob = float(best.get("probabilityPercent") or 0.0)
        tie = bool(selection.get("tieDetected"))

        if sid and sid != "insufficient_data":
            coverage_hits += 1
        if tie:
            tie_hits += 1
            if action == "recommended":
                tie_recommended_hits += 1
        if "best_species_tie_forced_pilot_only" in warnings:
            forced_pilot_hits += 1

        top_obis_sid, top_obis_count = obis_top_species(session, case.lat, case.lon)
        external_match = bool(top_obis_sid and sid == top_obis_sid)
        if top_obis_sid:
            external_valid_total += 1
            if external_match:
                external_agreement_hits += 1

        if action == "recommended" and prob >= 70.0 and top_obis_sid:
            high_conf_total += 1
            if not external_match:
                high_conf_mismatch += 1

        labeled_match = None
        if case.expected_species_id:
            labeled_total += 1
            labeled_match = sid == case.expected_species_id
            if labeled_match:
                labeled_hits += 1

        rows.append(
            {
                "location_name": case.location_name,
                "lat": case.lat,
                "lon": case.lon,
                "expected_species_id": case.expected_species_id,
                "predicted_species_id": sid,
                "actionability": action,
                "probability_percent": prob,
                "decision_source": pred.get("decisionSource"),
                "tie_detected": tie,
                "warnings": warnings,
                "top_obis_species_id": top_obis_sid,
                "top_obis_count": top_obis_count,
                "external_match": external_match if top_obis_sid else None,
                "labeled_match": labeled_match,
            }
        )

    total = len(cases)
    metrics = {
        "total_cases": total,
        "coverage_rate": (coverage_hits / total) if total else 0.0,
        "tie_rate": (tie_hits / total) if total else 0.0,
        "forced_pilot_rate": (forced_pilot_hits / total) if total else 0.0,
        "recommended_when_tie_rate": (tie_recommended_hits / tie_hits) if tie_hits else 0.0,
        "top1_agreement_external": (external_agreement_hits / external_valid_total) if external_valid_total else 0.0,
        "high_conf_mismatch_rate": (high_conf_mismatch / high_conf_total) if high_conf_total else 0.0,
        "labeled_top1_accuracy": (labeled_hits / labeled_total) if labeled_total else 0.0,
        "labeled_total": labeled_total,
    }
    out = {
        "metrics": metrics,
        "stop_criteria_targets": {
            "coverage_rate_min": 0.95,
            "top1_agreement_external_min": 0.80,
            "high_conf_mismatch_rate_max": 0.05,
            "tie_rate_max": 0.10,
            "recommended_when_tie_rate_max": 0.02,
        },
        "model_release": {
            "kappaphycus_release": model_api.KAPPA.get("release"),
            "multispecies_release": model_api.ACTIVE_MULTI_RELEASE,
        },
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Local 50-point species gate (model API direct, OBIS consistency).")
    p.add_argument("--out_json", type=Path, default=Path("artifacts/reports/local_species_gate_50.json"))
    args = p.parse_args()
    out = run(args.out_json)
    print(json.dumps(out["metrics"], indent=2))
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
