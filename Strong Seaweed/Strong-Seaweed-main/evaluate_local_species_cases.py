import argparse
import json
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases_json must be a JSON list")
    out = []
    for row in data:
        out.append(
            {
                "location_name": str(row["location_name"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "expected_species_id": str(row["expected_species_id"]),
                "acceptable_species_ids": [str(x) for x in (row.get("acceptable_species_ids") or [row["expected_species_id"]])],
                "evidence_url": str(row.get("evidence_url", "")),
                "note": str(row.get("note", "")),
            }
        )
    return out


def run(cases_json: Path, out_json: Path) -> dict[str, Any]:
    import serve_species_api as model_api

    cases = load_cases(cases_json)
    rows = []
    correct = 0
    tie_hits = 0
    uncertainty_hits = 0
    fallback_hits = 0
    pilot_hits = 0

    for c in cases:
        pred = model_api.predict_species(c["lat"], c["lon"], {})
        best = pred.get("bestSpecies") or {}
        selection = pred.get("selectionDiagnostics") or {}
        warnings = [str(w) for w in (pred.get("warnings") or [])]
        predicted_sid = str(best.get("speciesId") or "")
        action = str(best.get("actionability") or "insufficient_data")
        reason = str(best.get("reason") or "")
        tie_detected = bool(selection.get("tieDetected"))
        acceptable = [str(x) for x in c.get("acceptable_species_ids", [c["expected_species_id"]])]
        is_match = predicted_sid in set(acceptable)

        if is_match:
            correct += 1
        if tie_detected:
            tie_hits += 1
        if "uncertainty_gate" in reason or any("prediction_uncertainty" in w for w in warnings):
            uncertainty_hits += 1
        if "fallback" in reason or any("fallback" in w for w in warnings):
            fallback_hits += 1
        if action == "test_pilot_only":
            pilot_hits += 1

        rows.append(
            {
                "location_name": c["location_name"],
                "lat": c["lat"],
                "lon": c["lon"],
                "expected_species_id": c["expected_species_id"],
                "acceptable_species_ids": acceptable,
                "predicted_species_id": predicted_sid,
                "match": is_match,
                "probability_percent": best.get("probabilityPercent"),
                "actionability": action,
                "reason": reason,
                "decision_source": pred.get("decisionSource"),
                "tie_detected": tie_detected,
                "selection_reason": selection.get("selectionReason"),
                "warnings": warnings,
                "evidence_url": c["evidence_url"],
                "note": c["note"],
            }
        )

    total = len(rows)
    metrics = {
        "total_cases": total,
        "top1_accuracy": (correct / total) if total else 0.0,
        "tie_rate": (tie_hits / total) if total else 0.0,
        "uncertainty_gate_rate": (uncertainty_hits / total) if total else 0.0,
        "fallback_rate": (fallback_hits / total) if total else 0.0,
        "pilot_only_rate": (pilot_hits / total) if total else 0.0,
    }
    out = {"metrics": metrics, "rows": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate local serve_species_api.py against labeled point cases.")
    p.add_argument("--cases_json", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=Path("artifacts/reports/local_species_eval.json"))
    args = p.parse_args()
    out = run(args.cases_json, args.out_json)
    print(json.dumps(out["metrics"], indent=2))
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
