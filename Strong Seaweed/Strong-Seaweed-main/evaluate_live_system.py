import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SpeciesCase:
    location_name: str
    lat: float
    lon: float
    season: str = "Post-Monsoon"
    expected_species_id: str | None = None


def default_cases() -> list[SpeciesCase]:
    # 50 benchmark points: known farms + west/east coast + islands.
    rows = [
        ("Mandapam_Gulf_of_Mannar", 9.28, 79.12, "kappaphycus_alvarezii"),
        ("Palk_Bay_Central", 9.35, 79.20, "kappaphycus_alvarezii"),
        ("Tuticorin", 8.78, 78.20, "gracilaria_edulis"),
        ("Kavaratti", 10.57, 72.64, None),
        ("Agatti", 10.86, 72.19, None),
        ("Minicoy", 8.28, 73.05, None),
        ("Port_Blair", 11.67, 92.75, None),
        ("Havelock", 11.99, 93.00, None),
        ("Diglipur", 13.27, 93.04, None),
        ("Dwarka", 22.24, 68.97, None),
        ("Okha", 22.47, 69.08, None),
        ("Jamnagar", 22.47, 70.07, None),
        ("Porbandar", 21.64, 69.61, None),
        ("Veraval", 20.91, 70.36, None),
        ("Diu", 20.72, 70.98, None),
        ("Daman", 20.41, 72.83, None),
        ("Mumbai_Offshore", 18.87, 72.80, None),
        ("Alibag", 18.64, 72.87, None),
        ("Ratnagiri", 16.98, 73.30, None),
        ("Malvan", 16.06, 73.46, None),
        ("Karwar", 14.81, 74.13, None),
        ("Udupi", 13.34, 74.74, None),
        ("Mangaluru", 12.90, 74.79, None),
        ("Kannur", 11.87, 75.35, None),
        ("Kozhikode", 11.25, 75.77, None),
        ("Kochi", 9.96, 76.24, None),
        ("Alappuzha", 9.49, 76.32, None),
        ("Kollam", 8.89, 76.61, None),
        ("Kanyakumari", 8.08, 77.54, None),
        ("Rameswaram", 9.29, 79.31, "kappaphycus_alvarezii"),
        ("Nagapattinam", 10.77, 79.84, None),
        ("Cuddalore", 11.75, 79.78, None),
        ("Puducherry", 11.93, 79.83, None),
        ("Chennai_Offshore", 13.08, 80.31, None),
        ("Pulicat", 13.42, 80.32, None),
        ("Nellore_Coast", 14.45, 80.02, None),
        ("Machilipatnam", 16.17, 81.15, None),
        ("Kakinada", 16.93, 82.25, None),
        ("Visakhapatnam", 17.70, 83.30, None),
        ("Srikakulam", 18.30, 84.05, None),
        ("Gopalpur", 19.27, 84.91, None),
        ("Puri", 19.81, 85.84, None),
        ("Chilika_Mouth", 19.75, 85.40, None),
        ("Digha", 21.63, 87.53, None),
        ("Sagar_Island", 21.65, 88.06, None),
        ("Paradip", 20.27, 86.67, None),
        ("Bakkhali", 21.57, 88.26, None),
        ("Amini_Lakshadweep", 11.12, 72.73, None),
        ("Little_Andaman", 10.57, 92.56, None),
        ("Car_Nicobar", 9.17, 92.82, None),
    ]
    return [SpeciesCase(location_name=a, lat=b, lon=c, expected_species_id=d) for a, b, c, d in rows]


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    token: str | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, Any] | str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = session.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


SPECIES_QUERY = {
    "kappaphycus_alvarezii": "Kappaphycus alvarezii",
    "gracilaria_edulis": "Hydropuntia edulis",
    "ulva_lactuca": "Ulva lactuca",
    "sargassum_wightii": "Sargassum swartzii",
}


def obis_top_species(session: requests.Session, lat: float, lon: float) -> tuple[str | None, int]:
    species_counts: dict[str, int] = {}
    lat1, lat2 = lat - 1.0, lat + 1.0
    lon1, lon2 = lon - 1.0, lon + 1.0
    geometry = (
        f"POLYGON(({lon1} {lat1}, {lon2} {lat1}, {lon2} {lat2}, "
        f"{lon1} {lat2}, {lon1} {lat1}))"
    )
    for sid, sci_name in SPECIES_QUERY.items():
        params = {
            "scientificname": sci_name,
            "geometry": geometry,
            "size": 1,
        }
        try:
            r = session.get("https://api.obis.org/v3/occurrence", params=params, timeout=20)
            out = r.json() if r.ok else {}
            species_counts[sid] = int(out.get("total") or 0)
        except Exception:
            species_counts[sid] = 0
    ranked = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None, 0
    return ranked[0][0], ranked[0][1]


def run(base_url: str, email: str, password: str, use_https: bool = True) -> dict[str, Any]:
    scheme = "https" if use_https else "http"
    base = f"{scheme}://{base_url}".rstrip("/")
    session = requests.Session()
    cases = default_cases()

    auth_payload = {"email": email, "password": password}
    status_code, auth_out = post_json(session, f"{base}/api/auth/signin", auth_payload, token=None)
    if (status_code != 200 or not isinstance(auth_out, dict) or "token" not in auth_out) and not use_https:
        status_code, auth_out = post_json(
            session,
            f"https://{base_url.rstrip('/')}/api/auth/signin",
            auth_payload,
            token=None,
        )
    if status_code != 200 or not isinstance(auth_out, dict) or "token" not in auth_out:
        raise RuntimeError(f"auth_failed status={status_code} response={str(auth_out)[:500]}")
    token = str(auth_out["token"])

    case_rows = []
    top1_label_hits = 0
    labeled_total = 0
    coverage_hits = 0
    tie_hits = 0
    forced_pilot_hits = 0
    tie_recommended_hits = 0
    external_agreement_hits = 0
    external_valid_total = 0
    high_conf_mismatch = 0
    high_conf_total = 0

    for case in cases:
        payload = {
            "lat": case.lat,
            "lon": case.lon,
            "formInput": {
                "locationName": case.location_name,
                "season": case.season,
                "depthM": 5,
            },
        }
        code, out = post_json(session, f"{base}/api/predict/species", payload, token=token)
        if code != 200 or not isinstance(out, dict):
            case_rows.append(
                {
                    "case": case.__dict__,
                    "ok": False,
                    "status": code,
                    "error": str(out)[:400],
                }
            )
            continue

        final = out.get("finalRecommendation") or {}
        best = out.get("bestSpecies") or {}
        warnings = [str(w) for w in (out.get("warnings") or [])]
        selection = out.get("selectionDiagnostics") or {}
        final_sid = str(final.get("speciesId") or "")
        final_prob = float(final.get("probabilityPercent") or 0.0)
        final_action = str(final.get("actionability") or "insufficient_data")
        tie = bool(selection.get("tieDetected"))

        if final_sid and final_sid != "insufficient_data":
            coverage_hits += 1
        if tie:
            tie_hits += 1
            if final_action == "recommended":
                tie_recommended_hits += 1
        if "best_species_tie_forced_pilot_only" in warnings:
            forced_pilot_hits += 1

        top_obis_sid, top_obis_count = obis_top_species(session, case.lat, case.lon)
        external_match = bool(top_obis_sid and final_sid == top_obis_sid)
        if top_obis_sid:
            external_valid_total += 1
            if external_match:
                external_agreement_hits += 1

        if final_action == "recommended" and final_prob >= 70.0 and top_obis_sid:
            high_conf_total += 1
            if not external_match:
                high_conf_mismatch += 1

        labeled_match = None
        if case.expected_species_id:
            labeled_total += 1
            labeled_match = final_sid == case.expected_species_id
            if labeled_match:
                top1_label_hits += 1

        case_rows.append(
            {
                "case": case.__dict__,
                "ok": True,
                "status": code,
                "modelRelease": out.get("modelRelease"),
                "decisionSource": out.get("decisionSource"),
                "finalSpeciesId": final_sid,
                "finalActionability": final_action,
                "finalProbabilityPercent": final_prob,
                "bestSpeciesId": best.get("speciesId"),
                "tieDetected": tie,
                "warnings": warnings,
                "topObisSpeciesId": top_obis_sid,
                "topObisCount": top_obis_count,
                "externalMatch": external_match if top_obis_sid else None,
                "labeledMatch": labeled_match,
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
        "labeled_top1_accuracy": (top1_label_hits / labeled_total) if labeled_total else 0.0,
        "labeled_total": labeled_total,
    }

    return {
        "base": base,
        "metrics": metrics,
        "stop_criteria_targets": {
            "coverage_rate_min": 0.95,
            "top1_agreement_external_min": 0.80,
            "high_conf_mismatch_rate_max": 0.05,
            "tie_rate_max": 0.10,
            "recommended_when_tie_rate_max": 0.02,
        },
        "cases": case_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Live benchmark evaluator for Akuara API (50-point suite + external consistency).")
    p.add_argument("--base_host", default="akuara.publicvm.com", help="Host without scheme")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--https", action="store_true", default=True, help="Use https instead of http")
    p.add_argument("--http", action="store_true", help="Force http")
    p.add_argument("--out_json", default="artifacts/reports/live_system_eval.json")
    args = p.parse_args()

    use_https = not args.http
    out = run(args.base_host, args.email, args.password, use_https=use_https)
    print(json.dumps(out, indent=2))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
