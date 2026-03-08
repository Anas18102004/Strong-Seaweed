import argparse
import json
from pathlib import Path
from typing import Any

import requests

from evaluate_live_system import SpeciesCase, default_cases, post_json, obis_top_species


def classify_case(
    final_species: str,
    final_action: str,
    top_obis_species: str | None,
    top_obis_count: int,
) -> str:
    if not top_obis_species or top_obis_count <= 0:
        return "uncertain"
    if final_species == top_obis_species:
        return "match"
    if final_action in ("test_pilot_only", "not_recommended", "insufficient_data") and top_obis_count < 10:
        return "reasonable_caution"
    return "mismatch"


def load_cases(path: Path | None) -> list[SpeciesCase]:
    if not path:
        return default_cases()
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in data:
        out.append(
            SpeciesCase(
                location_name=str(row["location_name"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                season=str(row.get("season", "Post-Monsoon")),
                expected_species_id=row.get("expected_species_id"),
            )
        )
    return out


def run(base_host: str, email: str, password: str, out_path: Path, cases_path: Path | None, use_https: bool) -> dict[str, Any]:
    scheme = "https" if use_https else "http"
    base = f"{scheme}://{base_host}".rstrip("/")
    session = requests.Session()
    cases = load_cases(cases_path)

    code, auth_out = post_json(session, f"{base}/api/auth/signin", {"email": email, "password": password})
    if code != 200 or not isinstance(auth_out, dict) or "token" not in auth_out:
        raise RuntimeError(f"auth_failed status={code} response={str(auth_out)[:500]}")
    token = str(auth_out["token"])

    rows = []
    counts = {"match": 0, "reasonable_caution": 0, "uncertain": 0, "mismatch": 0}

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
            rows.append(
                {
                    "location_name": case.location_name,
                    "lat": case.lat,
                    "lon": case.lon,
                    "ok": False,
                    "status": code,
                    "error": str(out)[:400],
                }
            )
            counts["uncertain"] += 1
            continue

        final = out.get("finalRecommendation") or {}
        final_sid = str(final.get("speciesId") or "")
        final_action = str(final.get("actionability") or "insufficient_data")
        top_obis_sid, top_obis_count = obis_top_species(session, case.lat, case.lon)
        verdict = classify_case(final_sid, final_action, top_obis_sid, top_obis_count)
        counts[verdict] += 1

        rows.append(
            {
                "location_name": case.location_name,
                "lat": case.lat,
                "lon": case.lon,
                "ok": True,
                "final_species_id": final_sid,
                "final_actionability": final_action,
                "final_probability_percent": final.get("probabilityPercent"),
                "top_obis_species_id": top_obis_sid,
                "top_obis_count": top_obis_count,
                "verdict": verdict,
            }
        )

    total = len(rows)
    summary = {
        "total": total,
        "match_rate": counts["match"] / total if total else 0.0,
        "reasonable_caution_rate": counts["reasonable_caution"] / total if total else 0.0,
        "uncertain_rate": counts["uncertain"] / total if total else 0.0,
        "mismatch_rate": counts["mismatch"] / total if total else 0.0,
        "counts": counts,
    }
    output = {"base": base, "summary": summary, "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate live prediction consistency against OBIS occurrence signal.")
    p.add_argument("--base_host", default="akuara.publicvm.com")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--cases_json", type=Path, default=None)
    p.add_argument("--out_json", type=Path, default=Path("artifacts/reports/species_external_consistency.json"))
    p.add_argument("--http", action="store_true", help="Force http")
    args = p.parse_args()
    out = run(
        base_host=args.base_host,
        email=args.email,
        password=args.password,
        out_path=args.out_json,
        cases_path=args.cases_json,
        use_https=not args.http,
    )
    print(json.dumps(out, indent=2))
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
