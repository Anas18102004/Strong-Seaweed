import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SpeciesCase:
    expected_species_id: str
    location_name: str
    lat: float
    lon: float
    season: str = "Post-Monsoon"


CASES: list[SpeciesCase] = [
    SpeciesCase("kappaphycus_alvarezii", "Mandapam_Gulf_of_Mannar", 9.28, 79.12),
    SpeciesCase("gracilaria_edulis", "Tuticorin_Gulf_of_Mannar", 8.78, 78.20),
    SpeciesCase("gracilaria_edulis", "Kavaratti_Lakshadweep", 10.57, 72.64),
]


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    token: str | None = None,
    timeout: int = 20,
) -> tuple[int, dict[str, Any] | str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = session.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def run(base_url: str, email: str, password: str, use_https: bool = False) -> dict[str, Any]:
    scheme = "https" if use_https else "http"
    base = f"{scheme}://{base_url}".rstrip("/")
    session = requests.Session()

    status_code, auth_out = post_json(
        session,
        f"{base}/api/auth/signin",
        {"email": email, "password": password},
        token=None,
    )
    if status_code != 200 or not isinstance(auth_out, dict) or "token" not in auth_out:
        raise RuntimeError(f"auth_failed status={status_code} response={str(auth_out)[:500]}")
    token = str(auth_out["token"])

    species_results = []
    top1_hits = 0
    top3_hits = 0

    for case in CASES:
        payload = {
            "lat": case.lat,
            "lon": case.lon,
            "formInput": {
                "locationName": case.location_name,
                "season": case.season,
                "depthM": 4,
                "overrides": {"temperatureC": 29, "salinityPpt": 34},
                "advanced": {"currentVelocityMs": 0.25, "waveHeightM": 0.4, "ph": 8.1},
            },
        }
        code, out = post_json(session, f"{base}/api/predict/species", payload, token=token)
        if code != 200 or not isinstance(out, dict):
            species_results.append(
                {
                    "case": case.__dict__,
                    "ok": False,
                    "status": code,
                    "error": str(out)[:400],
                }
            )
            continue

        best = out.get("bestSpecies") or {}
        best_id = best.get("speciesId")
        top3 = [s.get("speciesId") for s in (out.get("species") or [])[:3]]
        top1_match = best_id == case.expected_species_id
        top3_match = case.expected_species_id in top3
        top1_hits += 1 if top1_match else 0
        top3_hits += 1 if top3_match else 0
        species_results.append(
            {
                "case": case.__dict__,
                "ok": True,
                "status": code,
                "inputMode": out.get("inputMode"),
                "modelRelease": out.get("modelRelease"),
                "bestSpeciesId": best_id,
                "bestSpeciesProb": best.get("probabilityPercent"),
                "top3SpeciesIds": top3,
                "warnings": out.get("warnings") or [],
                "top1_match": top1_match,
                "top3_match": top3_match,
            }
        )

    ai_context = {
        "mode": "ask",
        "locationName": "Gulf of Mannar",
        "speciesHint": "Kappaphycus",
        "season": "Post-Monsoon",
        "lat": 9.28,
        "lon": 79.12,
        "depthM": 4,
        "overrides": {"temperatureC": 29, "salinityPpt": 34},
        "advanced": {"currentVelocityMs": 0.25, "waveHeightM": 0.4, "ph": 8.1},
    }
    ai_code, ai_out = post_json(
        session,
        f"{base}/api/ai/chat",
        {
            "question": "Which seaweed should I cultivate here and why?",
            "context": ai_context,
        },
        token=token,
    )
    ai_summary: dict[str, Any]
    if ai_code == 200 and isinstance(ai_out, dict):
        ai_summary = {
            "ok": True,
            "status": ai_code,
            "provider": ai_out.get("provider"),
            "statusField": ai_out.get("status"),
            "modelGrounded": ai_out.get("modelGrounded"),
            "model": ai_out.get("model"),
            "answerPreview": str(ai_out.get("answer", ""))[:260],
        }
    else:
        ai_summary = {
            "ok": False,
            "status": ai_code,
            "error": str(ai_out)[:400],
        }

    total = len(CASES)
    return {
        "base": base,
        "species_eval": {
            "total_cases": total,
            "top1_hits": top1_hits,
            "top1_accuracy": (top1_hits / total) if total else 0.0,
            "top3_hits": top3_hits,
            "top3_accuracy": (top3_hits / total) if total else 0.0,
            "cases": species_results,
        },
        "ai_eval": ai_summary,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Live end-to-end evaluator for Akuara API")
    p.add_argument("--base_host", default="akuara.publicvm.com", help="Host without scheme")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--https", action="store_true", help="Use https instead of http")
    p.add_argument("--out_json", default="artifacts/reports/live_system_eval.json")
    args = p.parse_args()

    out = run(args.base_host, args.email, args.password, use_https=args.https)
    print(json.dumps(out, indent=2))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()

