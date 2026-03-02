# v1.1 QA Rubric For Positive Ingestion

## Purpose
Define strict acceptance criteria for adding new positive records to the v1.1 training batch.

## Accepted Source Types
- `literature`
- `government`
- `satellite_digitized`

## Mandatory Fields
- `record_id`
- `source_type`
- `source_name`
- `source_reference`
- `citation_url` (can be empty for internal government references if `source_reference` is provided)
- `species`
- `lon`
- `lat`
- `label`
- `coordinate_precision_km`
- `species_confirmed`
- `confidence_score`
- `is_verified`
- `qa_reviewer`
- `qa_status`
- `rationale`

## Acceptance Rules (Strict)
1. `species` must be Kappaphycus (string contains `kappaphycus`).
2. `label` must be `1` (presence).
3. `source_type` must be in the accepted list above.
4. `coordinate_precision_km <= 1.0`.
5. `species_confirmed == True`.
6. `is_verified == True`.
7. `qa_status` must be one of: `approved`, `verified`, `accepted`.
8. Candidate must not be within 1 km of existing positive points.
9. Candidate must not snap to the same 1 km grid cell as an existing positive.

## Rejection Rules
- Unknown species or mixed species unresolved.
- Village/administrative centroid without farm-level precision.
- Single-date ambiguous satellite structure.
- Missing mandatory metadata fields.

## Recommended Confidence Thresholds
- Literature / government: `confidence_score >= 0.8`
- Satellite digitized: `confidence_score >= 0.7` and two-date confirmation

## Source Mix Guidance For v1.1
- Literature: 10-12
- Government: 8-10
- Satellite digitized: 3-5
- Satellite share should remain <= 20% of new positives.

## Operational Command
Use strict mode when ingesting:

```bash
python ingest_presence_records.py --inputs v1_1_data_plan.csv --strict_acceptance
```

