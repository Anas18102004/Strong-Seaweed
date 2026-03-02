# CMFRI2020 Seed Dataset QA

- Grade: **WARN_DATASET_STAGE**
- Rows: 90 | Pos: 15 | Neg: 75 | Neg/Pos: 5.000

## Checks
- Missing cells: 0
- Rows with missing: 0
- Duplicate lon/lat/label rows: 0
- Duplicate lon/lat rows: 0
- Label-conflict cells (same lon/lat with both labels): 0
- Coordinate ranges valid: lon=True, lat=True
- Constant columns: shallow_mask, extreme_wave_days, policy_core_zone_exclusion, policy_go2005_review_required

## Failed checks
- too_few_positives(<50)
- too_few_total_rows(<300)