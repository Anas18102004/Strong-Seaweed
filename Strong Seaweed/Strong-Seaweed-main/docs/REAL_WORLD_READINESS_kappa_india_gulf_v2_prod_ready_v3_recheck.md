# Real-World Readiness Check

- Decision: **PASS**
- Model version: `kappa_india_gulf_v2_prod_ready_v3`

## Key metrics
- Train spatial AUC: 0.7211
- Train OOF AP (cal): 0.5123
- Hard50 AUC / P / R: 0.8616 / 0.9091 / 0.8000
- Independent n / pos / neg: 44 / 22 / 22
- Independent AUC / P / R: 0.8388 / 0.8947 / 0.7727

## Dataset QA
- Rows: 449 | Pos: 123 | Neg: 326
- Label conflicts (same lon/lat, mixed labels): 0 (ratio 0.0000)
- Missing cells: 0 | Rows with missing: 0
- Duplicate lon/lat/label rows: 0

## OOF QA
- OOF file found: True
- OOF missing rows: 111 / 449 (ratio 0.2472)

## Checks
- PASS | train_spatial_auc
- PASS | train_oof_ap_calibrated
- PASS | hard50_auc
- PASS | hard50_precision
- PASS | hard50_recall
- PASS | independent_n
- PASS | independent_pos
- PASS | independent_neg
- PASS | independent_auc
- PASS | independent_precision
- PASS | independent_recall
- PASS | dataset_bad_labels
- PASS | dataset_missing_cells
- PASS | dataset_no_dup_lon_lat_label
- PASS | dataset_conflict_ratio
- FAIL | oof_missing_ratio

## Failed checks
- oof_missing_ratio