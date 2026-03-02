# Production Gate: v1.2_from_pack_auto

- Result: **FAIL**

## Checks
- PASS | train_spatial_auc>=0.70
- PASS | train_oof_ap_cal>=0.45
- FAIL | hard50_auc>=0.85
- FAIL | hard50_precision>=0.80
- PASS | hard50_recall>=0.40
- FAIL | independent_n>=40
- FAIL | independent_pos>=20
- FAIL | independent_neg>=20

## Metrics
- train_spatial_auc: 0.7725214735412104
- train_oof_ap_calibrated: 0.598132989543609
- hard50_auc: 0.8192
- hard50_precision: 0.7727272727272727
- hard50_recall: 0.68
- independent_n: 12
- independent_pos: 1
- independent_neg: 11