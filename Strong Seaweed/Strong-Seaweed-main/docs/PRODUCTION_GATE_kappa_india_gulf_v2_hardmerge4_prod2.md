# Production Gate: kappa_india_gulf_v2_hardmerge4_prod2

- Result: **FAIL**

## Checks
- PASS | train_spatial_auc>=0.70
- PASS | train_oof_ap_cal>=0.45
- FAIL | hard50_auc>=0.85
- PASS | hard50_precision>=0.80
- PASS | hard50_recall>=0.40
- PASS | independent_n>=40
- PASS | independent_pos>=20
- PASS | independent_neg>=20

## Metrics
- train_spatial_auc: 0.7928346082022553
- train_oof_ap_calibrated: 0.5865684745712645
- hard50_auc: 0.8288
- hard50_precision: 0.9
- hard50_recall: 0.72
- independent_n: 44
- independent_pos: 22
- independent_neg: 22