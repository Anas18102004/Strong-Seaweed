# Model Card: Gulf of Mannar Regional Ecological Screening Model (kappa_india_gulf_v2_prod_ready_v4_with_v2labels)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.479413, 79.799937], lat [8.430069, 9.768559]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_kappa_india_gulf_v2_hardmerge4_augmented_plus_v2labels_2026_02_28_deployable
- Samples: 450
- Positives: 124
- Negatives: 326
- Features: 29
- Spatial folds: 8

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.7824 +/- 0.0610
- Spatial AP: 0.7066
- OOF calibrated AP: 0.5772
- OOF calibrated Brier: 0.1820

## Deployment Policy
- Threshold: 0.785714
- Precision at threshold: 0.8000
- Recall at threshold: 0.1000
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (450 rows, 124 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
