# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.0)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.102121, 79.799937], lat [8.448035, 9.705677]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_1_deployable
- Samples: 108
- Positives: 18
- Negatives: 90
- Features: 21
- Spatial folds: 3

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.8996 +/- 0.1420
- Spatial AP: 0.9022
- OOF calibrated AP: 0.7423
- OOF calibrated Brier: 0.1362

## Deployment Policy
- Threshold: 1.000000
- Precision at threshold: 1.0000
- Recall at threshold: 0.2222
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (108 rows, 18 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
