# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.2_from_pack_auto)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1.2_from_pack_auto_deployable
- Samples: 366
- Positives: 107
- Negatives: 259
- Features: 29
- Spatial folds: 6

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.7725 +/- 0.1359
- Spatial AP: 0.6956
- OOF calibrated AP: 0.5981
- OOF calibrated Brier: 0.1898

## Deployment Policy
- Threshold: 0.750000
- Precision at threshold: 0.7500
- Recall at threshold: 0.1154
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (366 rows, 107 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
