# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1_2_pu)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.102121, 79.799937], lat [8.421086, 9.777542]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_2_pu_augmented_deployable
- Samples: 322
- Positives: 46
- Negatives: 276
- Features: 28
- Spatial folds: 4

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.9762 +/- 0.0228
- Spatial AP: 0.9540
- OOF calibrated AP: 0.9052
- OOF calibrated Brier: 0.0682

## Deployment Policy
- Threshold: 0.500000
- Precision at threshold: 0.8200
- Recall at threshold: 0.9535
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (322 rows, 46 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
