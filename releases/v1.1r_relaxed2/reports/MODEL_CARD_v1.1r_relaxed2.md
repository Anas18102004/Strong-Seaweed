# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.1r_relaxed2)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_1_merged46_plus_hn30_augmented_plus_research_relaxed2_deployable
- Samples: 307
- Positives: 48
- Negatives: 259
- Features: 29
- Spatial folds: 5

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.6146 +/- 0.1815
- Spatial AP: 0.4255
- OOF calibrated AP: 0.2905
- OOF calibrated Brier: 0.1716

## Deployment Policy
- Threshold: 0.218750
- Precision at threshold: 0.2756
- Recall at threshold: 0.9556
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (307 rows, 48 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
