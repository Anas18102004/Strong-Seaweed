# Release Workflow

## Frozen `v1.0`
- Stored in `releases/v1.0/`
- Contains:
  - `models/`
  - `reports/`
  - `snapshots/`

## Create `v1.1` (separate artifacts, no overwrite)
1. Train with release tag:
```bash
python train_realtime_production.py --release_tag v1.1
```
2. Score and save baseline snapshot into release folder:
```bash
python score_realtime_production.py --release_tag v1.1 --snapshot-tag v1_1_baseline
```

## One-command refresh with release tag
```bash
python run_realtime_refresh.py --release_tag v1.1
```

This writes `v1.1` model/report artifacts under `releases/v1.1/` and keeps `v1.0` untouched.
