import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from project_paths import REPORTS_DIR, TABULAR_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict geographic holdout gate for multispecies datasets.")
    p.add_argument("--dataset_glob", type=str, default="training_dataset_*_multispecies_cop_india_v2_prod_softpos_v4_grid.csv")
    p.add_argument("--grid_deg", type=float, default=1.0)
    p.add_argument("--test_pct", type=float, default=0.20)
    p.add_argument("--val_pct", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_rows", type=int, default=500)
    p.add_argument("--min_pos_train", type=int, default=30)
    p.add_argument("--min_pos_val", type=int, default=10)
    p.add_argument("--min_pos_test", type=int, default=10)
    p.add_argument("--min_auc", type=float, default=0.80)
    p.add_argument("--min_ap", type=float, default=0.35)
    p.add_argument("--max_brier", type=float, default=0.20)
    p.add_argument("--max_auc_gap", type=float, default=0.15, help="Max |AUC_train - AUC_test| allowed.")
    p.add_argument("--out_tag", type=str, default="multispecies_softpos_v4_grid")
    p.add_argument("--exclude_species", type=str, default="kappaphycus_alvarezii")
    return p.parse_args()


def _hash_unit(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _spatial_split(df: pd.DataFrame, grid_deg: float, val_pct: float, test_pct: float) -> pd.Series:
    lon_cell = np.floor(df["lon"].to_numpy(dtype=np.float64) / grid_deg).astype(int)
    lat_cell = np.floor(df["lat"].to_numpy(dtype=np.float64) / grid_deg).astype(int)
    out = []
    for a, b in zip(lon_cell, lat_cell):
        u = _hash_unit(f"{a}_{b}")
        if u < test_pct:
            out.append("test")
        elif u < test_pct + val_pct:
            out.append("val")
        else:
            out.append("train")
    return pd.Series(out, index=df.index, dtype="object")


def _fit_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> XGBClassifier:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / max(pos, 1)
    m = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_estimators=120,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.50,
        colsample_bytree=0.50,
        min_child_weight=20,
        gamma=1.0,
        reg_alpha=2.0,
        reg_lambda=10.0,
        max_delta_step=2,
        scale_pos_weight=spw,
        n_jobs=-1,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _evaluate_species(df: pd.DataFrame, seed: int, grid_deg: float, val_pct: float, test_pct: float) -> dict:
    drop_cols = {
        "label",
        "label_weight",
        "species_id",
        "species_target",
        "sample_type",
        "nearest_hp_km",
        "soft_similarity",
        "lon",
        "lat",
    }
    feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 1]
    if not feat_cols:
        raise RuntimeError("no numeric feature columns available")

    df = df.dropna(subset=["lon", "lat", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if "label_weight" in df.columns:
        df["label_weight"] = pd.to_numeric(df["label_weight"], errors="coerce").fillna(1.0).clip(lower=0.05)
    else:
        df["label_weight"] = 1.0
    df = df.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
    df["split"] = _spatial_split(df, grid_deg=grid_deg, val_pct=val_pct, test_pct=test_pct)

    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    x_tr = tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = tr["label"].to_numpy(dtype=np.int32)
    w_tr = tr["label_weight"].to_numpy(dtype=np.float32)
    x_va = va[feat_cols].to_numpy(dtype=np.float32)
    y_va = va["label"].to_numpy(dtype=np.int32)
    x_te = te[feat_cols].to_numpy(dtype=np.float32)
    y_te = te["label"].to_numpy(dtype=np.int32)

    model = _fit_model(x_tr, y_tr, w_tr, seed=seed)
    p_tr = model.predict_proba(x_tr)[:, 1]
    p_va_raw = model.predict_proba(x_va)[:, 1]
    p_te_raw = model.predict_proba(x_te)[:, 1]

    # Calibrate on held-out validation only.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_va_raw, y_va)
    p_va = calibrator.predict(p_va_raw)
    p_te = calibrator.predict(p_te_raw)

    return {
        "rows": int(len(df)),
        "positives": int(df["label"].sum()),
        "negatives": int((df["label"] == 0).sum()),
        "train_rows": int(len(tr)),
        "val_rows": int(len(va)),
        "test_rows": int(len(te)),
        "train_pos": int(y_tr.sum()),
        "val_pos": int(y_va.sum()),
        "test_pos": int(y_te.sum()),
        "features": feat_cols,
        "metrics": {
            "train_auc_raw": _safe_auc(y_tr, p_tr),
            "val_auc_cal": _safe_auc(y_va, p_va),
            "test_auc_cal": _safe_auc(y_te, p_te),
            "test_ap_cal": _safe_ap(y_te, p_te),
            "test_brier_cal": float(brier_score_loss(y_te, p_te)),
        },
    }


def main() -> None:
    ensure_dirs()
    args = parse_args()
    paths = sorted(TABULAR_DIR.glob(args.dataset_glob))
    if not paths:
        raise FileNotFoundError(f"No dataset files matched: {args.dataset_glob}")

    excluded = {x.strip() for x in str(args.exclude_species).split(",") if x.strip()}
    species_results = []
    for p in paths:
        df = pd.read_csv(p)
        if "species_id" not in df.columns:
            continue
        species_id = str(df["species_id"].dropna().iloc[0]) if len(df) else p.stem
        if species_id in excluded:
            species_results.append({"species_id": species_id, "status": "skip", "reason": "excluded_species"})
            continue
        df = df.dropna(subset=["lon", "lat", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        n_rows = int(len(df))
        n_pos = int(df["label"].sum())
        if n_rows < int(args.min_rows):
            species_results.append(
                {"species_id": species_id, "status": "skip", "reason": "insufficient_rows", "rows": n_rows, "pos": n_pos}
            )
            continue
        try:
            r = _evaluate_species(df, seed=args.seed, grid_deg=args.grid_deg, val_pct=args.val_pct, test_pct=args.test_pct)
        except Exception as e:
            species_results.append({"species_id": species_id, "status": "fail", "reason": f"eval_error: {e}"})
            continue

        checks = {
            "min_pos_train": r["train_pos"] >= int(args.min_pos_train),
            "min_pos_val": r["val_pos"] >= int(args.min_pos_val),
            "min_pos_test": r["test_pos"] >= int(args.min_pos_test),
            "auc_test": float(r["metrics"]["test_auc_cal"]) >= float(args.min_auc),
            "ap_test": float(r["metrics"]["test_ap_cal"]) >= float(args.min_ap),
            "brier_test": float(r["metrics"]["test_brier_cal"]) <= float(args.max_brier),
            "auc_gap": abs(float(r["metrics"]["train_auc_raw"]) - float(r["metrics"]["test_auc_cal"])) <= float(args.max_auc_gap),
        }
        r["checks"] = checks
        r["status"] = "pass" if all(checks.values()) else "fail"
        species_results.append({"species_id": species_id, **r})

    trained = [x for x in species_results if x.get("status") in {"pass", "fail"}]
    gate_pass = len(trained) > 0 and all(x.get("status") == "pass" for x in trained)
    out = {
        "gate": "strict_geographic_holdout_multispecies",
        "out_tag": args.out_tag,
        "dataset_glob": args.dataset_glob,
        "thresholds": {
            "min_rows": args.min_rows,
            "min_pos_train": args.min_pos_train,
            "min_pos_val": args.min_pos_val,
            "min_pos_test": args.min_pos_test,
            "min_auc": args.min_auc,
            "min_ap": args.min_ap,
            "max_brier": args.max_brier,
            "max_auc_gap": args.max_auc_gap,
            "grid_deg": args.grid_deg,
            "val_pct": args.val_pct,
            "test_pct": args.test_pct,
        },
        "species": species_results,
        "decision": "PASS" if gate_pass else "FAIL",
    }

    out_json = REPORTS_DIR / f"strict_geo_holdout_gate_{args.out_tag}.json"
    out_md = Path("docs") / f"STRICT_GEO_HOLDOUT_GATE_{args.out_tag}.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Strict Geographic Holdout Gate",
        "",
        f"- Decision: **{out['decision']}**",
        f"- Dataset glob: `{args.dataset_glob}`",
        "",
    ]
    for s in species_results:
        sid = s.get("species_id", "unknown")
        status = s.get("status", "skip")
        lines.append(f"## {sid}")
        lines.append(f"- Status: **{status.upper()}**")
        if status == "skip":
            lines.append(f"- Reason: {s.get('reason')}")
            lines.append("")
            continue
        if status == "fail" and "metrics" not in s:
            lines.append(f"- Reason: {s.get('reason')}")
            lines.append("")
            continue
        m = s["metrics"]
        lines.append(
            f"- Rows train/val/test: {s['train_rows']}/{s['val_rows']}/{s['test_rows']} | "
            f"Pos train/val/test: {s['train_pos']}/{s['val_pos']}/{s['test_pos']}"
        )
        lines.append(
            f"- Test AUC/AP/Brier (cal): {m['test_auc_cal']:.4f}/{m['test_ap_cal']:.4f}/{m['test_brier_cal']:.4f}"
        )
        lines.append(f"- Train AUC raw: {m['train_auc_raw']:.4f}")
        failed_checks = [k for k, ok in s.get("checks", {}).items() if not ok]
        lines.append(f"- Failed checks: {', '.join(failed_checks) if failed_checks else 'none'}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_json}")
    print(f"[DONE] {out_md}")
    print(f"Decision: {out['decision']}")
    if out["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
