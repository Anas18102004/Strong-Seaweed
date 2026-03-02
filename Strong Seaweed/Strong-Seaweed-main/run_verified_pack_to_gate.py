import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="From reviewed verification pack to strict ingestion, retrain, and gate report."
    )
    p.add_argument(
        "--reviewed_pack_csv",
        type=Path,
        default=Path("artifacts/experiments/v1_2_verification_pack_top50.csv"),
    )
    p.add_argument(
        "--base_training_csv",
        type=Path,
        default=Path("data/tabular/training_dataset_v1_1_merged46_plus_hn30_augmented_plus11web.csv"),
    )
    p.add_argument(
        "--master_csv",
        type=Path,
        default=Path("data/tabular/master_feature_matrix_v1_1.csv"),
    )
    p.add_argument(
        "--master_aug_csv",
        type=Path,
        default=Path("data/tabular/master_feature_matrix_v1_1_augmented.csv"),
    )
    p.add_argument(
        "--release_tag",
        type=str,
        default="v1.2_strict_from_pack",
    )
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--max_snap_m",
        type=float,
        default=1500.0,
    )
    p.add_argument(
        "--allow_provisional_fallback",
        action="store_true",
        help="If no strict-verified rows, fallback to provisional accepted rows for continuous iteration.",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def accepted_mask(df: pd.DataFrame) -> pd.Series:
    decision = df.get("verification_decision", "").astype(str).str.strip().str.lower()
    accepted = decision.isin({"accept", "accepted", "approve", "approved", "yes", "true", "1"})
    return accepted


def confidence_col(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(df.get("verification_confidence", pd.NA), errors="coerce")
    if c.isna().all():
        c = pd.to_numeric(df.get("p_calibrated", 0.0), errors="coerce")
    return c.fillna(0.0)


def has_verified_dates(df: pd.DataFrame) -> pd.Series:
    vd = df.get("verified_dates", "").astype(str)
    return vd.str.contains(";", na=False)


def pack_to_verified_ingestion(
    reviewed_pack_csv: Path,
    out_csv: Path,
    min_confidence: float,
    provisional: bool,
) -> int:
    if not reviewed_pack_csv.exists():
        raise FileNotFoundError(f"Missing reviewed pack: {reviewed_pack_csv}")

    df = pd.read_csv(reviewed_pack_csv)
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("reviewed pack must contain lon/lat columns")

    if provisional:
        keep = pd.Series([True] * len(df))
        qa_status = "pending"
        is_verified = False
        species_confirmed = False
    else:
        keep = accepted_mask(df) & has_verified_dates(df) & (confidence_col(df) >= float(min_confidence))
        qa_status = "approved"
        is_verified = True
        species_confirmed = True

    out = df[keep].copy()
    if out.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "record_id",
                "source_type",
                "source_name",
                "source_reference",
                "citation_url",
                "species",
                "eventDate",
                "year",
                "lon",
                "lat",
                "label",
                "coordinate_precision_km",
                "species_confirmed",
                "confidence_score",
                "is_verified",
                "qa_reviewer",
                "qa_status",
                "rationale",
                "notes",
            ]
        ).to_csv(out_csv, index=False)
        return 0

    out2 = pd.DataFrame()
    out2["record_id"] = out.get("record_id", "").astype(str)
    out2["source_type"] = "satellite_digitized"
    out2["source_name"] = "v1_2_verification_pack_top50"
    out2["source_reference"] = "verified from reviewed pack"
    out2["citation_url"] = out.get("evidence_url_1", "").astype(str)
    out2["species"] = "Kappaphycus alvarezii"
    out2["eventDate"] = ""
    out2["year"] = pd.NA
    out2["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out2["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out2["label"] = 1
    out2["coordinate_precision_km"] = 1.0
    out2["species_confirmed"] = species_confirmed
    out2["confidence_score"] = confidence_col(out)
    out2["is_verified"] = is_verified
    out2["qa_reviewer"] = out.get("reviewer", "").astype(str)
    out2["qa_status"] = qa_status
    out2["rationale"] = "reviewed_verification_pack_candidate"
    out2["notes"] = (
        "verified_dates="
        + out.get("verified_dates", "").astype(str).replace({"": "YYYY-MM-DD;YYYY-MM-DD;YYYY-MM-DD"})
    )
    out2 = out2.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out2.to_csv(out_csv, index=False)
    return int(len(out2))


def merge_snapped_into_training(
    base_training_csv: Path,
    master_aug_csv: Path,
    snapped_csv: Path,
    out_csv: Path,
) -> tuple[int, int, int]:
    base = pd.read_csv(base_training_csv)
    master_aug = pd.read_csv(master_aug_csv)
    snap = pd.read_csv(snapped_csv)

    if snap.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(out_csv, index=False)
        pos = int((base["label"] == 1).sum())
        return 0, pos, int(len(base))

    pos_set = set(zip(base[base["label"] == 1]["lon"].round(8), base[base["label"] == 1]["lat"].round(8)))
    snap["k"] = list(zip(snap["lon"].round(8), snap["lat"].round(8)))
    snap = snap[~snap["k"].isin(pos_set)].copy()
    if snap.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(out_csv, index=False)
        pos = int((base["label"] == 1).sum())
        return 0, pos, int(len(base))

    rows = snap[["lon", "lat"]].merge(master_aug, on=["lon", "lat"], how="left")
    rows["label"] = 1
    for c in base.columns:
        if c not in rows.columns:
            rows[c] = 0
    rows = rows[base.columns].dropna().drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)

    merged = pd.concat([base, rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["lon", "lat", "label"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return int(len(rows)), int((merged["label"] == 1).sum()), int(len(merged))


def write_gate_summary(release_tag: str, out_json: Path, out_md: Path) -> None:
    train_report = Path(f"releases/{release_tag}/reports/xgboost_realtime_report_{release_tag}.json")
    hard50 = Path(f"artifacts/reports/{release_tag}_hard50_eval.json")
    nonoverlap = Path(f"artifacts/reports/{release_tag}_hard50_eval_nonoverlap.json")
    if not (train_report.exists() and hard50.exists() and nonoverlap.exists()):
        raise FileNotFoundError("Missing one of train_report/hard50/nonoverlap for gate summary.")

    train = json.loads(train_report.read_text(encoding="utf-8"))
    hard = json.loads(hard50.read_text(encoding="utf-8"))
    non = json.loads(nonoverlap.read_text(encoding="utf-8"))
    sm = train.get("selected_metrics", {})

    checks = [
        ("train_spatial_auc>=0.70", float(sm.get("spatial_auc_mean", 0)) >= 0.70),
        ("train_oof_ap_cal>=0.45", float(sm.get("oof_ap_calibrated", 0)) >= 0.45),
        ("hard50_auc>=0.85", float(hard.get("roc_auc", 0)) >= 0.85),
        ("hard50_precision>=0.80", float(hard.get("precision", 0)) >= 0.80),
        ("hard50_recall>=0.40", float(hard.get("recall", 0)) >= 0.40),
        ("independent_n>=40", int(non.get("n", 0)) >= 40),
        ("independent_pos>=20", int(non.get("positives", 0)) >= 20),
        ("independent_neg>=20", int(non.get("negatives", 0)) >= 20),
    ]

    hard_pass = all(ok for _, ok in checks[:5])
    indep_ready = all(ok for _, ok in checks[5:8])
    if hard_pass and indep_ready:
        gate = "PASS"
    elif hard_pass:
        gate = "WARN"
    else:
        gate = "FAIL"

    obj = {
        "release": release_tag,
        "gate_result": gate,
        "checks": [{"name": n, "pass": ok} for n, ok in checks],
        "metrics": {
            "train_spatial_auc": sm.get("spatial_auc_mean"),
            "train_oof_ap_calibrated": sm.get("oof_ap_calibrated"),
            "hard50_auc": hard.get("roc_auc"),
            "hard50_precision": hard.get("precision"),
            "hard50_recall": hard.get("recall"),
            "independent_n": non.get("n"),
            "independent_pos": non.get("positives"),
            "independent_neg": non.get("negatives"),
            "independent_auc": non.get("roc_auc"),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    lines = [f"# Production Gate: {release_tag}", "", f"- Result: **{gate}**", "", "## Checks"]
    for n, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} | {n}")
    lines += [
        "",
        "## Metrics",
        f"- train_spatial_auc: {obj['metrics']['train_spatial_auc']}",
        f"- train_oof_ap_calibrated: {obj['metrics']['train_oof_ap_calibrated']}",
        f"- hard50_auc: {obj['metrics']['hard50_auc']}",
        f"- hard50_precision: {obj['metrics']['hard50_precision']}",
        f"- hard50_recall: {obj['metrics']['hard50_recall']}",
        f"- independent_n: {obj['metrics']['independent_n']}",
        f"- independent_pos: {obj['metrics']['independent_pos']}",
        f"- independent_neg: {obj['metrics']['independent_neg']}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def evaluate_nonoverlap(release_tag: str) -> None:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    eval_csv = Path(f"artifacts/experiments/{release_tag}_hard50_eval.csv")
    train_csv = Path(f"data/tabular/training_dataset_{release_tag}.csv")
    report_json = Path(f"releases/{release_tag}/reports/xgboost_realtime_report_{release_tag}.json")
    out_csv = Path(f"artifacts/experiments/{release_tag}_hard50_eval_nonoverlap.csv")
    out_json = Path(f"artifacts/reports/{release_tag}_hard50_eval_nonoverlap.json")

    df = pd.read_csv(eval_csv)
    tr = pd.read_csv(train_csv)
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    thr = float(rep["deployment_policy"]["recommended_threshold"])
    cells = set(zip(tr["lon"].round(8), tr["lat"].round(8)))
    df["k"] = list(zip(df["snap_lon"].round(8), df["snap_lat"].round(8)))
    h = df[~df["k"].isin(cells)].copy()

    if len(h) > 0 and h["expected"].nunique() == 2:
        y = h["expected"].astype(int).to_numpy()
        p = h["p_calibrated"].astype(float).to_numpy()
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        out = {
            "release": release_tag,
            "n": int(len(h)),
            "positives": int((y == 1).sum()),
            "negatives": int((y == 0).sum()),
            "threshold": thr,
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "avg_precision": float(average_precision_score(y, p)),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        }
    else:
        out = {
            "release": release_tag,
            "n": int(len(h)),
            "positives": int((h["expected"] == 1).sum()) if len(h) else 0,
            "negatives": int((h["expected"] == 0).sum()) if len(h) else 0,
            "note": "insufficient class balance",
        }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    tag = args.release_tag.strip()
    verified_csv = Path(f"data/tabular/{tag}_verified_from_pack.csv")
    snapped_csv = Path(f"artifacts/experiments/{tag}_verified_snapped_strict.csv")
    ingest_report = Path(f"artifacts/reports/{tag}_verified_ingestion_report.json")
    merged_training = Path(f"data/tabular/training_dataset_{tag}.csv")

    # 1) Convert reviewed pack to strict ingestion schema.
    strict_count = pack_to_verified_ingestion(
        reviewed_pack_csv=args.reviewed_pack_csv,
        out_csv=verified_csv,
        min_confidence=float(args.min_confidence),
        provisional=False,
    )
    provisional_mode = False
    if strict_count == 0 and args.allow_provisional_fallback:
        strict_count = pack_to_verified_ingestion(
            reviewed_pack_csv=args.reviewed_pack_csv,
            out_csv=verified_csv,
            min_confidence=float(args.min_confidence),
            provisional=True,
        )
        provisional_mode = True

    # 2) Ingest (strict if not provisional fallback).
    cmd = [
        "python",
        "ingest_presence_records.py",
        "--inputs",
        str(verified_csv),
        "--master_csv",
        str(args.master_csv),
        "--training_csv",
        str(args.base_training_csv),
        "--max_snap_m",
        str(float(args.max_snap_m)),
        "--species_filter",
        "kappaphycus",
        "--out_csv",
        str(snapped_csv),
        "--out_report",
        str(ingest_report),
    ]
    if not provisional_mode:
        cmd.extend(
            [
                "--strict_acceptance",
                "--strict_min_confidence",
                str(float(args.min_confidence)),
                "--strict_min_verified_dates",
                "2",
            ]
        )
    run(cmd)

    # 3) Merge snapped positives into base training set.
    added, merged_pos, merged_rows = merge_snapped_into_training(
        base_training_csv=args.base_training_csv,
        master_aug_csv=args.master_aug_csv,
        snapped_csv=snapped_csv,
        out_csv=merged_training,
    )
    print(
        f"merged_training={merged_training} | added_positive_rows={added} | merged_pos={merged_pos} | merged_rows={merged_rows}",
        flush=True,
    )

    # 4) Train release.
    run(
        [
            "python",
            "train_realtime_production.py",
            "--fast",
            "--production",
            "--release_tag",
            tag,
            "--dataset_paths",
            str(merged_training),
            "--inference_feature_source",
            str(args.master_aug_csv),
        ]
    )

    # 5) Hard50 eval.
    run(
        [
            "python",
            "-c",
            (
                "import json,joblib,pandas as pd,numpy as np;"
                f"r='{tag}';"
                "from pathlib import Path;"
                "from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix,accuracy_score;"
                "base=Path('releases')/r;"
                "df=pd.read_csv('artifacts/experiments/v1_1_hard_test_50_web_with_features.csv');"
                "feat=json.loads((base/'models'/f'xgboost_realtime_features_{r}.json').read_text());"
                "rep=json.loads((base/'reports'/f'xgboost_realtime_report_{r}.json').read_text());"
                "thr=float(rep['deployment_policy']['recommended_threshold']);"
                "mods=joblib.load(base/'models'/f'xgboost_realtime_ensemble_{r}.pkl');"
                "cal=joblib.load(base/'models'/f'xgboost_realtime_calibrator_{r}.pkl');"
                "X=df[feat].to_numpy(dtype=np.float32);"
                "raw=np.mean(np.vstack([m.predict_proba(X)[:,1] for m in mods]),axis=0);"
                "p=cal.predict(raw);pred=(p>=thr).astype(int);y=df['expected'].astype(int).to_numpy();"
                "cm=confusion_matrix(y,pred,labels=[0,1]);"
                "out={'release':r,'n':int(len(df)),'positives':int((y==1).sum()),'negatives':int((y==0).sum()),"
                "'threshold':thr,'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),"
                "'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),"
                "'roc_auc':float(roc_auc_score(y,p)),'avg_precision':float(average_precision_score(y,p)),"
                "'confusion_matrix':{'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1])}};"
                "df['p_raw']=raw;df['p_calibrated']=p;df['pred_label']=pred;"
                "df.to_csv(f'artifacts/experiments/{r}_hard50_eval.csv',index=False);"
                "Path(f'artifacts/reports/{r}_hard50_eval.json').write_text(json.dumps(out,indent=2));"
                "print(json.dumps(out,indent=2));"
            ),
        ]
    )

    # 6) Non-overlap eval.
    evaluate_nonoverlap(tag)

    # 7) Production gate summary.
    gate_json = Path(f"artifacts/reports/production_gate_{tag}.json")
    gate_md = Path(f"docs/PRODUCTION_GATE_{tag}.md")
    write_gate_summary(tag, gate_json, gate_md)
    print(f"gate_json={gate_json}")
    print(f"gate_md={gate_md}")


if __name__ == "__main__":
    main()
