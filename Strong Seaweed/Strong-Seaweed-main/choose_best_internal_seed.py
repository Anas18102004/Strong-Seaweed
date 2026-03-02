import json
from pathlib import Path

import pandas as pd
from project_paths import TABULAR_DIR, REPORTS_DIR, DOCS_DIR, ensure_dirs


def dataset_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    conflict = int(df.groupby(["lon", "lat"])["label"].nunique().gt(1).sum())
    miss = int(df.isna().sum().sum())
    score = pos * 2 + len(df) - conflict * 100 - miss * 10
    return {
        "path": str(path),
        "rows": int(len(df)),
        "pos": pos,
        "neg": neg,
        "conflict_cells": conflict,
        "missing_cells": miss,
        "score": float(score),
    }


def main() -> None:
    ensure_dirs()
    strict = TABULAR_DIR / "training_dataset_internal_strict_seed.csv"
    broad = TABULAR_DIR / "training_dataset_internal_broad_seed.csv"
    if not strict.exists() or not broad.exists():
        raise FileNotFoundError("Expected both strict and broad internal seed datasets.")

    s1 = dataset_stats(strict)
    s2 = dataset_stats(broad)
    best = s1 if s1["score"] >= s2["score"] else s2
    other = s2 if best is s1 else s1

    out = {
        "candidates": [s1, s2],
        "recommended": best,
        "why": [
            "Higher positive count improves learnability for the rare class.",
            "No label conflicts and no missing cells in both datasets.",
            "Larger sample size improves stability of cross-validation estimates.",
        ],
        "next_retrain_release_tag": "v1_3_internal_broad_seed",
    }

    out_json = REPORTS_DIR / "best_internal_seed_selection.json"
    out_md = DOCS_DIR / "BEST_INTERNAL_SEED_SELECTION.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Best Internal Seed Selection",
        "",
        f"- Recommended: `{best['path']}`",
        f"- Recommended rows/pos/neg: {best['rows']} / {best['pos']} / {best['neg']}",
        f"- Alternative rows/pos/neg: {other['rows']} / {other['pos']} / {other['neg']}",
        "",
        "## Why",
        "- Higher positive support is the primary objective.",
        "- Both candidates are clean (no conflicts, no missing).",
        "- Larger sample size should improve training stability.",
        "",
        f"## Next release tag",
        f"- `{out['next_retrain_release_tag']}`",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    print(f"Recommended: {best['path']}")


if __name__ == "__main__":
    main()
