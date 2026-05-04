"""Compute inter-rater reliability between two SoMe annotation sets.

Usage:
    python src/evaluation/interrater.py my_labels.csv annotator2_labels.csv

Both CSVs must have columns: content, VST, HST, SA, OR, WB

Outputs:
    results/interrater_reliability.json
    Console table with per-category Cohen's kappa and agreement %
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES = ['VST', 'HST', 'SA', 'OR', 'WB']
RESULTS_DIR = 'results'


def load_annotations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure label columns exist and are int
    for cat in CATEGORIES:
        if cat not in df.columns:
            raise ValueError(f"Missing column '{cat}' in {path}")
    df[CATEGORIES] = df[CATEGORIES].fillna(0).astype(int)
    return df


def align_annotations(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Align two annotation sets on shared content strings.
    Returns (df1_aligned, df2_aligned).
    """
    df1 = df1.set_index('content')
    df2 = df2.set_index('content')
    shared = df1.index.intersection(df2.index)

    if len(shared) == 0:
        raise ValueError(
            "No shared 'content' entries found between the two files. "
            "Make sure both files contain the same diary excerpts."
        )

    return df1.loc[shared][CATEGORIES], df2.loc[shared][CATEGORIES]


def cohen_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Compute Cohen's kappa for two binary arrays.
    kappa = (p_o - p_e) / (1 - p_e)
    """
    n = len(y1)
    if n == 0:
        return float('nan')

    # Observed agreement
    p_o = (y1 == y2).mean()

    # Expected agreement by chance
    p1_pos = y1.mean()
    p2_pos = y2.mean()
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) kappa interpretation."""
    if k < 0:
        return 'Poor'
    elif k < 0.20:
        return 'Slight'
    elif k < 0.40:
        return 'Fair'
    elif k < 0.60:
        return 'Moderate'
    elif k < 0.80:
        return 'Substantial'
    else:
        return 'Almost perfect'


def compute_interrater(path1: str, path2: str):
    print(f"\n=== Inter-rater Reliability ===")
    print(f"  Rater 1: {path1}")
    print(f"  Rater 2: {path2}")

    df1 = load_annotations(path1)
    df2 = load_annotations(path2)
    print(f"\n  Rater 1: {len(df1)} entries")
    print(f"  Rater 2: {len(df2)} entries")

    df1_aligned, df2_aligned = align_annotations(df1, df2)
    n = len(df1_aligned)
    print(f"  Shared:  {n} entries\n")

    results = {'n_shared': n, 'file1': path1, 'file2': path2, 'categories': {}}

    # ── Per-category table ────────────────────────────────────────────────
    print(f"  {'Category':<8} {'Agree%':>7} {'Kappa':>7} {'R1 pos':>7} "
          f"{'R2 pos':>7} {'Both 1':>7} {'Both 0':>7} {'Disagree':>9} {'Strength'}")
    print(f"  {'-'*80}")

    all_kappas = []
    for cat in CATEGORIES:
        y1 = df1_aligned[cat].values
        y2 = df2_aligned[cat].values

        agree = (y1 == y2).mean() * 100
        kappa = cohen_kappa(y1, y2)
        both_1 = int(((y1 == 1) & (y2 == 1)).sum())
        both_0 = int(((y1 == 0) & (y2 == 0)).sum())
        r1_only = int(((y1 == 1) & (y2 == 0)).sum())
        r2_only = int(((y1 == 0) & (y2 == 1)).sum())
        disagree = r1_only + r2_only

        strength = interpret_kappa(kappa) if not np.isnan(kappa) else 'N/A'
        kappa_str = f'{kappa:.3f}' if not np.isnan(kappa) else '  N/A'

        print(f"  {cat:<8} {agree:>6.1f}% {kappa_str:>7} {int(y1.sum()):>7} "
              f"{int(y2.sum()):>7} {both_1:>7} {both_0:>7} {disagree:>9} "
              f"  {strength}")

        results['categories'][cat] = {
            'agreement_pct': round(agree, 2),
            'cohen_kappa': round(kappa, 4) if not np.isnan(kappa) else None,
            'kappa_strength': strength,
            'n_rater1_positive': int(y1.sum()),
            'n_rater2_positive': int(y2.sum()),
            'n_both_positive': both_1,
            'n_both_negative': both_0,
            'n_rater1_only': r1_only,
            'n_rater2_only': r2_only,
            'n_disagree': disagree,
        }
        if not np.isnan(kappa):
            all_kappas.append(kappa)

    # ── Overall statistics ────────────────────────────────────────────────
    # Flatten all categories into single binary vector for overall stats
    y1_all = df1_aligned[CATEGORIES].values.ravel()
    y2_all = df2_aligned[CATEGORIES].values.ravel()
    overall_agree = (y1_all == y2_all).mean() * 100
    overall_kappa = cohen_kappa(y1_all, y2_all)

    mean_kappa = np.mean(all_kappas) if all_kappas else float('nan')

    print(f"  {'-'*80}")
    print(f"  {'Mean κ':<8} {' ':>7} {mean_kappa:>7.3f}")
    print(f"  {'Overall':<8} {overall_agree:>6.1f}% {overall_kappa:>7.3f}  "
          f"(all categories flattened)  {interpret_kappa(overall_kappa)}")

    results['overall'] = {
        'agreement_pct': round(overall_agree, 2),
        'cohen_kappa': round(overall_kappa, 4),
        'kappa_strength': interpret_kappa(overall_kappa),
        'mean_category_kappa': round(mean_kappa, 4) if not np.isnan(mean_kappa) else None,
    }

    # ── Entry-level exact match ───────────────────────────────────────────
    v1 = df1_aligned[CATEGORIES].values
    v2 = df2_aligned[CATEGORIES].values
    exact_match = (v1 == v2).all(axis=1).mean() * 100
    print(f"\n  Exact label-vector match: {exact_match:.1f}% of entries "
          f"({int((v1==v2).all(axis=1).sum())}/{n})")
    results['exact_match_pct'] = round(exact_match, 2)

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'interrater_reliability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute inter-rater reliability for SoMe annotations'
    )
    parser.add_argument('rater1', help='Path to first rater CSV')
    parser.add_argument('rater2', help='Path to second rater CSV')
    args = parser.parse_args()
    compute_interrater(args.rater1, args.rater2)
