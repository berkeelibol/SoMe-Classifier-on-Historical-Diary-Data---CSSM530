"""Active learning sample selection for SoMe multi-label annotation.

Loads the previous round's SetFit model, scores unlabeled diary entries by
uncertainty, and saves the top-50 most uncertain for human annotation.

Usage:
    python src/training/active_learning.py <round_number> <author>

Example:
    python src/training/active_learning.py 1 pepys
    python src/training/active_learning.py 2 wesley

Reads:  models/setfit_round{N-1}/
        data/processed/{author}.csv
        data/labeled/seed_some_items.csv       (to exclude already-labeled)
        data/labeled/round{i}_labeled.csv      (for i in 1..N-1, to exclude)

Writes: data/labeled/round{N}_to_label.csv    (label columns blank for manual annotation)

After running, open the CSV, fill in VST/HST/SA/OR/WB (0 or 1), and save as
data/labeled/round{N}_labeled.csv before retraining.
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES = ['VST', 'HST', 'SA', 'OR', 'WB']
SEED = 42
N_UNCERTAIN = 25
N_CERTAIN = 25


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_already_labeled(round_num: int) -> set:
    """Return set of content strings that are already labeled."""
    labeled_contents = set()

    seed_path = os.path.join('data', 'labeled', 'seed_some_items.csv')
    if os.path.exists(seed_path):
        df = pd.read_csv(seed_path)
        labeled_contents.update(df['content'].astype(str).tolist())

    for i in range(1, round_num):
        path = os.path.join('data', 'labeled', f'round{i}_labeled.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            labeled_contents.update(df['content'].astype(str).tolist())

    return labeled_contents


def uncertainty_score(probs: np.ndarray) -> np.ndarray:
    """
    Compute uncertainty score for each entry.

    Strategy: for each entry take the minimum distance from 0.5
    across all 5 category probabilities — the entry is "most uncertain"
    when ANY category prediction is close to 0.5.

    Score = min over categories of |prob_k - 0.5|
    Lower score = more uncertain.
    We negate for ranking (higher = more uncertain).
    """
    distances = np.abs(probs - 0.5)          # (n, 5)
    min_dist = distances.min(axis=1)          # (n,)  most uncertain dimension
    return -min_dist                          # higher = more uncertain


def run_active_learning(round_num: int, author: str):
    from setfit import SetFitModel

    set_seed(SEED)

    # ── Load model from previous round ──────────────────────────────────────
    prev_round = round_num - 1
    model_path = os.path.join('models', f'setfit_round{prev_round}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model for round {prev_round} not found at {model_path}. "
            f"Run train_setfit.py {prev_round} first."
        )
    print(f"Loading model from {model_path}...")
    model = SetFitModel.from_pretrained(model_path)

    # ── Load processed diary data ────────────────────────────────────────────
    diary_path = os.path.join('data', 'processed', f'{author}.csv')
    if not os.path.exists(diary_path):
        raise FileNotFoundError(f"Processed diary not found: {diary_path}")
    df_diary = pd.read_csv(diary_path)
    print(f"Loaded {len(df_diary)} entries from {diary_path}")

    # ── Exclude already-labeled entries ──────────────────────────────────────
    already_labeled = load_already_labeled(round_num)
    print(f"Excluding {len(already_labeled)} already-labeled entries...")
    mask = ~df_diary['content'].astype(str).isin(already_labeled)
    df_unlabeled = df_diary[mask].reset_index(drop=True)
    print(f"Remaining unlabeled: {len(df_unlabeled)} entries")

    if len(df_unlabeled) == 0:
        print("No unlabeled entries remaining for this author.")
        return

    # ── Score all unlabeled entries ───────────────────────────────────────────
    texts = df_unlabeled['content'].tolist()
    print(f"Scoring {len(texts)} entries...")

    probs = model.predict_proba(texts, as_numpy=True)   # (n, 5)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(CATEGORIES))

    scores = uncertainty_score(probs)   # (n,)  higher = more uncertain

    df_unlabeled = df_unlabeled.copy()
    df_unlabeled['_uncertainty'] = scores
    for i, cat in enumerate(CATEGORIES):
        df_unlabeled[f'_prob_{cat}'] = probs[:, i]

    # Print uncertainty distribution
    print(f"\nUncertainty score distribution (negated min-dist-from-0.5):")
    print(f"  Mean:    {scores.mean():.4f}")
    print(f"  Median:  {np.median(scores):.4f}")
    print(f"  Max:     {scores.max():.4f}  (most uncertain)")
    print(f"  Min:     {scores.min():.4f}  (most certain)")

    # ── Select 25 most uncertain + 25 most certain ───────────────────────────
    df_uncertain = df_unlabeled.nlargest(N_UNCERTAIN, '_uncertainty')
    df_certain = df_unlabeled.nsmallest(N_CERTAIN, '_uncertainty')

    df_uncertain = df_uncertain.copy()
    df_certain = df_certain.copy()
    df_uncertain['_split'] = 'uncertain'
    df_certain['_split'] = 'certain'

    df_selected = pd.concat([df_uncertain, df_certain]).reset_index(drop=True)

    print(f"\nSelected {len(df_selected)} entries for labeling (round {round_num})")
    print(f"  {N_UNCERTAIN} most uncertain  |  {N_CERTAIN} most certain")
    print(f"\n  Avg predicted probabilities:")
    print(f"  {'Cat':<5} {'Uncertain':>10} {'Certain':>10}")
    for cat in CATEGORIES:
        col = f'_prob_{cat}'
        u = df_uncertain[col].mean()
        c = df_certain[col].mean()
        print(f"  {cat:<5} {u:>10.3f} {c:>10.3f}")

    # ── Build output CSV ──────────────────────────────────────────────────────
    meta_cols = ['content', 'author', 'year', 'date_mm_dd', '_split']
    for col in ['author', 'year', 'date_mm_dd']:
        if col not in df_selected.columns:
            df_selected[col] = ''

    df_out = df_selected[meta_cols].copy()
    df_out = df_out.rename(columns={'_split': 'split'})
    for cat in CATEGORIES:
        df_out[cat] = ''

    out_path = os.path.join('data', 'labeled', f'round{round_num}_to_label.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Manually fill in the VST/HST/SA/OR/WB columns (0 or 1),")
    print(f"then save as: data/labeled/round{round_num}_labeled.csv")

    # Save selection metadata
    meta = {
        'round': round_num,
        'author': author,
        'model_path': model_path,
        'n_unlabeled_scored': len(df_unlabeled),
        'n_uncertain': N_UNCERTAIN,
        'n_certain': N_CERTAIN,
        'n_selected': len(df_selected),
        'uncertainty_stats': {
            'mean': float(scores.mean()),
            'median': float(np.median(scores)),
            'max': float(scores.max()),
            'min': float(scores.min()),
        },
    }
    meta_path = os.path.join('data', 'labeled', f'round{round_num}_selection_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Active learning: select uncertain diary entries for labeling'
    )
    parser.add_argument('round', type=int,
                        help='Round number (selects for this round, uses round-1 model)')
    parser.add_argument('author', type=str,
                        help='Author name (pepys | wesley | wordsworth)')
    args = parser.parse_args()
    run_active_learning(args.round, args.author)
