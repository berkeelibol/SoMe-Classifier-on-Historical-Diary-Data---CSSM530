"""Create a balanced, non-overlapping 250-entry test set across all 4 authors.

Samples are drawn from the unlabeled pool (excluding all seed, labeled, and
to_label entries) with stratification by year for temporal coverage.

Usage:
    python src/evaluation/create_test_set.py

Output:
    data/evaluation/test_kappa_250.csv   (label columns blank — fill manually)

Default allocation: 63 Pepys, 63 Wesley, 62 Wordsworth, 62 Thoreau = 250
"""

import os
import sys
import glob
import argparse
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES   = ['VST', 'HST', 'SA', 'OR', 'WB']
OUTPUT_PATH  = os.path.join('data', 'evaluation', 'test_kappa_250.csv')
SEED         = 42

# How many entries to sample per author
AUTHOR_QUOTA = {
    'pepys':       63,
    'wesley':      63,
    'wordsworth':  62,
    'thoreau':     62,
}


def load_exclusion_set() -> set:
    """
    Collect every content string that has already been labeled or queued for
    labeling so the test set has zero overlap with training data.
    """
    excluded = set()

    # Seed items
    seed_path = os.path.join('data', 'labeled', 'seed_some_items.csv')
    if os.path.exists(seed_path):
        df = pd.read_csv(seed_path)
        excluded.update(df['content'].astype(str).tolist())

    # All round_*_labeled.csv and round_*_to_label.csv
    for pattern in ('round*_labeled.csv', 'round*_to_label.csv'):
        for path in glob.glob(os.path.join('data', 'labeled', pattern)):
            df = pd.read_csv(path)
            excluded.update(df['content'].astype(str).tolist())

    return excluded


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """
    Sample n rows from df with stratification by year.
    Each year contributes rows proportional to its share of the corpus.
    Falls back to plain random sample if year data is missing or n > len(df).
    """
    df = df.copy()
    years = pd.to_numeric(df['year'], errors='coerce')
    df['_year'] = years

    if df['_year'].isna().all() or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=seed)

    year_counts   = df['_year'].value_counts()
    total         = len(df)
    samples       = []
    remainder     = n

    # Sort years so allocation is deterministic
    for i, (year, count) in enumerate(year_counts.sort_index().items()):
        if i == len(year_counts) - 1:
            # Give all remaining quota to last year to avoid rounding loss
            quota = remainder
        else:
            quota = max(1, round(n * count / total))
            quota = min(quota, remainder)

        year_df = df[df['_year'] == year]
        take    = min(quota, len(year_df))
        samples.append(year_df.sample(n=take, random_state=seed))
        remainder -= take

        if remainder <= 0:
            break

    # If rounding left us short, top up from remaining rows
    result = pd.concat(samples).drop_duplicates(subset='content')
    if len(result) < n:
        already = set(result['content'].tolist())
        extra   = df[~df['content'].isin(already)]
        shortfall = n - len(result)
        if len(extra) >= shortfall:
            result = pd.concat([result, extra.sample(n=shortfall, random_state=seed)])

    return result.drop(columns=['_year']).head(n)


def create_test_set(author_quota: dict = AUTHOR_QUOTA, output_path: str = OUTPUT_PATH):
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading exclusion set (seed + all labeled/to_label rounds)...")
    excluded = load_exclusion_set()
    print(f"  Excluding {len(excluded)} already-seen entries\n")

    frames = []

    for author, quota in author_quota.items():
        corpus_path = os.path.join('data', 'processed', f'{author}.csv')
        if not os.path.exists(corpus_path):
            print(f"  [{author}] SKIPPED — {corpus_path} not found")
            continue

        df = pd.read_csv(corpus_path)
        before = len(df)
        df = df[~df['content'].astype(str).isin(excluded)].reset_index(drop=True)
        after  = len(df)

        print(f"  {author:<12} corpus={before}  available={after}  quota={quota}")

        if after < quota:
            print(f"    WARNING: only {after} entries available, taking all of them")
            quota = after

        sampled = stratified_sample(df, quota)
        sampled = sampled.copy()
        # Ensure author column is set (some processed CSVs may have NaN)
        sampled['author'] = author
        # Add blank label columns
        for cat in CATEGORIES:
            sampled[cat] = ''

        frames.append(sampled[['author', 'year', 'date_mm_dd', 'content', 'word_count'] + CATEGORIES])

    if not frames:
        print("No data collected — check that data/processed/ CSVs exist.")
        return

    test_df = pd.concat(frames, ignore_index=True)
    # Shuffle so authors are interleaved during annotation
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Test set saved to {output_path}")
    print(f"Total entries: {len(test_df)}")
    print(f"\nBreakdown by author:")
    for author, count in test_df['author'].value_counts().sort_index().items():
        years = pd.to_numeric(test_df[test_df['author'] == author]['year'], errors='coerce')
        print(f"  {author:<12} {count:>4} entries  "
              f"years {int(years.min())}–{int(years.max())}")

    wc = test_df['word_count']
    print(f"\nWord count — mean={wc.mean():.0f}  median={wc.median():.0f}  "
          f"min={wc.min()}  max={wc.max()}")
    print(f"\nLabel columns (VST/HST/SA/OR/WB) are blank — fill in manually.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create 250-entry test set')
    parser.add_argument('--output', default=OUTPUT_PATH,
                        help=f'Output path (default: {OUTPUT_PATH})')
    parser.add_argument('--pepys',       type=int, default=AUTHOR_QUOTA['pepys'])
    parser.add_argument('--wesley',      type=int, default=AUTHOR_QUOTA['wesley'])
    parser.add_argument('--wordsworth',  type=int, default=AUTHOR_QUOTA['wordsworth'])
    parser.add_argument('--thoreau',     type=int, default=AUTHOR_QUOTA['thoreau'])
    args = parser.parse_args()

    quota = {
        'pepys':      args.pepys,
        'wesley':     args.wesley,
        'wordsworth': args.wordsworth,
        'thoreau':    args.thoreau,
    }
    create_test_set(quota, args.output)
