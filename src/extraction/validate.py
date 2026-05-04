"""Validate and inspect processed diary CSV files."""

import re
import sys
import random
import argparse
import pandas as pd
import numpy as np


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*60}")


def check_word_distribution(df: pd.DataFrame):
    """Print word count distribution statistics."""
    print_separator("Word Count Distribution")
    wc = df['word_count']
    print(f"  Count:  {len(wc)}")
    print(f"  Mean:   {wc.mean():.1f}")
    print(f"  Median: {wc.median():.1f}")
    print(f"  Std:    {wc.std():.1f}")
    print(f"  Min:    {wc.min()}")
    print(f"  Max:    {wc.max()}")
    print(f"  Q25:    {wc.quantile(0.25):.0f}")
    print(f"  Q75:    {wc.quantile(0.75):.0f}")

    # Simple text histogram
    print("\n  Histogram (word counts):")
    bins = [0, 30, 50, 100, 150, 200, 300, 500, 1000]
    for j in range(len(bins) - 1):
        count = ((wc >= bins[j]) & (wc < bins[j + 1])).sum()
        bar = '#' * min(count, 50)
        print(f"  {bins[j]:>4}-{bins[j+1]:>4}: {count:>5} {bar}")
    overflow = (wc >= bins[-1]).sum()
    if overflow:
        print(f"  {bins[-1]:>4}+   : {overflow:>5} {'#' * min(overflow, 50)}")


def check_date_coverage(df: pd.DataFrame):
    """Analyze date coverage: years, months, gaps."""
    print_separator("Date Coverage")

    # Year distribution
    years = df['year'].dropna()
    if years.empty:
        print("  No year data available.")
        return

    # Convert to numeric, coercing errors
    years_numeric = pd.to_numeric(years, errors='coerce').dropna()
    if years_numeric.empty:
        print("  Could not parse year data.")
        return

    print(f"  Year range: {int(years_numeric.min())} - {int(years_numeric.max())}")
    year_counts = years_numeric.value_counts().sort_index()
    print(f"  Years with entries: {len(year_counts)}")
    print("\n  Entries per year:")
    for year, count in year_counts.items():
        bar = '#' * min(count, 40)
        print(f"    {int(year)}: {count:>4} {bar}")

    # Check for month coverage within each year
    print("\n  Month coverage by year:")
    for year in sorted(year_counts.index):
        year_df = df[pd.to_numeric(df['year'], errors='coerce') == year]
        months = set()
        for d in year_df['date_mm_dd'].dropna():
            parts = str(d).split('-')
            if len(parts) >= 1:
                try:
                    m = int(parts[0])
                    if 1 <= m <= 12:
                        months.add(m)
                except (ValueError, IndexError):
                    pass
        month_str = ', '.join(str(m) for m in sorted(months))
        missing = set(range(1, 13)) - months
        gap_str = f" (missing: {', '.join(str(m) for m in sorted(missing))})" if missing else " (complete)"
        print(f"    {int(year)}: months [{month_str}]{gap_str}")


def check_suspicious_entries(df: pd.DataFrame):
    """Flag entries that may contain artifacts or problems."""
    print_separator("Suspicious Entries")

    issues_found = False

    # Very short entries (below segmentation minimum — shouldn't happen)
    short = df[df['word_count'] < 20]
    if len(short) > 0:
        issues_found = True
        print(f"\n  Very short entries (<20 words): {len(short)}")
        for _, row in short.head(5).iterrows():
            print(f"    [{row['year']} {row['date_mm_dd']}] ({row['word_count']}w): {row['content'][:80]}...")

    # Very long entries (above segmentation maximum — shouldn't happen)
    long = df[df['word_count'] > 350]
    if len(long) > 0:
        issues_found = True
        print(f"\n  Very long entries (>350 words): {len(long)}")
        for _, row in long.head(5).iterrows():
            print(f"    [{row['year']} {row['date_mm_dd']}] ({row['word_count']}w): {row['content'][:80]}...")

    # OCR artifact detection: unusual character patterns
    artifact_patterns = [
        (r'[^\x00-\x7F]{3,}', 'Non-ASCII sequences'),
        (r'[A-Z]{10,}', 'Long uppercase sequences'),
        (r'\d{5,}', 'Long number sequences'),
        (r'(.)\1{4,}', 'Repeated characters'),
        (r'[|}{~`\\]{2,}', 'Unusual punctuation clusters'),
    ]

    for pattern, desc in artifact_patterns:
        matches = df[df['content'].str.contains(pattern, regex=True, na=False)]
        if len(matches) > 0:
            issues_found = True
            print(f"\n  {desc}: {len(matches)} entries")
            for _, row in matches.head(3).iterrows():
                # Find the match
                m = re.search(pattern, row['content'])
                context = row['content'][max(0, m.start()-20):m.end()+20] if m else row['content'][:60]
                print(f"    [{row['year']} {row['date_mm_dd']}]: ...{context}...")

    if not issues_found:
        print("  No suspicious entries found.")


def print_random_entries(df: pd.DataFrame, n: int = 10, seed: int = 42):
    """Print random entries for manual inspection."""
    print_separator(f"Random Sample ({n} entries)")
    random.seed(seed)
    indices = random.sample(range(len(df)), min(n, len(df)))
    for idx in indices:
        row = df.iloc[idx]
        print(f"\n  [{row['year']} {row['date_mm_dd']}] ({row['word_count']}w)")
        content = row['content']
        # Wrap at ~80 chars
        words = content.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 80:
                print(line)
                line = "    " + word
            else:
                line = line + " " + word if line.strip() else "    " + word
        if line.strip():
            print(line)


def validate(csv_path: str, sample_size: int = 10, seed: int = 42):
    """Run all validation checks on a processed CSV."""
    print(f"Validating: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Author(s): {df['author'].unique()}")

    check_word_distribution(df)
    check_date_coverage(df)
    check_suspicious_entries(df)
    print_random_entries(df, n=sample_size, seed=seed)

    print_separator("Validation Complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate processed diary CSV')
    parser.add_argument('csv_path', help='Path to processed CSV file')
    parser.add_argument('--sample', type=int, default=10,
                        help='Number of random entries to display')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()
    validate(args.csv_path, sample_size=args.sample, seed=args.seed)
