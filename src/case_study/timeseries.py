"""Time-series analysis of SoMe category distributions over time."""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_category_proportions(df: pd.DataFrame, time_col: str = 'year',
                                  label_col: str = 'predicted_label') -> pd.DataFrame:
    """Compute SoMe category proportions per time period."""
    grouped = df.groupby(time_col)[label_col].value_counts(normalize=True).unstack(fill_value=0)
    return grouped


def detect_changepoints(series: pd.Series, n_bkps: int = 3):
    """Detect changepoints in a time series using ruptures."""
    import ruptures as rpt

    signal = series.values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=1)
    return result


def plot_timeseries(df: pd.DataFrame, label_col: str = 'predicted_label',
                    output_path: str = 'results/timeseries.png'):
    """Plot SoMe category proportions over time."""
    props = compute_category_proportions(df, label_col=label_col)

    fig, ax = plt.subplots(figsize=(14, 6))
    props.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion')
    ax.set_title('Sources of Meaning Over Time')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def run_trend_tests(df: pd.DataFrame, label_col: str = 'predicted_label'):
    """Run Mann-Kendall trend tests for each category."""
    props = compute_category_proportions(df, label_col=label_col)

    print("Trend analysis (Spearman correlation with time):")
    for col in props.columns:
        r, p = stats.spearmanr(range(len(props)), props[col])
        sig = "*" if p < 0.05 else ""
        print(f"  {col}: r={r:.3f}, p={p:.3f} {sig}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-series SoMe analysis')
    parser.add_argument('--input', required=True, help='Path to classified CSV')
    parser.add_argument('--output', default='results/timeseries.png')
    parser.add_argument('--label-col', default='predicted_label')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_timeseries(df, args.label_col, args.output)
    run_trend_tests(df, args.label_col)
