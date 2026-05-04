"""Sentiment analysis for diary entries using VADER."""

import sys
import argparse
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_sentiment(input_path: str, output_path: str):
    """Add VADER sentiment scores to diary entries."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    analyzer = SentimentIntensityAnalyzer()

    scores = []
    for text in df['content']:
        vs = analyzer.polarity_scores(str(text))
        scores.append(vs)

    df['sentiment_compound'] = [s['compound'] for s in scores]
    df['sentiment_pos'] = [s['pos'] for s in scores]
    df['sentiment_neg'] = [s['neg'] for s in scores]
    df['sentiment_neu'] = [s['neu'] for s in scores]

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} entries with sentiment to {output_path}")

    print(f"\nSentiment summary:")
    print(f"  Mean compound: {df['sentiment_compound'].mean():.3f}")
    print(f"  Positive entries: {(df['sentiment_compound'] > 0.05).sum()}")
    print(f"  Negative entries: {(df['sentiment_compound'] < -0.05).sum()}")
    print(f"  Neutral entries: {((df['sentiment_compound'] >= -0.05) & (df['sentiment_compound'] <= 0.05)).sum()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add sentiment scores')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', required=True, help='Output path')
    args = parser.parse_args()
    add_sentiment(args.input, args.output)
