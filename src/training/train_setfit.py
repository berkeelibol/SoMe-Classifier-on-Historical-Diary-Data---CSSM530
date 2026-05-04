"""Train a multi-label SetFit model for SoMe category classification.

Usage:
    python src/training/train_setfit.py 0          # seed data only
    python src/training/train_setfit.py 2          # seed + round1 + round2

Training data loaded:
    Round 0:  data/labeled/seed_some_items.csv
    Round N:  seed + data/labeled/round{1..N}_labeled.csv

Model saved to: models/setfit_round{N}/
"""

import os
import sys
import glob
import json
import argparse
import random
import numpy as np
import pandas as pd

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES = ['VST', 'HST', 'SA', 'OR', 'WB']
BASE_MODEL = 'sentence-transformers/all-mpnet-base-v2'
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_training_data(round_num: int) -> pd.DataFrame:
    """Load seed data plus all labeled rounds up to round_num."""
    frames = []

    seed_path = os.path.join('data', 'labeled', 'seed_some_items.csv')
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed data not found: {seed_path}")
    df_seed = pd.read_csv(seed_path)
    df_seed[CATEGORIES] = df_seed[CATEGORIES].fillna(0).astype(int)
    frames.append(df_seed)
    print(f"  Seed data:  {len(df_seed)} entries")

    for i in range(1, round_num + 1):
        path = os.path.join('data', 'labeled', f'round{i}_labeled.csv')
        if os.path.exists(path):
            df_r = pd.read_csv(path)
            df_r[CATEGORIES] = df_r[CATEGORIES].fillna(0).astype(int)
            frames.append(df_r)
            print(f"  Round {i}:    {len(df_r)} entries ({path})")
        else:
            print(f"  Round {i}:    NOT FOUND — {path} (skipping)")

    df = pd.concat(frames, ignore_index=True)
    # Drop duplicates by content, keeping last (most recent labels win)
    df = df.drop_duplicates(subset='content', keep='last').reset_index(drop=True)
    return df


def build_dataset(df: pd.DataFrame):
    """Convert DataFrame to HuggingFace Dataset with multi-label format."""
    from datasets import Dataset

    records = {
        'text': df['content'].tolist(),
        'label': df[CATEGORIES].values.tolist(),   # list of [VST, HST, SA, OR, WB]
    }
    return Dataset.from_dict(records)


def evaluate_on_test(model, test_path: str) -> dict:
    """Evaluate model on test set and return per-category metrics."""
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    df_test = pd.read_csv(test_path)
    df_test[CATEGORIES] = df_test[CATEGORIES].fillna(0).astype(int)

    texts = df_test['content'].tolist()
    y_true = df_test[CATEGORIES].values  # (n, 5)

    print(f"  Running predictions on {len(texts)} test entries...")
    preds = model.predict(texts, as_numpy=True)  # (n, 5) binary
    if preds.ndim == 1:
        # Fallback: shouldn't happen with one-vs-rest, but guard anyway
        preds = preds.reshape(-1, len(CATEGORIES))

    metrics = {}
    print(f"\n  {'Category':<8} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*40}")
    for i, cat in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], preds[:, i], average='binary', zero_division=0
        )
        metrics[cat] = {'precision': round(p, 4), 'recall': round(r, 4),
                        'f1': round(f, 4), 'support': int(y_true[:, i].sum())}
        print(f"  {cat:<8} {p:>6.3f} {r:>6.3f} {f:>6.3f} {int(sup):>8}")

    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    metrics['macro_f1'] = round(macro_f1, 4)
    metrics['micro_f1'] = round(micro_f1, 4)
    print(f"  {'-'*40}")
    print(f"  {'Macro F1':<8} {macro_f1:>6.3f}")
    print(f"  {'Micro F1':<8} {micro_f1:>6.3f}")

    return metrics


def train(round_num: int):
    from setfit import SetFitModel, Trainer, TrainingArguments

    set_seed(SEED)

    print(f"\n=== SetFit Training — Round {round_num} ===")
    print("Loading training data...")
    df = load_training_data(round_num)
    print(f"  Total training entries: {len(df)}")
    print(f"  Label distribution:")
    for cat in CATEGORIES:
        n = df[cat].sum()
        print(f"    {cat}: {n} positive ({n/len(df)*100:.1f}%)")

    train_dataset = build_dataset(df)

    print(f"\nLoading base model: {BASE_MODEL}")
    model = SetFitModel.from_pretrained(
        BASE_MODEL,
        multi_target_strategy='one-vs-rest',
        labels=CATEGORIES,
        device='cpu',
    )

    output_dir = os.path.join('models', f'setfit_round{round_num}')
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_epochs=(1, 16),       # (contrastive epochs, classifier epochs)
        batch_size=(8, 2),        # smaller batch = half the MPS memory per step
        max_steps=-1,             # run all natural steps
        seed=SEED,
        report_to='none',
        sampling_strategy='oversampling',
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        column_mapping={'text': 'text', 'label': 'label'},
    )

    print("\nTraining...")
    trainer.train()

    print(f"\nSaving model to {output_dir}/")
    model.save_pretrained(output_dir)

    # Save round metadata
    meta = {
        'round': round_num,
        'base_model': BASE_MODEL,
        'n_train': len(df),
        'label_counts': {cat: int(df[cat].sum()) for cat in CATEGORIES},
        'seed': SEED,
    }
    with open(os.path.join(output_dir, 'training_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Evaluate on test set if available
    test_path = os.path.join('data', 'evaluation', 'test_kappa_250.csv')
    if os.path.exists(test_path):
        print(f"\nEvaluating on test set: {test_path}")
        metrics = evaluate_on_test(model, test_path)
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {output_dir}/test_metrics.json")
    else:
        print(f"\nNo test set found at {test_path}, skipping evaluation.")

    print(f"\nDone. Model saved to {output_dir}/")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SetFit SoMe multi-label classifier')
    parser.add_argument('round', type=int, help='Training round number (0 = seed only)')
    args = parser.parse_args()
    train(args.round)
