"""Evaluate a trained SoMe classifier on the test set.

Usage:
    python src/evaluation/evaluate.py setfit models/setfit_round2/
    python src/evaluation/evaluate.py bert   models/bert_final/

Reads:  data/evaluation/test_kappa_250.csv
Writes: results/metrics_{model_type}.json
        results/confusion_matrix_{model_type}.png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES = ['VST', 'HST', 'SA', 'OR', 'WB']
TEST_PATH = os.path.join('data', 'evaluation', 'test_kappa_250.csv')
RESULTS_DIR = 'results'


def load_test_data() -> pd.DataFrame:
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test set not found: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    df[CATEGORIES] = df[CATEGORIES].fillna(0).astype(int)
    return df


# ── Prediction functions ──────────────────────────────────────────────────────

def predict_setfit(model_path: str, texts: list) -> np.ndarray:
    from setfit import SetFitModel
    print(f"Loading SetFit model from {model_path}...")
    model = SetFitModel.from_pretrained(model_path)
    preds = model.predict(texts, as_numpy=True)
    if preds.ndim == 1:
        preds = preds.reshape(-1, len(CATEGORIES))
    return preds.astype(int)


def predict_setfit_proba(model_path: str, texts: list) -> np.ndarray:
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(model_path)
    probs = model.predict_proba(texts, as_numpy=True)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(CATEGORIES))
    return probs


def predict_bert(model_path: str, texts: list) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset, DataLoader

    print(f"Loading BERT model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    model = model.to(device)
    model.eval()

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = tokenizer(
                texts, truncation=True, padding='max_length',
                max_length=512, return_tensors='pt'
            )
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (torch.sigmoid(outputs.logits) > 0.5).int()
            all_preds.append(preds.cpu().numpy())

    return np.vstack(all_preds)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        precision_recall_fscore_support, f1_score,
        accuracy_score, hamming_loss,
    )

    metrics = {}
    for i, cat in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        support = int(y_true[:, i].sum())   # sklearn returns None for support when average!= None
        metrics[cat] = {
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1': round(float(f), 4),
            'support': support,
        }

    metrics['macro_f1'] = round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4)
    metrics['micro_f1'] = round(float(f1_score(y_true, y_pred, average='micro', zero_division=0)), 4)
    metrics['weighted_f1'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
    metrics['hamming_loss'] = round(float(hamming_loss(y_true, y_pred)), 4)
    # Exact match ratio
    metrics['exact_match'] = round(float((y_true == y_pred).all(axis=1).mean()), 4)
    return metrics


def print_metrics(metrics: dict):
    print(f"\n  {'Category':<8} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*46}")
    for cat in CATEGORIES:
        m = metrics[cat]
        print(f"  {cat:<8} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")
    print(f"  {'-'*46}")
    print(f"  {'Macro F1':<8} {metrics['macro_f1']:>10.3f}")
    print(f"  {'Micro F1':<8} {metrics['micro_f1']:>10.3f}")
    print(f"  {'Weighted F1':<8} {metrics['weighted_f1']:>10.3f}")
    print(f"  {'Hamming':<8} {metrics['hamming_loss']:>10.4f}")
    print(f"  {'ExactMatch':<8} {metrics['exact_match']:>10.3f}")


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                             model_type: str, output_dir: str):
    """Plot one binary confusion matrix per category."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("  [WARN] matplotlib/seaborn not available, skipping plots.")
        return

    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(18, 3.5))
    fig.suptitle(f'Per-Category Confusion Matrices — {model_type.upper()}', fontsize=13)

    for i, (cat, ax) in enumerate(zip(CATEGORIES, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'],
            cbar=False,
        )
        ax.set_title(cat)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(model_type: str, model_path: str):
    print(f"\n=== Evaluation: {model_type.upper()} — {model_path} ===")

    df_test = load_test_data()
    print(f"Test set: {len(df_test)} entries")
    print(f"Label distribution:")
    for cat in CATEGORIES:
        n = df_test[cat].sum()
        print(f"  {cat}: {n} ({n/len(df_test)*100:.1f}%)")

    texts = df_test['content'].tolist()
    y_true = df_test[CATEGORIES].values

    if model_type == 'setfit':
        y_pred = predict_setfit(model_path, texts)
    elif model_type == 'bert':
        y_pred = predict_bert(model_path, texts)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'setfit' or 'bert'.")

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, f'metrics_{model_type}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    plot_confusion_matrices(y_true, y_pred, model_type, RESULTS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SoMe classifier')
    parser.add_argument('model_type', choices=['setfit', 'bert'],
                        help='Model type to evaluate')
    parser.add_argument('model_path', help='Path to trained model directory')
    args = parser.parse_args()
    evaluate(args.model_type, args.model_path)
