"""Fine-tune BERT-base for multi-label SoMe classification.

Loads all available training data (seed + all round files), fine-tunes
bert-base-uncased with a 5-output sigmoid head and BCEWithLogitsLoss,
with early stopping on validation loss.

Usage:
    python src/training/train_bert.py

Output:
    models/bert_final/
"""

import os
import sys
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
# Disable MPS — triggers OOM on M-series Macs with the BERT forward pass size.
torch.backends.mps.is_available = lambda: False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CATEGORIES = ['VST', 'HST', 'SA', 'OR', 'WB']
BASE_MODEL = 'bert-base-uncased'
OUTPUT_DIR = os.path.join('models', 'bert_final')
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 10
PATIENCE = 3       # early stopping on val loss
LR = 2e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.2


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_all_training_data() -> pd.DataFrame:
    """Load seed + all available round files."""
    frames = []

    seed_path = os.path.join('data', 'labeled', 'seed_some_items.csv')
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed data not found: {seed_path}")
    df_seed = pd.read_csv(seed_path)
    df_seed[CATEGORIES] = df_seed[CATEGORIES].fillna(0).astype(int)
    frames.append(df_seed)
    print(f"  Seed: {len(df_seed)} entries")

    round_files = sorted(glob.glob(os.path.join('data', 'labeled', 'round*_labeled.csv')))
    for path in round_files:
        df_r = pd.read_csv(path)
        df_r[CATEGORIES] = df_r[CATEGORIES].fillna(0).astype(int)
        frames.append(df_r)
        print(f"  {os.path.basename(path)}: {len(df_r)} entries")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset='content', keep='last').reset_index(drop=True)
    return df


class SoMeDataset:
    """PyTorch Dataset for multi-label SoMe classification."""

    def __init__(self, texts, labels, tokenizer):
        import torch
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_model(df_train, df_val, tokenizer, model):
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    print(f"  Using device: {device}")
    model = model.to(device)

    train_dataset = SoMeDataset(
        df_train['content'].tolist(),
        df_train[CATEGORIES].values.tolist(),
        tokenizer,
    )
    val_dataset = SoMeDataset(
        df_val['content'].tolist(),
        df_val[CATEGORIES].values.tolist(),
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                val_loss += loss_fn(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_preds.append(preds.cpu().numpy())
                all_true.append(labels.int().cpu().numpy())

        val_loss /= len(val_loader)

        from sklearn.metrics import f1_score
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_true)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'macro_f1': round(macro_f1, 4),
        })
        print(f"  Epoch {epoch:>2}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"macro_f1={macro_f1:.4f}")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_bert(model, tokenizer, test_path: str) -> dict:
    import torch
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    model = model.to(device)
    model.eval()

    df_test = pd.read_csv(test_path)
    df_test[CATEGORIES] = df_test[CATEGORIES].fillna(0).astype(int)

    test_dataset = SoMeDataset(
        df_test['content'].tolist(),
        df_test[CATEGORIES].values.tolist(),
        tokenizer,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (torch.sigmoid(outputs.logits) > 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_true.append(batch['labels'].int().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)

    metrics = {}
    print(f"\n  {'Category':<8} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*40}")
    for i, cat in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        metrics[cat] = {'precision': round(p, 4), 'recall': round(r, 4),
                        'f1': round(f, 4), 'support': int(y_true[:, i].sum())}
        print(f"  {cat:<8} {p:>6.3f} {r:>6.3f} {f:>6.3f} {int(sup):>8}")

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['macro_f1'] = round(macro_f1, 4)
    metrics['micro_f1'] = round(micro_f1, 4)
    print(f"  {'-'*40}")
    print(f"  {'Macro F1':<8} {macro_f1:>6.3f}")
    print(f"  {'Micro F1':<8} {micro_f1:>6.3f}")
    return metrics


def train(output_dir: str = OUTPUT_DIR):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split

    set_seed(SEED)

    print("\n=== BERT Multi-label Training ===")
    print("Loading training data...")
    df = load_all_training_data()
    print(f"  Total: {len(df)} entries")
    print(f"  Label distribution:")
    for cat in CATEGORIES:
        n = df[cat].sum()
        print(f"    {cat}: {n} ({n/len(df)*100:.1f}%)")

    # ── Train/val split ────────────────────────────────────────────────────
    # Stratify on the most common single label (OR sum, pick dominant category)
    stratify_col = df[CATEGORIES].idxmax(axis=1)
    try:
        df_train, df_val = train_test_split(
            df, test_size=VAL_SPLIT, random_state=SEED, stratify=stratify_col
        )
    except ValueError:
        # Fallback: no stratification if some classes too small
        df_train, df_val = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)

    print(f"\nTrain: {len(df_train)}  Val: {len(df_val)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        problem_type='multi_label_classification',
        id2label={i: cat for i, cat in enumerate(CATEGORIES)},
        label2id={cat: i for i, cat in enumerate(CATEGORIES)},
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nTraining...")
    model, history = train_model(df_train, df_val, tokenizer, model)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        'base_model': BASE_MODEL,
        'n_train': len(df_train),
        'n_val': len(df_val),
        'label_counts': {cat: int(df[cat].sum()) for cat in CATEGORIES},
        'seed': SEED,
        'max_epochs': NUM_EPOCHS,
        'early_stopping_patience': PATIENCE,
        'history': history,
    }
    with open(os.path.join(output_dir, 'training_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {output_dir}/")

    # ── Evaluate ───────────────────────────────────────────────────────────
    test_path = os.path.join('data', 'evaluation', 'test_kappa_250.csv')
    if os.path.exists(test_path):
        print(f"\nEvaluating on test set: {test_path}")
        metrics = evaluate_bert(model, tokenizer, test_path)
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {output_dir}/test_metrics.json")
    else:
        print(f"\nNo test set found at {test_path}, skipping evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT for multi-label SoMe classification'
    )
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()
    train(args.output)
