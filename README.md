# SoMe-Classifier-on-Historical-Diary-Data---CSSM530
This project aims to train a Sources of Meaning classification model using the SoMe scale in the literature and historical diary data of real people, as well as presenting a applied case study(upcoming).

## Project Structure

```
some-diary-nlp/
├── data/
│   ├── raw/              # Raw downloaded texts (place here manually)
│   ├── processed/        # Clean segmented CSVs per author
│   ├── labeled/          # Training data (seed + active learning rounds)
│   └── evaluation/       # Gold standard test set
├── src/
│   ├── extraction/       # Per-author extraction scripts + shared utils
│   ├── training/         # Model training + active learning
│   ├── evaluation/       # Metrics computation
│   └── case_study/       # Time-series analysis on Thoreau
├── results/              # Tables, figures, metrics
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Raw Data

Place raw text files in `data/raw/` before running extraction:

| Author | File | Source |
|--------|------|--------|
| Samuel Pepys | `pepys_raw.txt` | [Project Gutenberg](https://www.gutenberg.org/ebooks/4200) |
| John Wesley | `wesley_raw.txt` | [Internet Archive](https://archive.org/details/journalrevjohnwe01wesluoft) |
| Dorothy Wordsworth | `wordsworth_raw.txt` | [Project Gutenberg](https://www.gutenberg.org/ebooks/42856) |

## Extraction

Run from the project root:

```bash
python src/extraction/extract_pepys.py
python src/extraction/extract_wesley.py
python src/extraction/extract_wordsworth.py
```

Validate processed output:

```bash
python src/extraction/validate.py data/processed/pepys.csv
```

Output CSVs are saved to `data/processed/` with columns: `author, year, date_mm_dd, content, word_count`.

---

## Active Learning Workflow

Each round follows the following

**1. Select uncertain entries for labeling:**
```bash
python src/training/active_learning.py <round> <author>
# e.g. python src/training/active_learning.py 1 pepys
```
Writes `data/labeled/round{N}_to_label.csv` with blank label columns.

**2. Manually annotate** — open `round{N}_to_label.csv`, fill in `VST/HST/SA/OR/WB` (0 or 1 per column), save as `data/labeled/round{N}_labeled.csv`.

**3. Retrain:**
```bash
python src/training/train_setfit.py <round>
# e.g. python src/training/train_setfit.py 1
```

**4. Evaluate:**
```bash
python src/evaluation/evaluate.py setfit models/setfit_round1/
```

---

### Labeled data files

| File | Description |
|------|-------------|
| `data/labeled/seed_some_items.csv` | 142 SoMe scale items, labeled |
| `data/labeled/round{N}_to_label.csv` | 50 entries selected by active learning (blank labels) |
| `data/labeled/round{N}_labeled.csv` | Same 50 entries after manual annotation |
| `data/evaluation/test_kappa_250.csv` | Gold standard test set (held out) |
