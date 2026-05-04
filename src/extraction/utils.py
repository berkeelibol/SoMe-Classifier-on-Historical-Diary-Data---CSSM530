"""Shared utilities for diary text extraction and segmentation."""

import re
import pandas as pd
from typing import List, Tuple, Optional


def clean_editorial_brackets(text: str) -> str:
    """Remove editorial notes enclosed in square brackets, including multi-line ones."""
    # Remove indented bracket blocks (Pepys style: lines starting with spaces + [)
    text = re.sub(r'\n\s{2,}\[.*?\]', '', text, flags=re.DOTALL)
    # Remove inline [Ed. note:...] style
    text = re.sub(r'\[Ed\.?\s*note:.*?\]', '', text, flags=re.DOTALL)
    return text


def clean_footnotes(text: str) -> str:
    """Remove footnote blocks like [Footnote N: ...] (Wordsworth style)."""
    text = re.sub(r'\[Footnote\s+\d+:.*?\]', '', text, flags=re.DOTALL)
    # Also remove bare footnote reference numbers like superscript digits
    text = re.sub(r'(?<=[a-z.,;])\d{1,2}(?=\s)', '', text)
    return text


def collapse_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, fix line breaks."""
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines to 2
    text = re.sub(r' +\n', '\n', text)  # Trailing spaces before newline
    return text.strip()


def fix_ocr_artifacts(text: str) -> str:
    """Fix common OCR issues: soft hyphens, broken words, comma-splits."""
    text = text.replace('\u00ad', '')  # Soft hyphen
    text = text.replace('¬\n', '')  # Broken words across lines with ¬
    text = text.replace('¬ \n', '')
    text = re.sub(r'(\w),(\w)', r'\1\2', text)  # Fix comma-split words like "Thoma,s"
    # But be careful not to break real commas - only fix single letter splits
    return text


def segment_entry(text: str, min_words: int = 30, max_words: int = 300) -> List[str]:
    """
    Segment a diary entry into chunks respecting sentence boundaries.

    If the entry is within the word limits, return it as-is.
    If too long, split at sentence boundaries trying to stay within limits.
    If too short, discard.
    """
    text = text.strip()
    words = text.split()

    if len(words) < min_words:
        return []

    if len(words) <= max_words:
        return [text]

    # Split into sentences (rough but effective)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_word_count + sentence_words > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= min_words:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    # Handle remaining
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= min_words:
            chunks.append(chunk_text)
        elif chunks:
            # Append to last chunk if too short on its own
            chunks[-1] = chunks[-1] + ' ' + chunk_text

    return chunks


def entries_to_dataframe(entries: List[dict], author: str) -> pd.DataFrame:
    """
    Convert list of entry dicts to a segmented DataFrame.

    Each entry dict should have: year, date_mm_dd, content
    Entries are segmented and word counts are added.
    """
    rows = []
    for entry in entries:
        segments = segment_entry(entry['content'])
        for seg in segments:
            rows.append({
                'author': author,
                'year': entry.get('year', ''),
                'date_mm_dd': entry.get('date_mm_dd', ''),
                'content': seg,
                'word_count': len(seg.split()),
            })

    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, author: str):
    """Save processed DataFrame to data/processed/."""
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{author}.csv')
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} entries to {path}")
    if len(df) > 0:
        print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"  Word count: mean={df['word_count'].mean():.0f}, median={df['word_count'].median():.0f}")
