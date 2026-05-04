"""Extract and segment diary entries from John Wesley's Journal raw text.

Wesley's text is OCR'd from archive.org and requires aggressive cleaning:
- Double/triple spaces, soft hyphens, broken words
- Volume headers, editorial prefaces, footnotes
- Entry dates: 'Mon. 10.—', 'Fri. 7.—', 'Sun. 9.—'
- Year/month markers: '1763.1 Jan. r, Sat.—' or '[Jan. 1763.'
- Section headers: 'From January i, 1763, to May 25, 1765'
"""

import re
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.extraction.utils import (
    fix_ocr_artifacts,
    collapse_whitespace,
    entries_to_dataframe,
    save_processed,
)

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12,
}

DAY_ABBREVS = ['Mon', 'Tue', 'Tues', 'Wed', 'Thu', 'Thur', 'Thurs',
               'Fri', 'Sat', 'Sun']


def clean_wesley_text(text: str) -> str:
    """Apply Wesley-specific cleaning steps."""
    # Fix OCR artifacts first
    text = fix_ocr_artifacts(text)

    # Collapse multiple spaces (very common in Wesley OCR)
    text = collapse_whitespace(text)

    # Remove volume headers like 'VOL. V' or 'VOL. IV.'
    text = re.sub(r'^VOL\.\s+[IVXLC]+\.?\s*$', '', text, flags=re.MULTILINE)

    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'^\d{1,4}\s*$', '', text, flags=re.MULTILINE)

    # Remove footnote blocks: lines starting with a number followed by text
    # These typically appear as '1 See Journal, vol. iv.' or similar
    # Be conservative — only remove if it looks like a footnote reference
    text = re.sub(r'^\d{1,2}\s+(?:See|Cf\.|Compare|Ed\.).*$', '', text, flags=re.MULTILINE)

    return text


def remove_editorial_prefaces(text: str) -> str:
    """Remove editorial preface sections that appear before 'THE JOURNAL'."""
    # Find 'THE JOURNAL' markers and remove everything before them within sections
    parts = re.split(r'THE\s+JOURNAL', text)
    if len(parts) > 1:
        # Keep content after each 'THE JOURNAL' marker
        text = 'THE JOURNAL'.join(parts)
    return text


def parse_section_header(text: str):
    """
    Parse section headers like 'From January i, 1763, to May 25, 1765'
    to extract the starting year.
    Returns (start_year, start_month) or (None, None).
    """
    # Fix OCR: 'i' often appears instead of '1'
    pattern = r'From\s+(\w+)\s+[\di]+,?\s+(\d{4})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        month_str = match.group(1).lower()
        year = int(match.group(2))
        month = MONTH_MAP.get(month_str, None)
        return year, month
    return None, None


def extract_wesley(input_path: str):
    """Extract diary entries from Wesley's Journal raw text."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_wesley_text(text)
    text = remove_editorial_prefaces(text)

    entries = []
    current_year = None
    current_month = None

    lines = text.split('\n')

    # Build day abbreviation pattern
    day_abbrev_pattern = '|'.join(DAY_ABBREVS)

    # Pattern for entry dates: 'Mon. 10.—' or 'Fri. 7. —' or 'Sun. 9.—'
    entry_re = re.compile(
        r'^(' + day_abbrev_pattern + r')\.?\s+(\d{1,2})\.?\s*[—\-–]'
    )

    # Pattern for year/month context markers:
    # '1763.1 Jan. r, Sat.—' or just the year at start
    year_month_re = re.compile(
        r'(\d{4})\.\s*(?:\d+\s+)?(\w+)\.?'
    )

    # Pattern for bracketed month/year: '[Jan. 1763.'
    bracket_month_re = re.compile(
        r'\[(\w+)\.?\s+(\d{4})'
    )

    # Section header pattern
    section_re = re.compile(
        r'From\s+(\w+)\s+[\di]+,?\s+(\d{4})',
        re.IGNORECASE
    )

    current_entry_lines = []
    current_entry_day = None

    def flush_entry():
        """Save the current accumulated entry."""
        if current_entry_lines and current_entry_day is not None and current_year is not None:
            content = ' '.join(current_entry_lines).strip()
            # Skip suspiciously short or garbled entries
            if len(content.split()) < 5:
                return
            # Skip entries that look like editorial content
            if content.startswith('See ') or content.startswith('Cf.'):
                return
            mm = f"{current_month:02d}" if current_month else "00"
            dd = f"{current_entry_day:02d}"
            entries.append({
                'year': current_year,
                'date_mm_dd': f"{mm}-{dd}",
                'content': content,
            })

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check for section headers to establish year context
        section_match = section_re.search(line)
        if section_match:
            flush_entry()
            current_entry_lines = []
            current_entry_day = None
            month_str = section_match.group(1).lower()
            year = int(section_match.group(2))
            current_year = year
            current_month = MONTH_MAP.get(month_str, current_month)
            i += 1
            continue

        # Check for bracketed month/year markers: [Jan. 1763.
        bracket_match = bracket_month_re.search(line)
        if bracket_match:
            month_str = bracket_match.group(1).lower().rstrip('.')
            year = int(bracket_match.group(2))
            current_year = year
            if month_str in MONTH_MAP:
                current_month = MONTH_MAP[month_str]

        # Check for year/month inline markers: '1763.1 Jan.'
        year_month_match = year_month_re.match(line)
        if year_month_match:
            year = int(year_month_match.group(1))
            month_str = year_month_match.group(2).lower().rstrip('.')
            if 1600 <= year <= 1900 and month_str in MONTH_MAP:
                current_year = year
                current_month = MONTH_MAP[month_str]

        # Check for entry start: 'Mon. 10.—'
        entry_match = entry_re.match(line)
        if entry_match:
            flush_entry()
            day = int(entry_match.group(2))
            current_entry_day = day

            # Handle month rollover: if day is less than previous day,
            # we may have moved to the next month
            if entries and current_month:
                prev = entries[-1]
                prev_dd = int(prev['date_mm_dd'].split('-')[1])
                if day < prev_dd and (prev_dd - day) > 15:
                    # Likely month rollover
                    current_month = min(current_month + 1, 12)
                    if current_month == 1:
                        current_year = (current_year or 0) + 1

            # Extract content after the date marker
            rest = line[entry_match.end():].strip()
            rest = rest.lstrip('—-–. ')
            current_entry_lines = [rest] if rest else []
            i += 1
            continue

        # Regular line — append to current entry if we're in one
        if current_entry_day is not None:
            # Skip lines that are clearly not journal content
            if not re.match(r'^(VOL\.|JOURNAL|THE JOURNAL|CHAPTER)', line, re.IGNORECASE):
                current_entry_lines.append(line)

        i += 1

    # Flush last entry
    flush_entry()

    print(f"Extracted {len(entries)} raw entries")

    # Convert to dataframe with segmentation
    df = entries_to_dataframe(entries, 'wesley')
    save_processed(df, 'wesley')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Wesley journal entries')
    parser.add_argument('--input', default='data/raw/wesley_raw.txt',
                        help='Path to raw Wesley text file')
    args = parser.parse_args()
    extract_wesley(args.input)
