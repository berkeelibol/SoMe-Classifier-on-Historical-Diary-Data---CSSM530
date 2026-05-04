"""Extract and segment diary entries from Samuel Pepys' raw text."""

import re
import os
import sys
import argparse
import pandas

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.extraction.utils import (
    clean_editorial_brackets,
    collapse_whitespace,
    entries_to_dataframe,
    save_processed,
)

# Month abbreviations used in Pepys' diary
MONTH_ABBREV = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'June': 6, 'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12,
}

# Full month names for header parsing
MONTH_FULL = {
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
    'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
    'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
}

# Ordinal suffixes
ORDINAL_RE = r'(\d{1,2})(?:st|nd|rd|th)'


def parse_header_year(year_str: str, month_num: int) -> int:
    """
    Parse year from header, handling Old Style dating.

    'JANUARY 1659-1660' -> 1660 (dual year, take second)
    'APRIL 1660' -> 1660 (single year)

    For dual years like 1659-1660, the second year is the modern (New Style) year.
    """
    year_str = year_str.strip()
    if '-' in year_str:
        parts = year_str.split('-')
        # Handle '1659-1660' or '1659-60'
        first = int(parts[0])
        second_str = parts[1]
        if len(second_str) <= 2:
            # '1659-60' -> 1660
            second = int(str(first)[:2] + second_str)
        else:
            second = int(second_str)
        return second
    return int(year_str)


def extract_pepys(input_path: str):
    """Extract diary entries from Pepys raw text."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove editorial bracket blocks
    text = clean_editorial_brackets(text)
    text = collapse_whitespace(text)

    entries = []
    current_year = None
    current_month = None

    lines = text.split('\n')
    i = 0

    # Pattern for month/year headers: 'JANUARY 1659-1660' or 'APRIL 1660'
    header_re = re.compile(
        r'^(' + '|'.join(MONTH_FULL.keys()) + r')\s+(\d{4}(?:-\d{2,4})?)\s*\.?$'
    )

    # Pattern for entry start with month abbreviation: 'Jan. 1st' or 'Jan. 1st (Lord's day).'
    entry_with_month_re = re.compile(
        r'^(' + '|'.join(k for k in MONTH_ABBREV if len(k) == 3) + r')\.?\s+' + ORDINAL_RE
    )

    # Pattern for entry start with just ordinal: '2nd.' or '15th.'
    entry_ordinal_re = re.compile(r'^' + ORDINAL_RE + r'\.')

    current_entry_lines = []
    current_entry_day = None
    current_entry_month = None
    current_entry_year = None

    def flush_entry():
        """Save the current accumulated entry."""
        if current_entry_lines and current_entry_day is not None:
            content = ' '.join(current_entry_lines).strip()
            if content:
                mm = f"{current_entry_month:02d}" if current_entry_month else "00"
                dd = f"{current_entry_day:02d}"
                entries.append({
                    'year': current_entry_year or '',
                    'date_mm_dd': f"{mm}-{dd}",
                    'content': content,
                })

    while i < len(lines):
        line = lines[i].strip()

        # Check for month/year header
        header_match = header_re.match(line)
        if header_match:
            flush_entry()
            current_entry_lines = []
            current_entry_day = None

            month_name = header_match.group(1)
            year_str = header_match.group(2)
            current_month = MONTH_FULL[month_name]
            current_year = parse_header_year(year_str, current_month)
            i += 1
            continue

        # Check for entry start with month abbreviation
        entry_month_match = entry_with_month_re.match(line)
        if entry_month_match:
            flush_entry()
            month_abbrev = entry_month_match.group(1)
            day = int(entry_month_match.group(2))
            current_entry_month = MONTH_ABBREV.get(month_abbrev, current_month)
            current_entry_day = day
            current_entry_year = current_year
            # Remove the date prefix from the line content
            rest = line[entry_month_match.end():].strip()
            # Strip trailing day context like '(Lord's day).' at the very start
            rest = re.sub(r'^\(.*?\)\.?\s*', '', rest).strip()
            # Also strip leading punctuation/dashes
            rest = rest.lstrip('—-–. ')
            current_entry_lines = [rest] if rest else []
            i += 1
            continue

        # Check for entry start with just ordinal
        entry_ord_match = entry_ordinal_re.match(line)
        if entry_ord_match:
            flush_entry()
            day = int(entry_ord_match.group(1))
            current_entry_day = day
            current_entry_month = current_month
            current_entry_year = current_year
            rest = line[entry_ord_match.end():].strip()
            rest = re.sub(r'^\(.*?\)\.?\s*', '', rest).strip()
            rest = rest.lstrip('—-–. ')
            current_entry_lines = [rest] if rest else []
            i += 1
            continue

        # Regular line — append to current entry
        if line and current_entry_day is not None:
            current_entry_lines.append(line)

        i += 1

    # Flush last entry
    flush_entry()

    print(f"Extracted {len(entries)} raw entries")

    # Convert to dataframe with segmentation
    df = entries_to_dataframe(entries, 'pepys')
    save_processed(df, 'pepys')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Pepys diary entries')
    parser.add_argument('--input', default='data/raw/pepys_raw.txt',
                        help='Path to raw Pepys text file')
    args = parser.parse_args()
    extract_pepys(args.input)
