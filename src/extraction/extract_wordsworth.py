"""Extract and segment diary entries from Dorothy Wordsworth's Journals raw text.

Wordsworth's text is the cleanest but smallest. Multiple journal sections:
- Alfoxden journal (1798)
- Hamburg journal (1798)
- Grasmere journal (1800-1803, in multiple volume sections)

Date formats vary:
- '_21st._ text' or '_22nd._--text' (short, day only)
- '_February 1st._--text' (month + day)
- '_May 14th, 1800._--text' (month + day + year)
- '_Saturday, 4th October 1800._--text' (weekday + day + month + year)
- '_Saturday._--text' (weekday only, no number)
- '_Monday, October 2nd._--text' (weekday + month + day)
"""

import re
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.extraction.utils import (
    clean_footnotes,
    collapse_whitespace,
    entries_to_dataframe,
    save_processed,
)

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

WEEKDAYS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}

# Matches the underscore-wrapped date block at the start of a line:
# Everything from the opening _ to the closing _.  followed by -- or space
# Examples: _21st._, _February 1st._, _Saturday, 4th October 1800._
DATE_BLOCK_RE = re.compile(r'^_([^_]+)_\.?\s*(?:--|-|—|–)?\s*')


def parse_date_block(block: str):
    """
    Parse the content inside the underscore date block.
    Returns (year, month, day) where any can be None.

    Handles:
    - '21st.' -> (None, None, 21)
    - 'February 1st.' -> (None, 2, 1)
    - 'May 14th, 1800.' -> (1800, 5, 14)
    - 'Saturday, 4th October 1800.' -> (1800, 10, 4)
    - 'Monday, October 2nd.' -> (None, 10, 2)
    - 'Saturday.' -> (None, None, None)  weekday-only
    - 'Sunday, 1st October.' -> (None, 10, 1)
    - 'Monday Morning, 8th February 1802.' -> (1802, 2, 8)
    """
    block = block.strip().rstrip('.')

    year = None
    month = None
    day = None

    # Extract year if present (4-digit number)
    year_match = re.search(r'(\d{4})', block)
    if year_match:
        year = int(year_match.group(1))

    # Extract ordinal day number
    day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)', block)
    if day_match:
        day = int(day_match.group(1))

    # Extract month name
    for word in re.findall(r'[A-Za-z]+', block):
        if word.lower() in MONTH_MAP:
            month = MONTH_MAP[word.lower()]
            break

    return year, month, day


def parse_section_header(line: str):
    """
    Parse section headers to extract date context.
    Returns (section_name, year, month) or None.

    Headers look like:
    - 'DOROTHY WORDSWORTH'S JOURNAL, WRITTEN AT ALFOXDEN IN 1798'
    - '(14TH MAY TO 21ST DECEMBER 1800)'
    - 'EXTRACTS FROM DOROTHY WORDSWORTH'S JOURNAL, WRITTEN AT GRASMERE'
    - '(FROM 10TH OCTOBER 1801 TO 29TH DECEMBER 1801)'
    - 'EXTRACTS FROM DOROTHY WORDSWORTH'S JOURNAL OF DAYS SPENT AT HAMBURGH'
    """
    upper = line.strip().upper()

    # Detect section by location name
    section = None
    if 'ALFOXDEN' in upper:
        section = 'alfoxden'
    elif 'HAMBURGH' in upper or 'HAMBURG' in upper:
        section = 'hamburg'
    elif 'GRASMERE' in upper:
        section = 'grasmere'

    if not section and 'DOROTHY WORDSWORTH' not in upper:
        return None

    # Try to extract year from headers like '(14TH MAY TO 21ST DECEMBER 1800)'
    # or '(FROM 10TH OCTOBER 1801 TO ...' or 'IN 1798'
    # Look for "FROM ... MONTH YEAR" pattern first (start date)
    from_match = re.search(r'FROM\s+\d+\w*\s+(\w+)\s+(\d{4})', upper)
    if from_match:
        month_str = from_match.group(1).lower()
        year = int(from_match.group(2))
        month = MONTH_MAP.get(month_str)
        return section, year, month

    # Look for standalone year references like '(14TH MAY TO 21ST DECEMBER 1800)'
    paren_match = re.search(r'\(.*?(\w+)\s+TO\s+\d+\w*\s+\w+\s+(\d{4})\)', upper)
    if paren_match:
        # Get the first month mentioned
        first_month_match = re.search(r'(\d+)\w*\s+(\w+)\s+TO', upper)
        if first_month_match:
            month_str = first_month_match.group(2).lower()
            year = int(paren_match.group(2))
            month = MONTH_MAP.get(month_str)
            return section, year, month

    # Fallback: just find a year
    year_match = re.search(r'(\d{4})', line)
    if year_match:
        year = int(year_match.group(1))
        return section, year, None

    if section:
        return section, None, None

    return None


def extract_wordsworth(input_path: str):
    """Extract diary entries from Wordsworth's Journals raw text."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove footnotes
    text = clean_footnotes(text)
    text = collapse_whitespace(text)

    entries = []
    current_year = None
    current_month = None
    current_section = None
    prev_day = None

    lines = text.split('\n')

    current_entry_lines = []
    current_entry_day = None
    current_entry_month = None
    current_entry_year = None

    def flush_entry():
        nonlocal prev_day
        """Save the current accumulated entry."""
        if current_entry_lines and current_entry_day is not None:
            content = ' '.join(current_entry_lines).strip()
            # Clean up underscores used for emphasis in source
            content = content.replace('_', '')
            if content:
                mm = f"{current_entry_month:02d}" if current_entry_month else "00"
                dd = f"{current_entry_day:02d}"
                entries.append({
                    'year': current_entry_year or '',
                    'date_mm_dd': f"{mm}-{dd}",
                    'content': content,
                })
                prev_day = current_entry_day

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Check for section headers (all-caps lines or lines with DOROTHY WORDSWORTH)
        if ('DOROTHY WORDSWORTH' in line.upper() or
            re.match(r'^\s*\(.*\d{4}\)\s*$', line) or
            'EXTRACTS FROM' in line.upper()):
            result = parse_section_header(line)
            if result:
                flush_entry()
                current_entry_lines = []
                current_entry_day = None
                section, year, month = result
                if section:
                    current_section = section
                if year:
                    current_year = year
                if month:
                    current_month = month
                prev_day = None
                i += 1
                continue

        # Check for date entry: line starting with _..._
        date_match = DATE_BLOCK_RE.match(line)
        if date_match:
            block = date_match.group(1)

            # Skip if this is just emphasis in mid-text (not a date)
            # Date blocks contain either a day number, a month name, or a weekday
            has_day = bool(re.search(r'\d{1,2}(?:st|nd|rd|th)', block))
            has_month = any(w.lower() in MONTH_MAP for w in re.findall(r'[A-Za-z]+', block))
            has_weekday = any(w.lower() in WEEKDAYS for w in re.findall(r'[A-Za-z]+', block))

            if has_day or has_month or has_weekday:
                flush_entry()
                year, month, day = parse_date_block(block)

                # Update tracking context
                if year:
                    current_year = year
                if month:
                    current_month = month

                # Detect month rollover for day-only entries
                if day and not month and prev_day and day < prev_day and (prev_day - day) > 15:
                    if current_month:
                        current_month += 1
                        if current_month > 12:
                            current_month = 1
                            if current_year:
                                current_year += 1

                current_entry_day = day  # Can be None for weekday-only
                current_entry_month = current_month if not month else month
                current_entry_year = current_year if not year else year

                # If weekday-only (no day number), try to infer day
                if not day:
                    # Skip weekday-only entries for now — we can't date them
                    # Actually, still collect the text; use prev_day + 1 as estimate
                    if prev_day:
                        current_entry_day = prev_day + 1
                    else:
                        current_entry_day = None

                rest = line[date_match.end():].strip()
                rest = rest.lstrip('—-–. ')
                current_entry_lines = [rest] if rest else []
                i += 1
                continue

        # Also handle the Alfoxden header which contains the first entry inline:
        # 'Alfoxden, _January 20th 1798_.--The green paths...'
        if 'Alfoxden' in line and '_' in line:
            alfoxden_match = re.search(r'Alfoxden,?\s*_([^_]+)_\.?\s*(?:--|-|—|–)\s*(.*)', line)
            if alfoxden_match:
                flush_entry()
                current_section = 'alfoxden'
                block = alfoxden_match.group(1)
                year, month, day = parse_date_block(block)
                if year:
                    current_year = year
                if month:
                    current_month = month
                current_entry_day = day
                current_entry_month = month or current_month
                current_entry_year = year or current_year
                rest = alfoxden_match.group(2).strip()
                current_entry_lines = [rest] if rest else []
                prev_day = day
                i += 1
                continue

        # Regular line — append to current entry
        if current_entry_day is not None:
            current_entry_lines.append(line)

        i += 1

    # Flush last entry
    flush_entry()

    print(f"Extracted {len(entries)} raw entries")

    # Convert to dataframe with segmentation
    df = entries_to_dataframe(entries, 'wordsworth')
    save_processed(df, 'wordsworth')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Wordsworth journal entries')
    parser.add_argument('--input', default='data/raw/wordsworth_raw.txt',
                        help='Path to raw Wordsworth text file')
    args = parser.parse_args()
    extract_wordsworth(args.input)
