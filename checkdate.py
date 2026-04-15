from dateutil import parser as dateparser
import sys

def parse_date(prompt):
    raw = input(prompt).strip()
    dt = dateparser.parse(raw)
    if dt is None:
        sys.exit(f"Error: cannot parse date '{raw}'. Try YYYY-MM-DD (e.g. 2020-01-01).")
    date_str = dt.strftime('%Y-%m-%d')
    print(f"  -> Parsed as {date_str}")
    return date_str, dt