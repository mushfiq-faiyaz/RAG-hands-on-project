import re
from datetime import datetime

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

PATTERN = re.compile(
    r"last\s+updated\s*:?\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
    re.IGNORECASE,
)


def extract_date(text):
    m = PATTERN.search(text)
    if not m:
        return None
    day, month_name, year = m.groups()
    month = MONTHS.get(month_name.lower())
    if not month:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


# if __name__ == "__main__":
#     tests = [
#         "Last updated: 1 October 2026",
#         "Last Updated: 5 October 2026",
#         "last updated:20 May 2026",
#         "no date here",
#     ]
#     for t in tests:
#         print(repr(t), "->", extract_date(t))