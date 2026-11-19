"""Make a big messy CSV whose right answers are known, then score Pandalyzer on it.

    python scripts/stress_test.py                # 100,000 rows -> database/messy_orders_100k.csv
    python scripts/stress_test.py --rows 20000   # a smaller file
    python scripts/stress_test.py --no-check     # only write the file

The file is a made-up Indian retail order export with most of the mess the README
lists. The file has a BOM, CRLF line endings, a title above the header and a
trailing delimiter. The header has stray spaces and a line break inside a cell.
The rows include repeated headers, comma-only rows, rows with an extra field,
Total rows, and duplicates (exact, and differing only by spacing). The values
have placeholders, stray and invisible whitespace, case variants, "₹1,234.50",
"INR 1,234.50", Indian "1,23,456.00", European "1.234,56 €", "15%", "(42.00)",
mixed yes/no words, day-first dates in six formats, Excel ="00042" codes,
22-digit tracking numbers, messy emails and multi-line notes. A few cells carry
mess the README says Pandalyzer leaves alone (synonyms, typos, number words).

The generator keeps each cell's right clean value, so the score shows how many
cells of each kind of mess came out right, column by column.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from autoclean import auto_clean, load_csv

ROOT = Path(__file__).resolve().parent.parent

PLACEHOLDERS = ("N/A", "n/a", "null", "NULL", "-", "\u2014", "TBD", "unknown", "(blank)", "#N/A")
# Stray characters around values: spaces, a tab, non-breaking and zero-width spaces.
PADDING = (" ", "  ", "\t", "\u00a0", "\u200b", " \t")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
FIRST_NAMES = (
    "Aarav", "Vivaan", "Aditya", "Arjun", "Ishaan", "Rohan", "Kabir", "Rahul", "Vikram", "Sanjay",
    "Ananya", "Diya", "Saanvi", "Kavya", "Priya", "Meera", "Isha", "Riya", "Neha", "Pooja",
    "Lakshmi", "Deepa", "Farhan", "Zoya", "Imran", "Ayesha", "José", "Zoë", "Chloé", "Søren",
    "Björn", "Renée", "Noël", "Mónica", "André", "Inês", "Joseph", "Mary", "Thomas", "Grace",
)  # fmt: skip
LAST_NAMES = (
    "Sharma", "Verma", "Gupta", "Patel", "Reddy", "Iyer", "Nair", "Menon", "Rao", "Singh",
    "Kumar", "Das", "Bose", "Chatterjee", "Banerjee", "Joshi", "Kulkarni", "Pillai", "Khan",
    "Ahmed", "Mehta", "Shah", "Agarwal", "Jain", "Kapoor", "Malhotra", "Fernandes", "D'Souza",
    "O'Brien", "García", "Müller", "Øster", "Dubois", "Lefèvre", "Núñez", "Brandão", "Kowalski",
    "Novák", "Thomas", "George",
)  # fmt: skip
DOMAINS = ("gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "example.com", "shop.co.in")
# City -> its old or informal name (None when there isn't a common one).
CITIES = {
    "Mumbai": "Bombay", "Delhi": None, "Bengaluru": "Bangalore", "Chennai": "Madras",
    "Kolkata": "Calcutta", "Hyderabad": None, "Pune": "Poona", "Ahmedabad": None, "Jaipur": None,
    "Lucknow": None, "Kochi": "Cochin", "Thiruvananthapuram": "Trivandrum", "Chandigarh": None,
    "Indore": None, "Bhopal": None, "Nagpur": None, "Surat": None, "Vadodara": "Baroda",
    "Coimbatore": None, "Mysuru": "Mysore", "Guwahati": None, "Bhubaneswar": None, "Patna": None,
    "Visakhapatnam": "Vizag", "Varanasi": "Benares",
}  # fmt: skip
CATEGORIES = (
    "Electronics", "Home & Kitchen", "Clothing", "Books", "Sports",
    "Beauty", "Toys", "Grocery", "Furniture", "Footwear",
)  # fmt: skip
STATUSES = {  # status -> a typo of it
    "Delivered": "Delivred", "Shipped": "Shiped", "Pending": "Pendng",
    "Cancelled": "Canceled", "Returned": "Retruned",
}  # fmt: skip
PAYMENTS = {  # payment method -> an abbreviation people use for it
    "UPI": None, "Credit Card": "CC", "Debit Card": None,
    "Cash on Delivery": "COD", "Net Banking": None, "Wallet": None,
}  # fmt: skip
NOTES = (
    "Gift order", "Leave at the door", "Call before delivery", "Fragile, handle with care",
    'Customer said "urgent"', "Deliver after 6 pm", "Leave with security\nCall on arrival",
)  # fmt: skip
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)  # fmt: skip
BOOL_WORDS = {
    "clean": ("Yes", "No"), "Y/N": ("Y", "N"), "TRUE/FALSE": ("TRUE", "FALSE"),
    "true/false": ("true", "false"), "YES/no": ("YES", "no"),
}  # fmt: skip

# Mess the README says Pandalyzer does not fix.
NOT_HANDLED = {"synonym", "typo", "number word"}
# Cells written in the column's normal format, or left blank on purpose.
UNTOUCHED = {"clean", "blank"}

# (name after cleaning, name as the export writes it, type after cleaning)
COLUMNS = [
    ("Order ID", " Order ID", "text"),
    ("Order Date", "Order Date", "date"),
    ("Customer Name", "Customer  Name", "text"),
    ("Email", "Email", "text"),
    ("Phone", "Phone", "text"),
    ("City", "City", "text"),
    ("Store Code", "Store Code", "text"),
    ("Category", "Category", "text"),
    ("Product Code", "Product Code", "text"),
    ("Quantity", "Quantity", "int"),
    ("Unit Price", "Unit\nPrice", "float"),
    ("Discount", "Discount", "int"),
    ("Shipping (EUR)", "Shipping (EUR) ", "float"),
    ("Order Total", "Order Total", "float"),
    ("Paid", "Paid", "bool"),
    ("Status", "Status", "text"),
    ("Payment Method", "Payment Method", "text"),
    ("Rating", "Rating", "int"),
    ("Tracking Number", "Tracking Number", "text"),
    ("Notes", "Notes", "text"),
]


# --- the right answers ------------------------------------------------------


def make_orders(n: int, r: random.Random) -> list[dict]:
    """Clean orders: the values every messy cell should come back as."""
    customers = []
    for k in range(max(50, n // 20)):
        first, last = r.choice(FIRST_NAMES), r.choice(LAST_NAMES)
        mobile = str(r.randrange(6_000_000_000, 10_000_000_000))
        phone = r.choice(
            (f"+91 {mobile[:5]} {mobile[5:]}", f"0{mobile[:5]} {mobile[5:]}", mobile,
             f"+91-{mobile[:5]}-{mobile[5:]}")
        )  # fmt: skip
        customers.append(
            {
                "Customer Name": f"{first} {last}",
                "Email": f"{first}.{last}{k}@{r.choice(DOMAINS)}".lower(),
                "Phone": phone,
                "City": r.choice(list(CITIES)),
            }
        )
    products = [
        (f"{code:05d}", r.choice(CATEGORIES), round(10 ** r.uniform(1.3, 4.5), 2))
        for code in range(1, max(100, n // 50) + 1)
    ]
    first_day = date(2023, 1, 1)
    orders = []
    for i in range(n):
        code, category, price = r.choice(products)
        quantity = r.choices(range(1, 11), weights=(30, 25, 15, 10, 6, 5, 3, 3, 2, 1))[0]
        discount = r.choice((0, 0, 0, 5, 10, 15, 20, 25, 30))
        status = r.choices(list(STATUSES), weights=(60, 15, 10, 8, 7))[0]
        total = round(quantity * price * (1 - discount / 100), 2)
        if r.random() < 0.01:  # a refund
            total, status = -total, "Returned"
        if r.random() < 0.2:
            shipping = 0.0
        else:
            shipping = round(r.uniform(1000, 5000) if r.random() < 0.05 else r.uniform(1, 300), 2)
        orders.append(
            {
                "Order ID": f"ORD-{i + 1:07d}",
                "Order Date": first_day + timedelta(days=r.randrange(1277)),
                **r.choice(customers),
                "Store Code": None if r.random() < 0.3 else f"{r.randrange(1, 121):04d}",
                "Category": category,
                "Product Code": code,
                "Quantity": quantity,
                "Unit Price": price,
                "Discount": discount,
                "Shipping (EUR)": shipping,
                "Order Total": total,
                "Paid": r.random() < 0.85,
                "Status": status,
                "Payment Method": r.choices(list(PAYMENTS), weights=(35, 20, 12, 18, 8, 7))[0],
                "Rating": (
                    None if r.random() < 0.35 else r.choices(range(1, 6), (5, 8, 15, 35, 37))[0]
                ),
                "Tracking Number": (
                    str(r.randrange(10**21, 10**22))
                    if status in ("Shipped", "Delivered", "Returned")
                    else None
                ),
                "Notes": None if r.random() < 0.7 else r.choice(NOTES),
            }
        )
    return orders


# --- making it messy ----------------------------------------------------------
# Each renderer takes (clean value, rng, whole order) and returns
# (text written to the file, value cleaning should produce, kind of mess).


def pick(r: random.Random, *table: tuple[float, str]) -> str:
    """Pick a kind of mess from (probability, kind) pairs, or "clean"."""
    x = r.random()
    for probability, kind in table:
        if x < probability:
            return kind
        x -= probability
    return "clean"


def pad(r: random.Random, text: str) -> str:
    return r.choice(PADDING) + text + r.choice(PADDING)


def restyle(kind: str, text: str, r: random.Random) -> str:
    """Apply a cosmetic kind of mess that cleaning should undo."""
    if kind == "whitespace":
        return pad(r, text)
    if kind == "doubled spaces":
        return text.replace(" ", r.choice(("  ", "   ")))
    if kind == "UPPER":
        return text.upper()
    if kind == "lower":
        return text.lower()
    return text


def placeholder(r: random.Random):
    return r.choice(PLACEHOLDERS), None, "placeholder"


def lakh(value: float) -> str:
    """Indian digit grouping: 1,23,456.00."""
    whole, cents = f"{value:.2f}".split(".")
    head, groups = whole[:-3], [whole[-3:]]
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(groups) + "." + cents


def european(value: float) -> str:
    """European notation: 1.234,56."""
    return f"{value:,.2f}".translate(str.maketrans(",.", ".,"))


DATE_FORMATS = {
    "clean": lambda d: d.isoformat(),
    "dd/mm/yyyy": lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
    "d Mon yyyy": lambda d: f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
    "Month d, yyyy": lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    "yyyy/mm/dd": lambda d: f"{d.year}/{d.month:02d}/{d.day:02d}",
    "dd-mm-yyyy": lambda d: f"{d.day:02d}-{d.month:02d}-{d.year}",
}


def order_id(v, r, _order):
    kind = pick(r, (0.03, "whitespace"))
    return restyle(kind, v, r), v, kind


def order_date(d, r, _order):
    kind = pick(
        r, (0.25, "dd/mm/yyyy"), (0.10, "d Mon yyyy"), (0.05, "Month d, yyyy"),
        (0.05, "yyyy/mm/dd"), (0.05, "dd-mm-yyyy"), (0.03, "whitespace"), (0.02, "placeholder"),
        (0.003, "impossible date"),
    )  # fmt: skip
    if kind == "placeholder":
        return placeholder(r)
    if kind == "impossible date":
        return f"{d.year}-02-{r.choice((30, 31))}", None, kind
    if kind == "whitespace":
        return pad(r, d.isoformat()), d, kind
    return DATE_FORMATS[kind](d), d, kind


def customer_name(v, r, _order):
    kind = pick(
        r, (0.05, "whitespace"), (0.03, "doubled spaces"), (0.02, "UPPER"), (0.02, "lower"),
        (0.01, "placeholder"),
    )  # fmt: skip
    return placeholder(r) if kind == "placeholder" else (restyle(kind, v, r), v, kind)


def email(v, r, order):
    kind = pick(
        r, (0.03, "UPPER"), (0.03, "Mixed.Case"), (0.04, "whitespace"), (0.01, "mailto:"),
        (0.01, "display name"), (0.015, "malformed"), (0.02, "placeholder"),
    )  # fmt: skip
    if kind == "placeholder":
        return placeholder(r)
    local, domain = v.split("@")
    if kind == "malformed":
        spaced = local.replace(".", " ")
        broken = (f"@{domain}", f"{local}@", f"{local}@@{domain}", f"{spaced}@{domain}")
        return r.choice(broken), None, kind
    text = {
        "Mixed.Case": f"{local.title()}@{domain.capitalize()}",
        "mailto:": f"mailto:{v}",
        "display name": f"{order['Customer Name']} <{v}>",
    }.get(kind) or restyle(kind, v, r)
    return text, v, kind


def phone(v, r, _order):
    kind = pick(r, (0.03, "whitespace"), (0.02, "placeholder"))
    return placeholder(r) if kind == "placeholder" else (restyle(kind, v, r), v, kind)


def label(alternatives=None, alternative_kind="synonym", rate=0.0):
    """Renderer for a column of labels (cities, statuses...)."""

    def render(v, r, _order):
        alternative = (alternatives or {}).get(v)
        kind = pick(
            r, (0.04, "UPPER"), (0.04, "lower"), (0.06, "whitespace"), (0.02, "placeholder"),
            (rate if alternative else 0.0, alternative_kind),
        )  # fmt: skip
        if kind == "placeholder":
            return placeholder(r)
        if kind == alternative_kind:
            return alternative, v, kind
        return restyle(kind, v, r), v, kind

    return render


def store_code(v, _r, _order):
    # Zero-padded digits, blank for online orders. The zeros survive here only because
    # the repeated header rows make pandas read every column as text; in a file where
    # the column holds nothing but digits and blanks, load_csv returns 42.0 for "0042".
    return ("", None, "blank") if v is None else (v, v, "zero-padded code")


def product_code(v, r, _order):
    kind = pick(r, (0.10, 'Excel ="00042"'), (0.03, "whitespace"), (0.01, "placeholder"))
    if kind == "placeholder":
        return placeholder(r)
    if kind == 'Excel ="00042"':
        return f'="{v}"', v, kind
    return restyle(kind, v, r), v, kind


def quantity(v, r, _order):
    kind = pick(
        r, (0.03, "whitespace"), (0.03, "3.0"), (0.02, "placeholder"), (0.003, "number word"),
        (0.002, "junk"),
    )  # fmt: skip
    if kind == "placeholder":
        return placeholder(r)
    if kind == "junk":
        return r.choice(("??", "#VALUE!", "see notes")), None, kind
    if kind == "number word":
        return NUMBER_WORDS[v], v, kind
    return (f"{v}.0" if kind == "3.0" else restyle(kind, str(v), r)), v, kind


def unit_price(v, r, _order):
    kind = pick(
        r, (0.15, "1,234.50"), (0.08, "₹ symbol"), (0.03, "INR code"), (0.03, "whitespace"),
        (0.02, "placeholder"), (0.002, "junk"),
    )  # fmt: skip
    if kind == "placeholder":
        return placeholder(r)
    if kind == "junk":
        return r.choice(("call for price", "price on request")), None, kind
    grouped = f"{v:,.2f}"
    if kind == "1,234.50":
        return grouped, v, kind
    if kind == "₹ symbol":
        return r.choice((f"₹{grouped}", f"₹ {grouped}")), v, kind
    if kind == "INR code":
        return r.choice((f"INR {grouped}", f"{grouped} INR")), v, kind
    return restyle(kind, f"{v:.2f}", r), v, kind


def discount(v, r, _order):
    kind = pick(r, (0.50, "15%"), (0.02, "whitespace"), (0.03, "placeholder"))
    if kind == "placeholder":
        return placeholder(r)
    return (f"{v}%" if kind == "15%" else restyle(kind, str(v), r)), v, kind


def shipping(v, r, _order):
    kind = pick(r, (0.10, "€ symbol"), (0.03, "whitespace"), (0.02, "placeholder"))
    if kind == "placeholder":
        return placeholder(r)
    text = european(v)
    if kind == "€ symbol":
        text = r.choice((f"€ {text}", f"{text} €", f"€{text}"))
    return restyle(kind, text, r), v, kind


def order_total(v, r, _order):
    if v < 0:
        kind = r.choice(("(1,234.56)", "-1234.56"))
        return (f"({-v:,.2f})" if kind == "(1,234.56)" else f"{v:.2f}"), v, kind
    kind = pick(r, (0.08, "1,234.56"), (0.02, "whitespace"), (0.02, "placeholder"))
    if v >= 100_000 and r.random() < 0.3:
        kind = "lakh 1,23,456.00"
    if kind == "placeholder":
        return placeholder(r)
    if kind == "1,234.56":
        return f"{v:,.2f}", v, kind
    if kind == "lakh 1,23,456.00":
        return lakh(v), v, kind
    return restyle(kind, f"{v:.2f}", r), v, kind


def paid(v, r, _order):
    kind = pick(
        r, (0.15, "Y/N"), (0.10, "TRUE/FALSE"), (0.10, "true/false"), (0.08, "YES/no"),
        (0.02, "whitespace"), (0.03, "placeholder"),
    )  # fmt: skip
    if kind == "placeholder":
        return placeholder(r)
    yes, no = BOOL_WORDS.get(kind, BOOL_WORDS["clean"])
    return restyle(kind, yes if v else no, r), v, kind


def rating(v, r, _order):
    if v is None:
        return "", None, "blank"
    kind = pick(r, (0.02, "whitespace"), (0.02, "placeholder"))
    return placeholder(r) if kind == "placeholder" else (restyle(kind, str(v), r), v, kind)


def tracking_number(v, r, _order):
    if v is None:
        return "", None, "blank"
    kind = pick(r, (0.02, "whitespace"), (0.01, "placeholder"))
    return placeholder(r) if kind == "placeholder" else (restyle(kind, v, r), v, kind)


def notes(v, r, _order):
    if v is None:
        return placeholder(r) if r.random() < 0.1 else ("", None, "blank")
    kind = pick(r, (0.05, "whitespace"), (0.03, "doubled spaces"))
    if kind == "clean" and "\n" in v:
        kind = "line break inside"
    elif kind == "clean" and ("," in v or '"' in v):
        kind = "comma or quote inside"
    return restyle(kind, v, r), v, kind


RENDERERS = {
    "Order ID": order_id,
    "Order Date": order_date,
    "Customer Name": customer_name,
    "Email": email,
    "Phone": phone,
    "City": label(CITIES, "synonym", 0.01),
    "Store Code": store_code,
    "Category": label(),
    "Product Code": product_code,
    "Quantity": quantity,
    "Unit Price": unit_price,
    "Discount": discount,
    "Shipping (EUR)": shipping,
    "Order Total": order_total,
    "Paid": paid,
    "Status": label(STATUSES, "typo", 0.005),
    "Payment Method": label(PAYMENTS, "synonym", 0.01),
    "Rating": rating,
    "Tracking Number": tracking_number,
    "Notes": notes,
}


def write_csv(path: Path, n_rows: int, r: random.Random):
    """Write the messy file; return the order IDs and, per column, each cell's text/answer/kind."""
    n_duplicates, n_near, n_empty = n_rows // 100, n_rows // 200, n_rows * 3 // 1000
    n_orders = n_rows - n_duplicates - n_near - n_empty - 4 - 2  # 4 repeated headers, 2 totals
    orders = make_orders(n_orders, r)
    names = [name for name, _, _ in COLUMNS]
    rendered = {name: [] for name in names}
    expected = {name: [] for name in names}
    kinds = {name: [] for name in names}
    rows = []
    for order in orders:
        row = []
        for name in names:
            text, want, kind = RENDERERS[name](order[name], r, order)
            row.append(text)
            rendered[name].append(text)
            expected[name].append(want)
            kinds[name].append(kind)
        rows.append([*row, ""])  # a trailing delimiter on every line, as some exports write
    for i in r.sample(range(n_orders), n_rows // 500):
        rows[i].append("extra")  # one field more than the header

    # Sort keys place each row; copies land after their original, like overlapping exports.
    placed = list(enumerate(rows))
    for _ in range(n_duplicates):
        i = r.randrange(n_orders)
        placed.append((r.uniform(i, n_orders), list(rows[i])))
    for _ in range(n_near):  # the same row again with stray spaces around every value
        i = r.randrange(n_orders)
        placed.append((r.uniform(i, n_orders), [pad(r, f) if f else f for f in rows[i]]))
    header = [*(written for _, written, _ in COLUMNS), ""]
    for k in range(4):  # concatenated exports repeat the header; one copy is upper-cased
        placed.append((r.uniform(0, n_orders), [h.upper() for h in header] if k == 0 else header))
    for _ in range(n_empty):
        placed.append((r.uniform(0, n_orders), [""] * len(header)))
    for _ in range(n_rows // 2000):  # blank lines, which pandas skips
        placed.append((r.uniform(0, n_orders), None))
    for offset, text in enumerate(("Total", "Grand total"), start=1):
        total = [""] * len(header)
        total[0] = text
        total[names.index("Quantity")] = str(sum(o["Quantity"] for o in orders))
        total[names.index("Order Total")] = f"{sum(o['Order Total'] for o in orders):.2f}"
        placed.append((n_orders + offset, total))
    placed.sort(key=lambda item: item[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("Retail order export\r\nGenerated: Nov 19, 2026 10:42\r\n\r\n")
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        for _, row in placed:
            if row is None:
                f.write("\r\n")
            else:
                writer.writerow(row)
    return [o["Order ID"] for o in orders], rendered, expected, kinds


# --- scoring --------------------------------------------------------------------


def matches(got: pd.Series, want: list, kind: str) -> pd.Series:
    """True where a cleaned cell holds the right value (or is missing where it should be)."""
    want = pd.Series(want, index=got.index, dtype=object)
    result = got.isna() & want.isna()
    both = got.notna() & want.notna()
    g, w = got[both], want[both]
    if kind == "text":
        ok = g.map(lambda v: isinstance(v, str)).astype(bool) & (g.astype(object) == w)
    elif kind == "bool":
        ok = (g.astype(object) == w) if pd.api.types.is_bool_dtype(got) else False
    elif kind == "date":
        is_date = pd.api.types.is_datetime64_any_dtype(got)
        ok = (g.dt.normalize() == pd.to_datetime(w)) if is_date else False
    elif pd.api.types.is_numeric_dtype(got) and not pd.api.types.is_bool_dtype(got):
        ok = (g.astype("float64") - w.astype("float64")).abs() < 0.005
    else:
        ok = False
    result[both] = ok
    return result


def score(path: Path, n_rows: int, ids, rendered, expected, kinds, *, dayfirst: bool) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        raw = load_csv(path)
        loaded = time.perf_counter()
        clean = auto_clean(raw, dayfirst=dayfirst)
        cleaned = time.perf_counter()

    print(f"\nload_csv    {loaded - started:5.1f} s   {len(raw):,} rows x {raw.shape[1]} columns")
    print(
        f"auto_clean  {cleaned - loaded:5.1f} s   {len(clean):,} rows x {clean.shape[1]} columns"
        f"   (dayfirst={dayfirst})"
    )
    for category, count in Counter(type(w.message).__name__ for w in caught).most_common():
        first = next(str(w.message) for w in caught if type(w.message).__name__ == category)
        print(f"  {count} {category}: {first.splitlines()[0][:110]}")

    id_column = "Order ID"
    names = [name for name, _, _ in COLUMNS]
    if id_column not in clean.columns:
        print(f"\nNo {id_column!r} column after cleaning. Columns: {list(clean.columns)}")
        return
    known = clean[id_column].isin(set(ids))
    leftovers = clean.loc[~known, id_column]
    print(f"\n{'Structure':<40}{'want':>10}{'got':>10}")
    for text, want, got in (
        ("rows loaded", n_rows, len(raw)),
        ("rows after cleaning", len(ids), len(clean)),
        ("header, Total or empty rows left", 0, len(leftovers)),
        ("duplicate orders left", 0, int(clean.loc[known, id_column].duplicated().sum())),
        ("orders lost", 0, len(ids) - clean.loc[known, id_column].nunique()),
        ("columns as expected", len(names), sum(name in clean.columns for name in names)),
    ):
        print(f"  {text:<38}{want:>10,}{got:>10,}{'' if want == got else '   <- check'}")
    if len(leftovers):
        print(f"    left over: {leftovers.head(6).tolist()}")
    unexpected = [c for c in clean.columns if c not in names]
    if unexpected:
        print(f"    unexpected columns: {unexpected}")
    print(f"  title lines kept in df.attrs['preamble']: {raw.attrs.get('preamble')}")

    aligned = clean[known].drop_duplicates(id_column).set_index(id_column).reindex(ids)
    totals, rights, examples = Counter(), Counter(), {}
    print(f"\n{'Column':<18}{'dtype':<17}{'right':>8}{'wrong':>8}")
    for name, _, kind in COLUMNS[1:]:
        if name in aligned.columns:
            got = aligned[name]
        else:
            got = pd.Series(pd.NA, index=aligned.index, dtype=object)
        ok = matches(got, expected[name], kind).to_numpy()
        for i, (mess, good) in enumerate(zip(kinds[name], ok, strict=True)):
            totals[name, mess] += 1
            if good:
                rights[name, mess] += 1
            elif (name, mess) not in examples:
                examples[name, mess] = (rendered[name][i], got.iloc[i], expected[name][i])
        print(f"  {name:<16}{got.dtype!s:<17}{ok.mean():>8.2%}{len(ok) - ok.sum():>8,}")

    def tally(keys) -> str:
        good, total = sum(rights[k] for k in keys), sum(totals[k] for k in keys)
        return f"{good:,} of {total:,} ({good / total:.2%})" if total else "none"

    messy = [k for k in totals if k[1] not in UNTOUCHED | NOT_HANDLED]
    untouched = [k for k in totals if k[1] in UNTOUCHED]
    not_handled = [k for k in totals if k[1] in NOT_HANDLED]
    print(f"\nCells with mess Pandalyzer should fix, right: {tally(messy)}")
    print(f"Cells already in the column's normal format, right: {tally(untouched)}")
    print(f"Mess the README says it leaves alone, fixed anyway: {tally(not_handled)}")

    wrong = sorted(
        ((totals[k] - rights[k], k) for k in totals if rights[k] < totals[k]), reverse=True
    )
    if wrong:
        print("\nWhat came out wrong, most cells first (* = the README says it doesn't fix this)")
    for count, (name, mess) in wrong:
        text, got, want = examples[name, mess]
        mark = "*" if mess in NOT_HANDLED else " "
        print(
            f" {mark}{name:<16}{mess:<20}{count:>7,} of {totals[name, mess]:>6,}"
            f"   e.g. {text!r} -> {got!r}, want {want!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=100_000, help="data rows (default 100000)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed (default 2026)")
    parser.add_argument("--out", type=Path, help="default: database/messy_orders_<rows>.csv")
    parser.add_argument("--no-check", action="store_true", help="only write the file")
    parser.add_argument(
        "--dayfirst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass dayfirst=True to auto_clean (the file's dates are DD/MM; default: on)",
    )
    args = parser.parse_args()
    if args.rows < 1000:
        parser.error("--rows must be at least 1000")
    size = f"{args.rows // 1000}k" if args.rows % 1000 == 0 else str(args.rows)
    out = args.out or ROOT / "database" / f"messy_orders_{size}.csv"

    started = time.perf_counter()
    truth = write_csv(out, args.rows, random.Random(args.seed))
    print(
        f"Wrote {out} ({args.rows:,} rows, {out.stat().st_size / 1e6:.1f} MB)"
        f" in {time.perf_counter() - started:.1f} s"
    )
    if not args.no_check:
        score(out, args.rows, *truth, dayfirst=args.dayfirst)


if __name__ == "__main__":
    main()
