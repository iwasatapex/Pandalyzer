"""Deterministic generator for the 10 end-to-end CSV fixtures in ``tests/data``.

Each fixture is a 100-row file with its own schema, format and planted
problems (duplicates, empty rows, placeholders, outliers, mixed formats).
Alongside the file, each fixture records the ground truth the tests check:
the raw shape, the dtype kind every column should end up with after
``auto_clean``, how many rows survive, how many out-of-range values the
validators must flag, and exact expected values for a few cells.

Regenerate the files after editing this module with::

    python -m tests.csv_fixtures
"""

import csv
import gzip
import io
import json
import random
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
ROWS = 100
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class CsvFixture:
    file_name: str
    description: str
    header: list[str]
    rows: list[list[str]]
    sep: str = ","
    encoding: str = "utf-8"
    container: str = "plain"  # "plain", "gzip" or "zip"
    load_kwargs: dict = field(default_factory=dict)
    clean_kwargs: dict = field(default_factory=dict)
    raw_columns: list[str] = field(default_factory=list)
    raw_rows: int = ROWS
    clean_rows: int = ROWS
    # Column -> "int" | "float" | "bool" | "datetime" | "text" after auto_clean.
    clean_kinds: dict[str, str] = field(default_factory=dict)
    # Column -> (minimum, maximum, number of values outside the range).
    ranges: dict[str, tuple] = field(default_factory=dict)
    # (row label, column) -> exact value expected after auto_clean.
    checks: dict[tuple[int, str], object] = field(default_factory=dict)
    # Row labels auto_clean must drop (duplicates and empty rows).
    dropped: set[int] = field(default_factory=set)
    expect_parser_warning: bool = False

    def __post_init__(self):
        assert len(self.rows) == ROWS, f"{self.file_name} has {len(self.rows)} rows"

    def to_bytes(self) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter=self.sep, lineterminator="\n")
        writer.writerow(self.header)
        writer.writerows(self.rows)
        payload = buffer.getvalue().encode(self.encoding)
        if self.container == "gzip":
            out = io.BytesIO()
            with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as archive:
                archive.write(payload)
            return out.getvalue()
        if self.container == "zip":
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo(self.file_name.removesuffix(".zip") + ".csv", (2024, 1, 1, 0, 0, 0))
                archive.writestr(info, payload)
            return out.getvalue()
        return payload

    def payload(self, data: bytes) -> bytes:
        """The uncompressed CSV bytes inside ``data`` (compressed bytes vary by zlib version)."""
        if self.container == "gzip":
            return gzip.decompress(data)
        if self.container == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return archive.read(archive.namelist()[0])
        return data


def _vary_case(rng: random.Random, value: str) -> str:
    return rng.choice([value.upper(), value.lower(), f"  {value} "])


# ---------------------------------------------------------------------------
# 1. Retail orders: comma, UTF-8, mixed date formats, ₹ prices, Yes/No
# ---------------------------------------------------------------------------


def retail_orders() -> CsvFixture:
    rng = random.Random(1)
    customers = ["Alice Smith", "Bob Jones", "Carol White", "Dan Brown", "Eve Black"]
    channels = ["Web", "Store", "Mobile App"]
    rows, checks = [], {}
    for i in range(ROWS):
        month, day, hour = rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23)
        when = [
            (f"2024-{month:02d}-{day:02d} {hour:02d}:15:00", datetime(2024, month, day, hour, 15)),
            (f"{day:02d}-{MONTHS[month - 1]}-2024", datetime(2024, month, day)),
            (f"{month:02d}/{day:02d}/2024 {hour:02d}:30", datetime(2024, month, day, hour, 30)),
        ][i % 3]
        quantity = rng.randint(1, 10)
        price = round(rng.uniform(100, 5000), 2)
        paid = rng.choice(["Yes", "No"])
        channel, customer = rng.choice(channels), rng.choice(customers)
        rows.append(
            [
                f"ORD{i:04d}",
                when[0],
                customer.lower() if i % 13 == 0 else customer,
                f" {quantity} " if i % 17 == 3 else str(quantity),
                f"₹{price:,.2f}" if i % 4 == 0 else f"{price:.2f}",
                f" {paid.upper()} " if i % 11 == 0 else paid,
                channel.upper() if i % 9 == 0 else channel,
            ]
        )
        if i in (1, 2, 4, 5):
            checks[(i, "order_date")] = pd.Timestamp(when[1])
            checks[(i, "unit_price")] = price
            checks[(i, "paid")] = paid == "Yes"
            checks[(i, "quantity")] = quantity
    rows[20][3] = rows[40][3] = "unknown"
    rows[30][3], rows[60][3] = "-1", "500"
    for target, source in ((90, 0), (91, 1), (92, 2)):  # Near-duplicates: spacing/case only.
        rows[target] = list(rows[source])
        rows[target][0] = f"  {rows[source][0]} "
        rows[target][6] = rows[source][6].upper()
    rows[95] = [""] * 7
    checks[(4, "customer")] = rows[4][2]
    checks[(20, "quantity")] = pd.NA
    return CsvFixture(
        file_name="retail_orders.csv",
        description="comma, UTF-8, mixed date formats, ₹ prices, Yes/No, near-duplicates",
        header=["order_id", "order_date", "customer", "quantity", "unit_price", "paid", "channel"],
        rows=rows,
        raw_columns=["order_id", "order_date", "customer", "quantity", "unit_price", "paid", "channel"],
        clean_rows=96,
        clean_kinds={
            "order_id": "text", "order_date": "datetime", "customer": "text", "quantity": "int",
            "unit_price": "float", "paid": "bool", "channel": "text",
        },
        ranges={"quantity": (1, 50, 2)},
        checks=checks,
        dropped={90, 91, 92, 95},
    )


# ---------------------------------------------------------------------------
# 2. Employees (Europe): semicolon, Windows-1252, decimal comma, dd.mm.yyyy
# ---------------------------------------------------------------------------


def _eu_number(value: float) -> str:
    return f"{value:,.2f}".translate(str.maketrans({",": ".", ".": ","}))


def employees_eu() -> CsvFixture:
    rng = random.Random(2)
    names = ["José Müller", "Zoë Dupont", "François Lévesque", "Søren Æbelø", "Ana Peña", "Jürgen Groß"]
    departments = ["Vertrieb", "Technik", "Personal", "Einkauf"]
    rows, checks = [], {}
    for i in range(ROWS):
        salary = round(rng.uniform(500, 999), 2) if i % 10 == 0 else round(rng.uniform(2000, 9000), 2)
        day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2015, 2023)
        remote = rng.choice(["true", "false"])
        rows.append(
            [
                f"E-{i:03d}",
                f"{rng.choice(names)} {i}",
                rng.choice(departments),
                _eu_number(salary),
                f"{day:02d}.{month:02d}.{year}",
                remote,
            ]
        )
        if i in (0, 1, 3):
            checks[(i, "Gehalt")] = salary
            checks[(i, "Eintritt")] = pd.Timestamp(year, month, day)
            checks[(i, "Remote")] = remote == "true"
    rows[50][3] = "-1.200,00"
    rows[98], rows[99] = list(rows[10]), list(rows[11])
    checks[(50, "Gehalt")] = -1200.0
    checks[(1, "Name")] = rows[1][1]
    return CsvFixture(
        file_name="employees_eu.csv",
        description="semicolon, Windows-1252, decimal comma (3.450,75), dd.mm.yyyy dates",
        header=["Mitarbeiter-ID", "Name", "Abteilung", "Gehalt", "Eintritt", "Remote"],
        rows=rows,
        sep=";",
        encoding="cp1252",
        clean_kwargs={"dayfirst": True},
        raw_columns=["Mitarbeiter-ID", "Name", "Abteilung", "Gehalt", "Eintritt", "Remote"],
        clean_rows=98,
        clean_kinds={
            "Mitarbeiter-ID": "text", "Name": "text", "Abteilung": "text", "Gehalt": "float",
            "Eintritt": "datetime", "Remote": "bool",
        },
        ranges={"Gehalt": (0, None, 1)},
        checks=checks,
        dropped={98, 99},
    )


# ---------------------------------------------------------------------------
# 3. Sensor readings: tab, ISO timestamps, ERR values, %, scientific notation
# ---------------------------------------------------------------------------


def sensor_readings() -> CsvFixture:
    rng = random.Random(3)
    start = datetime(2024, 5, 1)
    rows, checks = [], {}
    for i in range(ROWS):
        stamp = start + timedelta(minutes=15 * i)
        temperature = round(rng.uniform(-10, 35), 2)
        humidity = rng.randint(20, 90)
        pressure = rng.uniform(990, 1030)
        battery = rng.choice(["true", "false"])
        rows.append(
            [
                f"S-{i % 10 + 1:02d}",
                stamp.isoformat(),
                str(temperature),
                f"{humidity}%",
                f"{pressure:.4e}",
                battery.upper() if i % 10 == 0 else battery,
                str(rng.randint(0, 1000)),
            ]
        )
        if i in (0, 1, 2):
            checks[(i, "timestamp")] = pd.Timestamp(stamp)
            checks[(i, "temperature_c")] = temperature
            checks[(i, "humidity_pct")] = humidity
            checks[(i, "pressure_hpa")] = float(f"{pressure:.4e}")
            checks[(i, "battery_ok")] = battery == "true"
    for i in (7, 47, 77):
        rows[i][2] = "ERR"
        checks[(i, "temperature_c")] = None
    rows[30][2] = rows[31][2] = ""
    rows[20][2], rows[60][2] = "999.9", "-100.0"
    return CsvFixture(
        file_name="sensor_readings.tsv",
        description="tab-separated, ISO timestamps, ERR readings, 45% humidity, 1.0132e+03",
        header=["sensor_id", "timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
                "battery_ok", "reading_count"],
        rows=rows,
        sep="\t",
        raw_columns=["sensor_id", "timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
                     "battery_ok", "reading_count"],
        clean_kinds={
            "sensor_id": "text", "timestamp": "datetime", "temperature_c": "float",
            "humidity_pct": "int", "pressure_hpa": "float", "battery_ok": "bool",
            "reading_count": "int",
        },
        ranges={"temperature_c": (-40, 60, 2)},
        checks=checks,
    )


# ---------------------------------------------------------------------------
# 4. Students: pipe, UTF-16, Unicode names, dd/mm/yyyy, Y/N, empty rows
# ---------------------------------------------------------------------------


def students() -> CsvFixture:
    rng = random.Random(4)
    names = ["Søren Kierkegaard", "李雷", "Анна Петрова", "Zoë Saldana", "Ola Nordmann"]
    rows, checks = [], {}
    for i in range(ROWS):
        grade = rng.randint(40, 100)
        day, month = rng.randint(1, 28), rng.randint(1, 12)
        gpa = round(rng.uniform(1.5, 4.0), 2)
        rows.append(
            [
                f"{i + 1:05d}",
                rng.choice(names),
                str(grade),
                "Y" if grade >= 50 else "N",
                f"{day:02d}/{month:02d}/2023",
                f"{gpa:.2f}",
                rng.choice(["Chess", "Drama", "Robotics", ""]),
            ]
        )
        if i in (0, 2, 5):
            checks[(i, "student_id")] = f"{i + 1:05d}"
            checks[(i, "grade")] = grade
            checks[(i, "passed")] = grade >= 50
            checks[(i, "enrolled_on")] = pd.Timestamp(2023, month, day)
            checks[(i, "gpa")] = gpa
    rows[15][2], rows[85][2] = "105", "-5"
    rows[40] = [""] * 7
    rows[70] = [""] * 7
    checks[(1, "full_name")] = rows[1][1]
    return CsvFixture(
        file_name="students.psv",
        description="pipe-separated, UTF-16, Unicode names, zero-padded IDs, dd/mm/yyyy, Y/N",
        header=["student_id", "full_name", "grade", "passed", "enrolled_on", "gpa", "club"],
        rows=rows,
        sep="|",
        encoding="utf-16",
        load_kwargs={"dtype": {"student_id": str}},
        clean_kwargs={"dayfirst": True},
        raw_columns=["student_id", "full_name", "grade", "passed", "enrolled_on", "gpa", "club"],
        clean_rows=98,
        clean_kinds={
            "student_id": "text", "full_name": "text", "grade": "int", "passed": "bool",
            "enrolled_on": "datetime", "gpa": "float", "club": "text",
        },
        ranges={"grade": (0, 100, 2)},
        checks=checks,
        dropped={40, 70},
    )


# ---------------------------------------------------------------------------
# 5. Products: comma, Latin-1, quoted commas and newlines, $ prices, SKUs
# ---------------------------------------------------------------------------


def products() -> CsvFixture:
    rng = random.Random(5)
    names = ["Café, Crème Brûlée", "Jalapeño Salsa, Hot", "Pâté de Campagne", "Smörgåsbord Tray",
             "Piña Colada Mix"]
    categories = ["Kitchen", "Garden", "Pantry"]
    rows, checks = [], {}
    for i in range(ROWS):
        price = round(rng.uniform(5, 2500), 2)
        discount = rng.choice([0, 5, 10, 15, 20])
        in_stock = rng.choice(["yes", "no"])
        category = rng.choice(categories)
        if i % 8 == 0:
            category = category.upper()
        elif i % 12 == 0:
            category = category.lower()
        rows.append(
            [
                f"SKU-{i}" if i % 10 == 9 else f"{i + 100:05d}",
                f"{rng.choice(names)} #{i}",
                "Line one\nLine two, with comma" if i % 7 == 0 else "Simple item",
                f"${price:,.2f}",
                f"{discount}%",
                in_stock,
                category,
            ]
        )
        if i in (0, 1, 7):
            checks[(i, "sku")] = f"{i + 100:05d}"
            checks[(i, "product_name")] = rows[i][1]
            checks[(i, "description")] = rows[i][2]
            checks[(i, "price")] = price
            checks[(i, "discount")] = discount
            checks[(i, "in_stock")] = in_stock == "yes"
    rows[33][4] = "150%"
    rows[99] = list(rows[3])
    return CsvFixture(
        file_name="products_latin1.csv",
        description="comma, Latin-1, quoted commas and line breaks, $1,299.00, 10%, SKU codes",
        header=["sku", "product_name", "description", "price", "discount", "in_stock", "category"],
        rows=rows,
        encoding="latin-1",
        raw_columns=["sku", "product_name", "description", "price", "discount", "in_stock", "category"],
        clean_rows=99,
        clean_kinds={
            "sku": "text", "product_name": "text", "description": "text", "price": "float",
            "discount": "int", "in_stock": "bool", "category": "text",
        },
        ranges={"discount": (0, 100, 1)},
        checks=checks,
        dropped={99},
    )


# ---------------------------------------------------------------------------
# 6. Survey: gzip, messy headers, placeholder values, "Mar 5, 2024" dates
# ---------------------------------------------------------------------------


def survey() -> CsvFixture:
    rng = random.Random(6)
    rows, checks = [], {}
    for i in range(ROWS):
        age = rng.randint(18, 80)
        month, day, hour, minute = rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59)
        recommend = rng.choice(["Yes", "No", "yes", "NO"])
        rows.append(
            [
                f"R{i:03d}",
                rng.choice(["N/A", "-", "?", "", " "]) if i % 10 == 1 else str(age),
                str(rng.randint(1, 5)),
                recommend,
                rng.choice(["Great service", "Too slow, honestly", "n/a", "Loved it!"]) if i % 3 == 0 else "",
                f"{MONTHS[month - 1]} {day}, 2024 {hour:02d}:{minute:02d}",
            ]
        )
        if i in (0, 2, 3):
            checks[(i, "age")] = age
            checks[(i, "would_recommend")] = recommend.lower() == "yes"
            checks[(i, "submitted_at")] = pd.Timestamp(2024, month, day, hour, minute)
    rows[55][1] = "150"
    checks[(1, "age")] = pd.NA
    return CsvFixture(
        file_name="survey.csv.gz",
        description="gzip, messy headers renamed to snake_case, N/A/-/? placeholders, month-name dates",
        header=["Respondent ID", "Age", "Satisfaction (1-5)", "Would Recommend?", "Comments", "Submitted At"],
        rows=rows,
        container="gzip",
        clean_kwargs={"column_names": "snake"},
        raw_columns=["Respondent ID", "Age", "Satisfaction (1-5)", "Would Recommend?", "Comments", "Submitted At"],
        clean_kinds={
            "respondent_id": "text", "age": "int", "satisfaction_1_5": "int",
            "would_recommend": "bool", "comments": "text", "submitted_at": "datetime",
        },
        ranges={"age": (0, 120, 1)},
        checks=checks,
    )


# ---------------------------------------------------------------------------
# 7. Transactions: zip, 20-digit IDs, (accounting) negatives, mixed time zones
# ---------------------------------------------------------------------------


def transactions() -> CsvFixture:
    rng = random.Random(7)
    merchants = ["Corner Cafe", "City Books", "Fresh Mart", "Tech Hub"]
    rows, checks = [], {}
    for i in range(ROWS):
        amount = round(rng.uniform(1, 5000), 2)
        refund = rng.random() < 0.1
        fee = rng.choice([1.5, 2.0, 2.5, 3.0])
        day, hour = rng.randint(1, 28), rng.randint(0, 23)
        offset = "+05:30" if i % 2 == 0 else "Z"
        merchant = rng.choice(merchants)
        rows.append(
            [
                "9" + "".join(rng.choice("0123456789") for _ in range(19)),
                f"({amount:,.2f})" if refund else f"{amount:,.2f}",
                f"{fee}%",
                f"2024-01-{day:02d}T{hour:02d}:00:00{offset}",
                merchant.upper() if i % 9 == 0 else merchant,
                "True" if refund else "False",
                rng.choice(["INR", "USD", "EUR"]),
            ]
        )
        if i in (0, 1, 2, 3):
            local = pd.Timestamp(2024, 1, day, hour, tz="Asia/Kolkata" if offset != "Z" else "UTC")
            checks[(i, "txn_id")] = rows[i][0]
            checks[(i, "amount")] = -amount if refund else amount
            checks[(i, "fee_pct")] = fee
            checks[(i, "created_at")] = local.tz_convert("UTC")
            checks[(i, "is_refund")] = refund
    rows[97], rows[98] = list(rows[5]), list(rows[6])
    return CsvFixture(
        file_name="transactions.zip",
        description="zip archive, 20-digit IDs, (1,234.56) negatives, 2.5%, mixed UTC offsets",
        header=["txn_id", "amount", "fee_pct", "created_at", "merchant", "is_refund", "currency"],
        rows=rows,
        container="zip",
        raw_columns=["txn_id", "amount", "fee_pct", "created_at", "merchant", "is_refund", "currency"],
        clean_rows=98,
        clean_kinds={
            "txn_id": "text", "amount": "float", "fee_pct": "float", "created_at": "datetime",
            "merchant": "text", "is_refund": "bool", "currency": "text",
        },
        ranges={"fee_pct": (0, 5, 0)},
        checks=checks,
        dropped={97, 98},
    )


# ---------------------------------------------------------------------------
# 8. Server logs: trailing delimiter, ragged rows, empty rows, IPs, versions
# ---------------------------------------------------------------------------


def server_logs() -> CsvFixture:
    rng = random.Random(8)
    start = datetime(2024, 6, 1, 12)
    rows, checks = [], {}
    for i in range(ROWS):
        status = rng.choice([200, 200, 200, 301, 404, 500])
        duration = round(rng.uniform(1, 900), 1)
        rows.append(
            [
                (start + timedelta(seconds=37 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                f"10.0.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                rng.choice(["1.2.3", "1.10.0", "2.0.1"]),
                str(status),
                str(duration),
                rng.choice(["/", "/api/items", "/login"]),
                "",
            ]
        )
        if i in (0, 1, 12):
            checks[(i, "timestamp")] = pd.Timestamp(start + timedelta(seconds=37 * i))
            checks[(i, "client_ip")] = rows[i][1]
            checks[(i, "app_version")] = rows[i][2]
            checks[(i, "status")] = status
            checks[(i, "duration_ms")] = duration
    rows[44][3] = "999"
    rows[12].append("unexpected-extra-field")
    rows[62].append("unexpected-extra-field")
    rows[25] = [""] * 7
    rows[75] = [""] * 7
    return CsvFixture(
        file_name="server_logs.csv",
        description="trailing delimiter, ragged rows with extra fields, empty rows, IPs and versions",
        header=["timestamp", "client_ip", "app_version", "status", "duration_ms", "path", ""],
        rows=rows,
        raw_columns=["timestamp", "client_ip", "app_version", "status", "duration_ms", "path", "Unnamed: 6"],
        clean_rows=98,
        clean_kinds={
            "timestamp": "datetime", "client_ip": "text", "app_version": "text", "status": "int",
            "duration_ms": "float", "path": "text",
        },
        ranges={"status": (100, 599, 1)},
        checks=checks,
        dropped={25, 75},
        expect_parser_warning=True,
    )


# ---------------------------------------------------------------------------
# 9. Wide mixed: UTF-8 BOM, duplicate headers, 0/1 flags, constant/empty columns
# ---------------------------------------------------------------------------


def wide_mixed() -> CsvFixture:
    rng = random.Random(9)
    rows, checks = [], {}
    for i in range(ROWS):
        level = rng.choice(["low", "medium", "high"])
        if i % 6 == 0:
            level = level.upper()
        elif i % 7 == 0:
            level = level.title()
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        score = round(rng.uniform(0, 10), 1)
        rows.append(
            [
                str(i + 1),
                str(rng.randint(0, 100)),
                f"{rng.uniform(0, 1):.3f}",
                rng.choice("01"),
                "same",
                "",
                json.dumps(rng.sample(["a", "b", "c"], 2)),
                f"2024-{month:02d}-{day:02d}",
                f"  {score} ",
                level,
                "check later" if i % 5 == 0 else "",
                str(3_000_000_000 + rng.randrange(10**9)),
                f"{rng.random():.2f}",
            ]
        )
        if i in (0, 1):
            checks[(i, "json_tags")] = rows[i][6]
            checks[(i, "event_date")] = pd.Timestamp(2024, month, day)
            checks[(i, "score")] = score
            checks[(i, "big_count")] = int(rows[i][11])
            checks[(i, "constant")] = "same"
    return CsvFixture(
        file_name="wide_mixed.csv",
        description="UTF-8 BOM, duplicate header, 0/1 flags, constant and all-empty columns, JSON text",
        header=["id", "value", "value", "flag01", "constant", "all_empty", "json_tags", "event_date",
                "score", "level", "notes", "big_count", "ratio"],
        rows=rows,
        encoding="utf-8-sig",
        raw_columns=["id", "value", "value.1", "flag01", "constant", "all_empty", "json_tags",
                     "event_date", "score", "level", "notes", "big_count", "ratio"],
        clean_kinds={
            "id": "int", "value": "int", "value.1": "float", "flag01": "int", "constant": "text",
            "all_empty": "float", "json_tags": "text", "event_date": "datetime", "score": "float",
            "level": "text", "notes": "text", "big_count": "int", "ratio": "float",
        },
        ranges={"value": (0, 100, 0), "ratio": (0, 1, 0)},
        checks=checks,
    )


# ---------------------------------------------------------------------------
# 10. Clean numeric: already tidy data must come through unchanged
# ---------------------------------------------------------------------------


def clean_numeric() -> CsvFixture:
    rng = random.Random(10)
    rows = [
        [
            str(i),
            f"{rng.gauss(0, 10):.4f}",
            str(rng.randint(-1000, -1)),
            rng.choice("AB"),
            f"{rng.uniform(1e-6, 1e6):.6e}",
        ]
        for i in range(ROWS)
    ]
    return CsvFixture(
        file_name="clean_numeric.csv",
        description="already clean numeric data: auto_clean must change nothing",
        header=["x", "y", "z", "label", "w"],
        rows=rows,
        raw_columns=["x", "y", "z", "label", "w"],
        clean_kinds={"x": "int", "y": "float", "z": "int", "label": "text", "w": "float"},
        ranges={"z": (None, -1, 0), "x": (0, 99, 0)},
        checks={(0, "x"): 0, (99, "x"): 99, (0, "y"): float(rows[0][1])},
    )


FIXTURES: list[CsvFixture] = [
    retail_orders(),
    employees_eu(),
    sensor_readings(),
    students(),
    products(),
    survey(),
    transactions(),
    server_logs(),
    wide_mixed(),
    clean_numeric(),
]


def write_all(directory: Path = DATA_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fixture in FIXTURES:
        path = directory / fixture.file_name
        path.write_bytes(fixture.to_bytes())
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_all():
        print(f"wrote {written}")
