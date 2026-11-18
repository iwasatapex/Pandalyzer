"""Deterministic generator for the 15 end-to-end CSV fixtures in ``tests/data``.

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
    line_terminator: str = "\n"
    # Title/notes lines written above the header (report exports).
    preamble: list[str] = field(default_factory=list)
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
    # Column -> number of malformed emails validate_emails must flag in the raw file.
    invalid_emails: dict[str, int] = field(default_factory=dict)
    expect_parser_warning: bool = False

    def __post_init__(self):
        assert len(self.rows) == ROWS, f"{self.file_name} has {len(self.rows)} rows"

    def to_bytes(self) -> bytes:
        buffer = io.StringIO(newline="")
        for line in self.preamble:
            buffer.write(line + self.line_terminator)
        writer = csv.writer(buffer, delimiter=self.sep, lineterminator=self.line_terminator)
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
                info = zipfile.ZipInfo(
                    self.file_name.removesuffix(".zip") + ".csv", (2024, 1, 1, 0, 0, 0)
                )
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
        raw_columns=[
            "order_id",
            "order_date",
            "customer",
            "quantity",
            "unit_price",
            "paid",
            "channel",
        ],
        clean_rows=96,
        clean_kinds={
            "order_id": "text",
            "order_date": "datetime",
            "customer": "text",
            "quantity": "int",
            "unit_price": "float",
            "paid": "bool",
            "channel": "text",
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
    names = [
        "José Müller",
        "Zoë Dupont",
        "François Lévesque",
        "Søren Æbelø",
        "Ana Peña",
        "Jürgen Groß",
    ]
    departments = ["Vertrieb", "Technik", "Personal", "Einkauf"]
    rows, checks = [], {}
    for i in range(ROWS):
        salary = (
            round(rng.uniform(500, 999), 2) if i % 10 == 0 else round(rng.uniform(2000, 9000), 2)
        )
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
            "Mitarbeiter-ID": "text",
            "Name": "text",
            "Abteilung": "text",
            "Gehalt": "float",
            "Eintritt": "datetime",
            "Remote": "bool",
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
        header=[
            "sensor_id",
            "timestamp",
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "battery_ok",
            "reading_count",
        ],
        rows=rows,
        sep="\t",
        raw_columns=[
            "sensor_id",
            "timestamp",
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "battery_ok",
            "reading_count",
        ],
        clean_kinds={
            "sensor_id": "text",
            "timestamp": "datetime",
            "temperature_c": "float",
            "humidity_pct": "int",
            "pressure_hpa": "float",
            "battery_ok": "bool",
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
            "student_id": "text",
            "full_name": "text",
            "grade": "int",
            "passed": "bool",
            "enrolled_on": "datetime",
            "gpa": "float",
            "club": "text",
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
    names = [
        "Café, Crème Brûlée",
        "Jalapeño Salsa, Hot",
        "Pâté de Campagne",
        "Smörgåsbord Tray",
        "Piña Colada Mix",
    ]
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
        raw_columns=[
            "sku",
            "product_name",
            "description",
            "price",
            "discount",
            "in_stock",
            "category",
        ],
        clean_rows=99,
        clean_kinds={
            "sku": "text",
            "product_name": "text",
            "description": "text",
            "price": "float",
            "discount": "int",
            "in_stock": "bool",
            "category": "text",
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
        month, day, hour, minute = (
            rng.randint(1, 12),
            rng.randint(1, 28),
            rng.randint(0, 23),
            rng.randint(0, 59),
        )
        recommend = rng.choice(["Yes", "No", "yes", "NO"])
        rows.append(
            [
                f"R{i:03d}",
                rng.choice(["N/A", "-", "?", "", " "]) if i % 10 == 1 else str(age),
                str(rng.randint(1, 5)),
                recommend,
                rng.choice(["Great service", "Too slow, honestly", "n/a", "Loved it!"])
                if i % 3 == 0
                else "",
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
        description="gzip, messy headers renamed to snake_case, N/A/-/? placeholders, "
        "month-name dates",
        header=[
            "Respondent ID",
            "Age",
            "Satisfaction (1-5)",
            "Would Recommend?",
            "Comments",
            "Submitted At",
        ],
        rows=rows,
        container="gzip",
        clean_kwargs={"column_names": "snake"},
        raw_columns=[
            "Respondent ID",
            "Age",
            "Satisfaction (1-5)",
            "Would Recommend?",
            "Comments",
            "Submitted At",
        ],
        clean_kinds={
            "respondent_id": "text",
            "age": "int",
            "satisfaction_1_5": "int",
            "would_recommend": "bool",
            "comments": "text",
            "submitted_at": "datetime",
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
        raw_columns=[
            "txn_id",
            "amount",
            "fee_pct",
            "created_at",
            "merchant",
            "is_refund",
            "currency",
        ],
        clean_rows=98,
        clean_kinds={
            "txn_id": "text",
            "amount": "float",
            "fee_pct": "float",
            "created_at": "datetime",
            "merchant": "text",
            "is_refund": "bool",
            "currency": "text",
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
        description="trailing delimiter, ragged rows with extra fields, empty rows, "
        "IPs and versions",
        header=["timestamp", "client_ip", "app_version", "status", "duration_ms", "path", ""],
        rows=rows,
        raw_columns=[
            "timestamp",
            "client_ip",
            "app_version",
            "status",
            "duration_ms",
            "path",
            "Unnamed: 6",
        ],
        clean_rows=98,
        clean_kinds={
            "timestamp": "datetime",
            "client_ip": "text",
            "app_version": "text",
            "status": "int",
            "duration_ms": "float",
            "path": "text",
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
        description="UTF-8 BOM, duplicate header, 0/1 flags, constant and all-empty columns, "
        "JSON text",
        header=[
            "id",
            "value",
            "value",
            "flag01",
            "constant",
            "all_empty",
            "json_tags",
            "event_date",
            "score",
            "level",
            "notes",
            "big_count",
            "ratio",
        ],
        rows=rows,
        encoding="utf-8-sig",
        raw_columns=[
            "id",
            "value",
            "value.1",
            "flag01",
            "constant",
            "all_empty",
            "json_tags",
            "event_date",
            "score",
            "level",
            "notes",
            "big_count",
            "ratio",
        ],
        clean_kinds={
            "id": "int",
            "value": "int",
            "value.1": "float",
            "flag01": "int",
            "constant": "text",
            "all_empty": "float",
            "json_tags": "text",
            "event_date": "datetime",
            "score": "float",
            "level": "text",
            "notes": "text",
            "big_count": "int",
            "ratio": "float",
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


# ---------------------------------------------------------------------------
# 11. CRM contacts: messy emails, names with repeated spaces, phones, zip codes
# ---------------------------------------------------------------------------


def crm_contacts() -> CsvFixture:
    rng = random.Random(11)
    firsts = ["Ann", "Ben", "Chloe", "Dev", "Ines", "Kofi", "Mei", "Omar"]
    lasts = ["Lee", "Okafor", "Garcia", "Patel", "Novak", "Silva"]
    phones = ["+1 (555) {a}-{b}", "555-{a}-{b}", "(555) {a} {b}", "555.{a}.{b}"]
    countries = ["USA", "Canada", "Mexico"]
    rows, checks = [], {}
    for i in range(ROWS):
        first, last = rng.choice(firsts), rng.choice(lasts)
        name = f"{first} {last}"
        address = f"{first}.{last}{i}@example.com".lower()
        email = [
            address.upper(),
            f"mailto:{address}",
            f"{name} <{address}>",
            f" {address} ",
            address,
            address,
        ][i % 6]
        day, month = rng.randint(1, 28), rng.randint(1, 12)
        country = rng.choice(countries)
        opted_in = rng.choice(["on", "off"])
        age = rng.randint(18, 80)
        rows.append(
            [
                f"C{i:04d}",
                f"{first}   {last}" if i % 7 == 0 else f"{first}\t{last}" if i % 11 == 0 else name,
                email,
                rng.choice(phones).format(a=rng.randint(100, 999), b=rng.randint(1000, 9999)),
                f"{rng.randint(1000, 99999):05d}",
                country.lower() if i % 5 == 0 else country,
                opted_in.upper() if i % 8 == 0 else opted_in,
                f"2023-{month:02d}-{day:02d}",
                str(age),
            ]
        )
        if i in (1, 2, 3, 4):
            checks[(i, "email")] = address
            checks[(i, "opted_in")] = opted_in == "on"
            checks[(i, "signup_date")] = pd.Timestamp(2023, month, day)
            checks[(i, "age")] = age
    rows[0][4] = "02139"  # Leading-zero zip codes must stay text.
    malformed = ["@example.com", "carol@", "DAN@@EXAMPLE.COM", "not an email", "eve@example"]
    for offset, bad in enumerate(malformed):
        rows[10 + offset][2] = bad
        checks[(10 + offset, "email")] = None
    rows[20][2], rows[21][2] = "\u2014", "(blank)"  # Placeholders, not malformed addresses.
    rows[30][7], rows[31][8] = "<NA>", "\u2013"  # En dash.
    rows[40][2] = "\u200b" + rows[40][2].strip()  # Zero-width space pasted from the web.
    rows[50][8], rows[51][8] = "150", "-1"
    rows[90] = list(rows[1])  # Same contact, email in capitals and name double-spaced.
    rows[90][1], rows[90][2] = rows[1][1].replace(" ", "  "), checks[(1, "email")].upper()
    rows[91] = list(rows[2])
    rows[91][2] = checks[(2, "email")].upper()
    rows[95] = [""] * 9
    checks[(0, "zip_code")] = "02139"
    checks[(7, "full_name")] = " ".join(rows[7][1].split())
    checks[(11, "full_name")] = " ".join(rows[11][1].split())
    checks[(20, "email")] = None
    checks[(40, "email")] = rows[40][2].lstrip("\u200b").lower()
    checks[(30, "signup_date")] = None
    columns = [
        "contact_id",
        "full_name",
        "email",
        "phone",
        "zip_code",
        "country",
        "opted_in",
        "signup_date",
        "age",
    ]
    return CsvFixture(
        file_name="crm_contacts.csv",
        description="emails in mixed case, mailto:, display names and malformed; "
        "repeated spaces; phone formats; on/off flags; dashes as placeholders",
        header=columns,
        rows=rows,
        load_kwargs={"dtype": {"zip_code": str}},
        raw_columns=columns,
        clean_rows=97,
        clean_kinds={
            "contact_id": "text",
            "full_name": "text",
            "email": "text",
            "phone": "text",
            "zip_code": "text",
            "country": "text",
            "opted_in": "bool",
            "signup_date": "datetime",
            "age": "int",
        },
        ranges={"age": (0, 120, 2)},
        checks=checks,
        dropped={90, 91, 95},
        # The 5 malformed addresses plus the 2 placeholders, which are not yet missing.
        invalid_emails={"email": 7},
    )


# ---------------------------------------------------------------------------
# 12. Bank statement: report title above the header, currency codes, totals
# ---------------------------------------------------------------------------


def bank_statement() -> CsvFixture:
    rng = random.Random(12)
    merchants = [
        "CARD PAYMENT    TESCO",
        "TRANSFER  FROM SAVINGS",
        "DIRECT DEBIT   ENERGY CO",
        "ATM",
    ]
    rows, checks, balances = [], {}, []
    balance = 2500.0
    for i in range(97):
        amount = round(rng.uniform(-2500, 2500), 2)
        balance = round(balance + amount, 2)
        balances.append(balance)
        day, month = rng.randint(1, 28), rng.randint(1, 3)
        code = rng.choice(["{sign}USD {value}", "{sign}{value} USD"])
        rows.append(
            [
                f"{day:02d} {MONTHS[month - 1]} 2024",
                rng.choice(merchants),
                f"TX{rng.randint(0, 99_999_999):08d}",
                "CR" if amount >= 0 else "DR",
                code.format(sign="-" if amount < 0 else "", value=f"{abs(amount):,.2f}"),
                f"{balance:,.2f}".replace(",", " "),
            ]
        )
        if i in (0, 3, 5):
            checks[(i, "Date")] = pd.Timestamp(2024, month, day)
            checks[(i, "Amount")] = amount
            checks[(i, "Balance")] = balance
            checks[(i, "Description")] = " ".join(rows[i][1].split())
    rows[50] = list(rows[49])  # The same transaction exported twice.
    rows[60][5], balances[60] = "-1 250.00", -1250.0
    negative = sum(value < 0 for row, value in enumerate(balances) if row != 50)
    rows.append([""] * 6)
    rows.append(["Total debits", "", "", "", "-USD 12,345.67", ""])
    rows.append(["Total credits", "", "", "", "USD 23,456.78", ""])
    checks[(60, "Balance")] = -1250.0
    columns = ["Date", "Description", "Reference", "Debit/Credit", "Amount", "Balance"]
    return CsvFixture(
        file_name="bank_statement.csv",
        description="report title and notes above the header, USD codes before and after, "
        "space thousands separators, Total rows at the end",
        header=columns,
        rows=rows,
        preamble=["Northwind Bank - Account statement", "Generated: Apr 2, 2024", ""],
        raw_columns=columns,
        clean_rows=96,
        clean_kinds={
            "Date": "datetime",
            "Description": "text",
            "Reference": "text",
            "Debit/Credit": "text",
            "Amount": "float",
            "Balance": "float",
        },
        ranges={"Balance": (0, None, negative)},
        checks=checks,
        dropped={50, 97, 98, 99},
    )


# ---------------------------------------------------------------------------
# 13. Merged exports: three files pasted together, header repeated mid-file
# ---------------------------------------------------------------------------


def merged_exports() -> CsvFixture:
    rng = random.Random(13)
    header = ["Region", "Store ID", "Month", "Units", "Revenue", "Returned"]
    rows, checks = [], {}
    for i in range(ROWS):
        month = 1 + i // 34
        units = rng.randint(0, 400)
        revenue = round(units * rng.uniform(2, 30), 2)
        returned = rng.choice(["Y", "N"])
        rows.append(
            [
                rng.choice(["North", "South", "East", "West"]),
                str(rng.randint(100, 120)),
                f"2024-{month:02d}-01",
                str(units),
                f"{revenue:.2f}",
                returned,
            ]
        )
        if i in (1, 2, 40):
            checks[(i, "Units")] = units
            checks[(i, "Revenue")] = revenue
            checks[(i, "Returned")] = returned == "Y"
            checks[(i, "Month")] = pd.Timestamp(2024, month, 1)
    rows[33] = list(header)
    rows[66] = [" REGION ", "store id", "Month", "UNITS", "Revenue", "returned"]
    rows[70] = list(rows[20])  # Overlap between two exports.
    rows[5][3], rows[6][3] = "(blank)", "<NA>"
    rows[7][4], rows[8][4] = "\u2014", "\u2013"  # Em and en dashes.
    rows[9][5] = "(null)"
    for row, column in ((5, "Units"), (6, "Units"), (7, "Revenue"), (8, "Revenue")):
        checks[(row, column)] = None
    checks[(9, "Returned")] = pd.NA
    return CsvFixture(
        file_name="merged_exports.csv",
        description="three exports concatenated: header repeated mid-file (once re-cased), "
        "overlapping rows, (blank)/<NA>/dash placeholders",
        header=header,
        rows=rows,
        raw_columns=header,
        clean_rows=97,
        clean_kinds={
            "Region": "text",
            "Store ID": "int",
            "Month": "datetime",
            "Units": "int",
            "Revenue": "float",
            "Returned": "bool",
        },
        ranges={"Units": (0, 500, 0), "Store ID": (100, 120, 0)},
        checks=checks,
        dropped={33, 66, 70},
    )


# ---------------------------------------------------------------------------
# 14. International prices: Swiss 1'234.50, French 1 234,56 €, CHF codes
# ---------------------------------------------------------------------------


def intl_prices() -> CsvFixture:
    rng = random.Random(14)
    products = ["Käse\u00a0Fondue", "Schoggi  Tafel", "Montre\u00a0Classique", "Café crème"]
    rows, checks = [], {}
    for i in range(ROWS):
        chf = round(rng.uniform(5, 15000), 2)
        eur = round(rng.uniform(5, 15000), 2)
        discount = rng.choice([0, 5, 7.5, 10, 12.5, 15])
        weight = round(rng.uniform(0.05, 25), 3)
        swiss = f"{chf:,.2f}".replace(",", "'")
        french = f"{eur:,.2f}".replace(",", "\u202f").replace(".", ",")
        rows.append(
            [
                f"CH-{i:04d}",
                rng.choice(products) + ("\u200b" if i % 9 == 0 else ""),
                f"CHF {swiss}" if i % 2 else swiss,
                f"{french} €",
                f"{discount} %",
                f"{weight}",
                rng.choice(["TRUE", "FALSE"]),
            ]
        )
        if i in (0, 3, 8):
            checks[(i, "price_chf")] = chf
            checks[(i, "price_eur")] = eur
            checks[(i, "discount")] = float(discount)
            checks[(i, "weight_kg")] = weight
            checks[(i, "product")] = " ".join(rows[i][1].replace("\u200b", "").split())
    rows[80] = list(rows[8])  # Same item, price written with the CHF code this time.
    rows[80][2] = "CHF " + rows[8][2]
    rows[40][4] = "120 %"
    checks[(40, "discount")] = 120.0
    columns = ["sku", "product", "price_chf", "price_eur", "discount", "weight_kg", "in_stock"]
    return CsvFixture(
        file_name="intl_prices.csv",
        description="semicolon; Swiss apostrophe and French narrow-space thousands, "
        "CHF/€ before or after, NBSP and zero-width characters in names",
        header=columns,
        rows=rows,
        sep=";",
        raw_columns=columns,
        clean_rows=99,
        clean_kinds={
            "sku": "text",
            "product": "text",
            "price_chf": "float",
            "price_eur": "float",
            "discount": "float",
            "weight_kg": "float",
            "in_stock": "bool",
        },
        ranges={"discount": (0, 100, 1)},
        checks=checks,
        dropped={80},
    )


# ---------------------------------------------------------------------------
# 15. Excel export: ="00123" codes, CRLF, line break in a header, #VALUE!
# ---------------------------------------------------------------------------


def excel_export() -> CsvFixture:
    rng = random.Random(15)
    names = ["Ana Costa", "Bo Nilsson", "Cem Yilmaz", "Dara Kim"]
    notes = ["", "", "", "late shift", "trainee", "#N/A"]
    rows, checks = [], {}
    for i in range(99):
        hours = rng.choice([4, 6.25, 7.5, 8, 9.75])
        month, day = rng.randint(1, 12), rng.randint(13, 28)
        rate = round(rng.uniform(15, 60), 2)
        overtime = rng.choice(["TRUE", "FALSE"])
        rows.append(
            [
                f'="{i + 1:05d}"',
                rng.choice(names) + ("\u00a0" if i % 4 == 0 else ""),
                ("\u200b" if i % 10 == 0 else "") + str(hours),
                overtime,
                f"{month}/{day}/2024",
                f"${rate:.2f}",
                rng.choice(notes),
            ]
        )
        if i in (0, 1, 4):
            checks[(i, "Employee Code")] = f"{i + 1:05d}"
            checks[(i, "Hours Worked")] = float(hours)
            checks[(i, "Overtime")] = overtime == "TRUE"
            checks[(i, "Start Date")] = pd.Timestamp(2024, month, day)
            checks[(i, "Rate")] = rate
    rows[12][2] = "#VALUE!"
    rows[30][2] = "30"
    rows[60] = list(rows[6])
    rows.append([""] * 7)  # Excel's trailing ",,,,,," row.
    checks[(12, "Hours Worked")] = None
    checks[(0, "Name")] = rows[0][1].strip()
    raw = ["Employee Code", "Name", "Hours\r\nWorked", "Overtime", "Start Date", "Rate", "Notes"]
    return CsvFixture(
        file_name="excel_export.csv",
        description='Excel: BOM, CRLF, ="00123" codes, line break inside a header, '
        "#VALUE!/#N/A, NBSP and zero-width characters, $ rates",
        header=raw,
        rows=rows,
        encoding="utf-8-sig",
        line_terminator="\r\n",
        load_kwargs={"dtype": {"Employee Code": str}},
        raw_columns=raw,
        clean_rows=98,
        clean_kinds={
            "Employee Code": "text",
            "Name": "text",
            "Hours Worked": "float",
            "Overtime": "bool",
            "Start Date": "datetime",
            "Rate": "float",
            "Notes": "text",
        },
        ranges={"Hours Worked": (0, 24, 1)},
        checks=checks,
        dropped={60, 99},
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
    crm_contacts(),
    bank_statement(),
    merged_exports(),
    intl_prices(),
    excel_export(),
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
