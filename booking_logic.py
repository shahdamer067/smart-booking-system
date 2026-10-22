"""Booking rules and storage (SQLite). No Streamlit or AI code here, so it is easy to test."""
import sqlite3
from datetime import date, datetime, timedelta

SERVICES = [
    "Haircut", "Massage", "Dental Checkup",
    "Doctor Consultation", "Fitness Training", "Beauty Treatment",
]

# Hours (24h clock) bookable on each weekday (Monday = 0 ... Sunday = 6).
# These follow data/booking_docs.txt.
WEEKDAY_SLOTS = [9, 10, 11, 13, 14, 15, 16, 17, 18]
SATURDAY_SLOTS = [10, 11, 13, 14, 15, 16]
MAX_DAYS_AHEAD = 30
DB_PATH = "bookings.db"


def slot_label(hour: int) -> str:
    """13 -> '1:00 PM'"""
    return datetime(2000, 1, 1, hour).strftime("%I:%M %p").lstrip("0")


def slots_for_day(day: date) -> list[int]:
    """Bookable hours for a date. Empty list means closed (Sunday)."""
    if day.weekday() == 6:
        return []
    if day.weekday() == 5:
        return SATURDAY_SLOTS
    return WEEKDAY_SLOTS


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bookings (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               service TEXT NOT NULL,
               day TEXT NOT NULL,
               hour INTEGER NOT NULL,
               created_at TEXT NOT NULL,
               UNIQUE(day, hour)
           )"""
    )
    return conn


def taken_hours(day: date) -> set[int]:
    with _connect() as conn:
        rows = conn.execute("SELECT hour FROM bookings WHERE day = ?", (day.isoformat(),)).fetchall()
    return {r[0] for r in rows}


def validate_booking(name: str, service: str, day: date, hour: int, today: date | None = None) -> list[str]:
    """Return a list of problems. An empty list means the booking is valid."""
    today = today or date.today()
    errors = []
    if not name or not name.strip():
        errors.append("Please enter your name.")
    if service not in SERVICES:
        errors.append("Please choose a valid service.")
    if day < today:
        errors.append("The date is in the past.")
    elif day > today + timedelta(days=MAX_DAYS_AHEAD):
        errors.append(f"You can only book up to {MAX_DAYS_AHEAD} days in advance.")
    if day.weekday() == 6:
        errors.append("We are closed on Sundays.")
    elif hour not in slots_for_day(day):
        errors.append("That time is outside our opening hours.")
    elif hour in taken_hours(day):
        errors.append("That slot is already booked. Please pick another time.")
    return errors


def save_booking(name: str, service: str, day: date, hour: int) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO bookings (name, service, day, hour, created_at) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), service, day.isoformat(), hour, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_bookings() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, service, day, hour FROM bookings ORDER BY day, hour"
        ).fetchall()
    return [
        {"ID": r[0], "Name": r[1], "Service": r[2], "Date": r[3], "Time": slot_label(r[4])}
        for r in rows
    ]


def cancel_booking(booking_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
