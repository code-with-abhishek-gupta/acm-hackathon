import sqlite3
from datetime import datetime
import pytz

# Get current time the same way the API does
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist)

conn = sqlite3.connect('attendance.db')
cursor = conn.execute("SELECT * FROM sessions WHERE id=8")
session = cursor.fetchone()

print(f"Current time: {current_time}")
print(f"Current time formatted: {current_time.strftime('%H:%M:%S')}")
print(f"Session start: {session[3]}")
print(f"Session end: {session[4]}")
print(f"Grace period: {session[6]} minutes")

# Parse session times like the API does
if len(session[3]) <= 5:  # HH:MM format
    today = current_time.date()
    start_time = datetime.strptime(session[3], '%H:%M').time()
    end_time = datetime.strptime(session[4], '%H:%M').time()
    session_start = datetime.combine(today, start_time).replace(tzinfo=ist)
    session_end = datetime.combine(today, end_time).replace(tzinfo=ist)

print(f"Parsed session_start: {session_start}")
print(f"Parsed session_end: {session_end}")

# Calculate windows
from datetime import timedelta
early_allowance_minutes = 15
grace_period_minutes = int(session[6])

early_window_start = session_start - timedelta(minutes=early_allowance_minutes)
present_cutoff = session_start + timedelta(minutes=grace_period_minutes)

print(f"Early window start: {early_window_start.strftime('%H:%M:%S')}")
print(f"Present cutoff: {present_cutoff.strftime('%H:%M:%S')}")

# Check conditions
print("\n=== Time Window Checks ===")
print(f"current_time < early_window_start: {current_time < early_window_start}")
print(f"early_window_start <= current_time <= present_cutoff: {early_window_start <= current_time <= present_cutoff}")
print(f"present_cutoff < current_time <= session_end: {present_cutoff < current_time <= session_end}")
print(f"current_time > session_end: {current_time > session_end}")

conn.close()
