import sqlite3
from datetime import datetime

conn = sqlite3.connect('attendance.db')
conn.row_factory = sqlite3.Row

print("=== Recent Attendance Logs for Student 123 ===")
cursor = conn.execute("""
    SELECT * FROM attendance_logs 
    WHERE student_id='123' 
    ORDER BY timestamp DESC LIMIT 10
""")
for row in cursor.fetchall():
    print(dict(row))

print("\n=== Active Session ===")
cursor = conn.execute("SELECT * FROM sessions WHERE is_active=1")
active_session = cursor.fetchone()
if active_session:
    print(dict(active_session))
else:
    print("No active session")

print("\n=== Session 7 Details ===")
cursor = conn.execute("SELECT * FROM sessions WHERE id=7")
session = cursor.fetchone()
if session:
    print(dict(session))

print("\n=== Attendance Summary for Student 123 ===")
cursor = conn.execute("""
    SELECT * FROM attendance_summary 
    WHERE student_id='123' 
    ORDER BY id DESC LIMIT 5
""")
for row in cursor.fetchall():
    print(dict(row))

conn.close()
