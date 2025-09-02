import sqlite3

conn = sqlite3.connect('attendance.db')

# Delete attendance records for student 123 in session 7
conn.execute("DELETE FROM attendance_logs WHERE student_id='123' AND session_id=8")
conn.execute("DELETE FROM attendance_summary WHERE student_id='123' AND session_id=8")

conn.commit()
conn.close()

print("Cleared attendance records for student 123 in session 8. Try recognition again.")
