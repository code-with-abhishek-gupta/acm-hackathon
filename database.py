import sqlite3
import os
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AttendanceDatabase:
    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            # Students table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    enrollment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Attendance events table - raw entry/exit events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY', 'EXIT')),
                    timestamp DATETIME NOT NULL,
                    confidence REAL,
                    source TEXT DEFAULT 'face_recognition',
                    FOREIGN KEY (student_id) REFERENCES students (enrollment_id)
                )
            """)
            
            # Attendance sessions table - daily summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    first_entry DATETIME,
                    last_exit DATETIME,
                    total_duration_minutes INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'INCOMPLETE',
                    FOREIGN KEY (student_id) REFERENCES students (enrollment_id),
                    UNIQUE(student_id, date)
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_student_ts ON attendance_events(student_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON attendance_sessions(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON attendance_events(DATE(timestamp))")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def add_student(self, enrollment_id: str, name: str) -> bool:
        """Add a new student to the database"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO students (enrollment_id, name)
                    VALUES (?, ?)
                """, (enrollment_id, name))
                conn.commit()
                logger.info(f"Added student: {name} ({enrollment_id})")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Student {enrollment_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            return False
    
    def get_all_students(self) -> List[Dict]:
        """Get all registered students"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT enrollment_id, name, created_at FROM students ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_student(self, enrollment_id: str) -> Optional[Dict]:
        """Get a specific student by enrollment ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT enrollment_id, name, created_at 
                FROM students 
                WHERE enrollment_id = ?
            """, (enrollment_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def record_attendance_event(self, student_id: str, event_type: str, confidence: float = None) -> bool:
        """Record an attendance event (ENTRY or EXIT)"""
        try:
            timestamp = datetime.now()
            with self.get_connection() as conn:
                # Insert the event
                conn.execute("""
                    INSERT INTO attendance_events (student_id, event_type, timestamp, confidence)
                    VALUES (?, ?, ?, ?)
                """, (student_id, event_type, timestamp, confidence))
                
                # Update or create session summary
                self._update_session_summary(conn, student_id, timestamp.date())
                
                conn.commit()
                logger.info(f"Recorded {event_type} for student {student_id}")
                return True
        except Exception as e:
            logger.error(f"Error recording attendance event: {e}")
            return False
    
    def _update_session_summary(self, conn: sqlite3.Connection, student_id: str, event_date: date):
        """Update the daily session summary for a student"""
        # Get all events for this student on this date
        cursor = conn.execute("""
            SELECT event_type, timestamp 
            FROM attendance_events 
            WHERE student_id = ? AND DATE(timestamp) = ?
            ORDER BY timestamp
        """, (student_id, event_date))
        
        events = cursor.fetchall()
        
        if not events:
            return
        
        # Calculate session data
        entry_events = [e for e in events if e['event_type'] == 'ENTRY']
        exit_events = [e for e in events if e['event_type'] == 'EXIT']
        
        first_entry = entry_events[0]['timestamp'] if entry_events else None
        last_exit = exit_events[-1]['timestamp'] if exit_events else None
        
        # Calculate total duration (simplified: last_exit - first_entry)
        total_duration_minutes = 0
        status = 'INCOMPLETE'
        
        if first_entry and last_exit:
            first_entry_dt = datetime.fromisoformat(first_entry)
            last_exit_dt = datetime.fromisoformat(last_exit)
            total_duration_minutes = int((last_exit_dt - first_entry_dt).total_seconds() / 60)
            status = 'COMPLETE'
        elif first_entry:
            status = 'IN_PROGRESS'
        
        # Upsert session summary
        conn.execute("""
            INSERT OR REPLACE INTO attendance_sessions 
            (student_id, date, first_entry, last_exit, total_duration_minutes, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, event_date, first_entry, last_exit, total_duration_minutes, status))
    
    def get_student_last_event(self, student_id: str, event_date: date = None) -> Optional[Dict]:
        """Get the last attendance event for a student on a specific date"""
        if event_date is None:
            event_date = date.today()
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT event_type, timestamp, confidence
                FROM attendance_events 
                WHERE student_id = ? AND DATE(timestamp) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (student_id, event_date))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_attendance_summary(self, target_date: date = None) -> List[Dict]:
        """Get attendance summary for a specific date"""
        if target_date is None:
            target_date = date.today()
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    s.enrollment_id,
                    s.name,
                    sess.first_entry,
                    sess.last_exit,
                    sess.total_duration_minutes,
                    sess.status
                FROM students s
                LEFT JOIN attendance_sessions sess ON s.enrollment_id = sess.student_id 
                    AND sess.date = ?
                ORDER BY s.name
            """, (target_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_attendance_events(self, target_date: date = None, student_id: str = None) -> List[Dict]:
        """Get detailed attendance events"""
        if target_date is None:
            target_date = date.today()
        
        query = """
            SELECT 
                ae.student_id,
                s.name,
                ae.event_type,
                ae.timestamp,
                ae.confidence
            FROM attendance_events ae
            JOIN students s ON ae.student_id = s.enrollment_id
            WHERE DATE(ae.timestamp) = ?
        """
        params = [target_date]
        
        if student_id:
            query += " AND ae.student_id = ?"
            params.append(student_id)
        
        query += " ORDER BY ae.timestamp"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get overall system statistics"""
        with self.get_connection() as conn:
            # Total students
            total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            
            # Today's attendance
            today = date.today()
            present_today = conn.execute("""
                SELECT COUNT(*) FROM attendance_sessions 
                WHERE date = ? AND status IN ('COMPLETE', 'IN_PROGRESS')
            """, (today,)).fetchone()[0]
            
            # Total events today
            events_today = conn.execute("""
                SELECT COUNT(*) FROM attendance_events 
                WHERE DATE(timestamp) = ?
            """, (today,)).fetchone()[0]
            
            return {
                "total_students": total_students,
                "present_today": present_today,
                "events_today": events_today,
                "attendance_rate": round(present_today / total_students * 100, 1) if total_students > 0 else 0
            }

    def delete_student(self, enrollment_id: str) -> bool:
        """Delete a student and all related attendance data.
        Because foreign keys were created without ON DELETE CASCADE, we manually remove dependent rows.
        """
        try:
            with self.get_connection() as conn:
                # Ensure student exists
                cur = conn.execute("SELECT 1 FROM students WHERE enrollment_id = ?", (enrollment_id,))
                if cur.fetchone() is None:
                    return False
                # Delete dependent rows
                conn.execute("DELETE FROM attendance_events WHERE student_id = ?", (enrollment_id,))
                conn.execute("DELETE FROM attendance_sessions WHERE student_id = ?", (enrollment_id,))
                # Delete student
                conn.execute("DELETE FROM students WHERE enrollment_id = ?", (enrollment_id,))
                conn.commit()
                logger.info(f"Deleted student {enrollment_id} and related attendance data")
                return True
        except Exception as e:
            logger.error(f"Error deleting student {enrollment_id}: {e}")
            return False
