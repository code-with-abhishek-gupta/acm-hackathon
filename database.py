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
            
            # Sessions table - for class/lab management
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('class', 'lab')),
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    room TEXT,
                    grace_period INTEGER DEFAULT 15,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0
                )
            """)
            
            # Attendance logs table - detailed entry/exit tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    session_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('ENTRY', 'EXIT')),
                    status TEXT DEFAULT 'PRESENT' CHECK(status IN ('EARLY', 'PRESENT', 'LATE', 'EXIT', 'BLOCKED', 'ABSENT')),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    message TEXT,
                    FOREIGN KEY (student_id) REFERENCES students (enrollment_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Attendance summary table - session-wise summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    session_id INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('present', 'absent', 'late', 'left_early')),
                    entry_time DATETIME,
                    exit_time DATETIME,
                    total_duration INTEGER DEFAULT 0,
                    is_late BOOLEAN DEFAULT 0,
                    FOREIGN KEY (student_id) REFERENCES students (enrollment_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    UNIQUE(student_id, session_id)
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_student_ts ON attendance_events(student_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON attendance_sessions(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON attendance_events(DATE(timestamp))")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON attendance_logs(session_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_session ON attendance_summary(session_id)")
            
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
    
    # Session Management Methods
    def create_session(self, name: str, session_type: str, start_time: str, end_time: str,
                      room: str = "", grace_period: int = 15) -> int:
        """Create a new session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO sessions (name, type, start_time, end_time, room, grace_period)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, session_type, start_time, end_time, room, grace_period))
                conn.commit()
                logger.info(f"Created session: {name} ({session_type})")
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def activate_session(self, session_id: int) -> bool:
        """Activate a session and deactivate all others"""
        try:
            with self.get_connection() as conn:
                # Deactivate all sessions
                conn.execute("UPDATE sessions SET is_active = 0")
                # Activate selected session
                conn.execute("UPDATE sessions SET is_active = 1 WHERE id = ?", (session_id,))
                conn.commit()
                logger.info(f"Activated session ID: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error activating session: {e}")
            return False
    
    def get_next_session(self) -> Optional[Dict]:
        """Get the next upcoming session"""
        from datetime import datetime
        
        current_time = datetime.now().strftime('%H:%M')
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM sessions 
                WHERE time(start_time) > time(?)
                ORDER BY start_time LIMIT 1
            """, (current_time,))
            
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and related attendance data"""
        try:
            with self.get_connection() as conn:
                # Delete related attendance data first
                conn.execute("DELETE FROM attendance_logs WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM attendance_summary WHERE session_id = ?", (session_id,))
                # Delete the session
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                logger.info(f"Deleted session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def update_session(self, session_id: int, name: str, session_type: str, 
                      start_time: str, end_time: str, room: str = "", 
                      grace_period: int = 15) -> bool:
        """Update an existing session"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE sessions SET 
                    name = ?, type = ?, start_time = ?, end_time = ?, 
                    room = ?, grace_period = ?
                    WHERE id = ?
                """, (name, session_type, start_time, end_time, room, grace_period, session_id))
                conn.commit()
                logger.info(f"Updated session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False

    def get_active_session(self) -> Optional[Dict]:
        """Get the currently active session - auto-activates based on time"""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        
        try:
            # Use IST timezone
            ist = ZoneInfo("Asia/Kolkata")
            current_time = datetime.now(ist)
        except ImportError:
            # Fallback for older python, though less accurate without timezone library
            from datetime import timedelta
            current_time = datetime.now() + timedelta(hours=5, minutes=30)

        current_date_str = current_time.strftime('%Y-%m-%d')
        current_time_str = current_time.strftime('%H:%M:%S')
        
        with self.get_connection() as conn:
            # First check if there's a manually active session
            cursor = conn.execute("SELECT * FROM sessions WHERE is_active = 1")
            active = cursor.fetchone()
            if active:
                # Determine if active session has expired; if yes, auto-deactivate
                try:
                    raw_start = active['start_time']
                    raw_end = active['end_time']
                    # Handle time-only vs full datetime (YYYY-MM-DD ...)
                    if len(raw_start) <= 8:  # time-only HH:MM or HH:MM:SS
                        # Build datetime for today (assumes same-day session)
                        from datetime import datetime, time as dtime
                        start_parts = list(map(int, raw_start.split(':')))
                        end_parts = list(map(int, raw_end.split(':')))
                        # Pad parts
                        while len(start_parts) < 3:
                            start_parts.append(0)
                        while len(end_parts) < 3:
                            end_parts.append(0)
                        start_dt = datetime(current_time.year, current_time.month, current_time.day,
                                             start_parts[0], start_parts[1], start_parts[2], tzinfo=current_time.tzinfo)
                        end_dt = datetime(current_time.year, current_time.month, current_time.day,
                                           end_parts[0], end_parts[1], end_parts[2], tzinfo=current_time.tzinfo)
                        # Support overnight sessions (end before start -> add a day)
                        if end_dt <= start_dt:
                            from datetime import timedelta
                            end_dt += timedelta(days=1)
                    else:
                        from datetime import datetime
                        # Expect ISO-like string; fallback to naive parse
                        try:
                            start_dt = datetime.fromisoformat(raw_start)
                            end_dt = datetime.fromisoformat(raw_end)
                            # If parsed naive, attach IST tz
                            if start_dt.tzinfo is None:
                                start_dt = start_dt.replace(tzinfo=current_time.tzinfo)
                            if end_dt.tzinfo is None:
                                end_dt = end_dt.replace(tzinfo=current_time.tzinfo)
                        except Exception:
                            # Fallback: treat as time-only if parsing failed
                            from datetime import datetime, timedelta
                            start_dt = datetime.strptime(raw_start[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=current_time.tzinfo)
                            end_dt = datetime.strptime(raw_end[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=current_time.tzinfo)
                    # If session ended, deactivate and continue to auto-activation logic
                    if current_time > end_dt:
                        conn.execute("UPDATE sessions SET is_active = 0 WHERE id = ?", (active['id'],))
                        conn.commit()
                        logger.info(f"Auto-deactivated expired session {active['id']} ({active['name']})")
                    else:
                        return dict(active)
                except Exception as e:
                    logger.warning(f"Failed to evaluate active session expiry: {e}; returning active session as-is")
                    return dict(active)
            
            # Auto-activate session based on time (15 min early allowance)
            # For sessions scheduled for today
            cursor = conn.execute("""
                SELECT * FROM sessions 
                WHERE (
                    -- For full datetime format (check if it's today's session)
                    (start_time LIKE ? AND 
                     TIME(datetime(start_time, '-15 minutes')) <= TIME(?) AND 
                     TIME(end_time) >= TIME(?))
                    OR
                    -- For time-only format (treat as today)
                    (start_time NOT LIKE '%-%-%T%:%' AND 
                     LENGTH(start_time) <= 8 AND
                     TIME(start_time, '-15 minutes') <= TIME(?) AND 
                     TIME(end_time) >= TIME(?))
                )
                ORDER BY start_time LIMIT 1
            """, (f"{current_date_str}%", current_time_str, current_time_str, current_time_str, current_time_str))
            
            session = cursor.fetchone()
            if session:
                # Auto-activate this session
                conn.execute("UPDATE sessions SET is_active = 0")  # Deactivate all
                conn.execute("UPDATE sessions SET is_active = 1 WHERE id = ?", (session['id'],))
                conn.commit()
                logger.info(f"Auto-activated session {session['id']} ({session['name']}) based on time")
                # Refresh the session data to get the updated is_active status
                cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session['id'],))
                updated_session = cursor.fetchone()
                return dict(updated_session)
            
            return None
    
    def log_attendance_action(self, student_id: str, session_id: int, action: str, 
                             confidence: float = None, timestamp: datetime = None, 
                             status: str = 'PRESENT', message: str = None) -> bool:
        """Log an attendance action (ENTRY or EXIT) with status and message"""
        try:
            with self.get_connection() as conn:
                if timestamp:
                    conn.execute("""
                        INSERT INTO attendance_logs (student_id, session_id, action, status, confidence, timestamp, message)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (student_id, session_id, action, status, confidence, timestamp.isoformat(), message))
                else:
                    conn.execute("""
                        INSERT INTO attendance_logs (student_id, session_id, action, status, confidence, message)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (student_id, session_id, action, status, confidence, message))
                conn.commit()
                logger.info(f"Logged {action} ({status}) for student {student_id} in session {session_id}: {message}")
                return True
        except Exception as e:
            logger.error(f"Error logging attendance action: {e}")
            return False
    
    def update_attendance_summary(self, student_id: str, session_id: int, action: str, 
                                 timestamp: datetime, session_data: Dict, actual_status: str = None) -> bool:
        """Update attendance summary for a student in a session.

        Rules:
        - Create summary only on first valid ENTRY (PRESENT / ABSENT / LATE)
        - BLOCKED entries are never written to summary
        - If an initial summary was ABSENT/LATE and later (still within present window) we detect PRESENT, upgrade status
        - EXIT only stamps exit_time & duration; doesn't downgrade status
        """
        try:
            # Normalize statuses to conform to summary table allowed values
            # Business spec only uses PRESENT / ABSENT (plus EXIT handling). Any EARLY -> PRESENT, LATE -> ABSENT.
            if actual_status:
                upper = actual_status.upper()
                if upper == 'EARLY':
                    actual_status = 'present'
                elif upper == 'LATE':
                    actual_status = 'absent'
                elif upper == 'PRESENT':
                    actual_status = 'present'
                elif upper == 'ABSENT':
                    actual_status = 'absent'
                elif upper == 'EXIT':
                    actual_status = 'present'  # EXIT marked as present with exit_time
            with self.get_connection() as conn:
                # Check if summary exists
                existing = conn.execute("""
                    SELECT * FROM attendance_summary 
                    WHERE student_id = ? AND session_id = ?
                """, (student_id, session_id)).fetchone()
                
                # Parse session start time - handle both full datetime and time-only formats
                start_time_str = session_data['start_time']
                if 'T' in start_time_str and len(start_time_str) > 10:
                    # Full datetime format
                    session_start = datetime.fromisoformat(start_time_str)
                else:
                    # Time-only format - use today's date with the session time
                    today = timestamp.date()
                    time_part = datetime.strptime(start_time_str, '%H:%M').time()
                    session_start = datetime.combine(today, time_part)
                
                grace_period = session_data['grace_period']
                
                if not existing:
                    # Create a new summary record only on a valid entry action
                    if action == 'ENTRY' and actual_status and actual_status.upper() != 'BLOCKED':
                        final_status = actual_status.lower()
                        is_late = final_status in ['absent', 'late']
                        
                        conn.execute("""
                            INSERT INTO attendance_summary 
                            (student_id, session_id, status, entry_time, is_late)
                            VALUES (?, ?, ?, ?, ?)
                        """, (student_id, session_id, final_status, timestamp.isoformat(), is_late))
                        logger.info(f"Created attendance summary for student {student_id} in session {session_id} with status {final_status}")
                    elif action == 'EXIT' and actual_status and actual_status.upper() == 'EXIT':
                        # Handle direct EXIT without prior entry (unusual case)
                        conn.execute("""
                            INSERT INTO attendance_summary 
                            (student_id, session_id, status, exit_time, total_duration, is_late)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (student_id, session_id, 'present', timestamp.isoformat(), 0, False))
                        logger.info(f"Created attendance summary with exit for student {student_id} in session {session_id}")
                
                else:
                    # An existing summary record needs updating
                    if action == 'ENTRY' and actual_status and actual_status.upper() != 'BLOCKED':
                        # Logic to upgrade status: if was 'absent' but now is 'present'
                        current_status = existing['status']
                        new_status = actual_status.lower()
                        
                        final_status = current_status
                        if new_status == 'present' and current_status in ['absent', 'late']:
                            final_status = 'present'
                        
                        is_late = final_status in ['absent', 'late']

                        # Update entry time if it's not set, or if the new entry is earlier
                        update_entry_time = not existing['entry_time'] or timestamp < datetime.fromisoformat(existing['entry_time'])
                        
                        if update_entry_time:
                            conn.execute("""
                                UPDATE attendance_summary 
                                SET entry_time = ?, status = ?, is_late = ?
                                WHERE id = ?
                            """, (timestamp.isoformat(), final_status, is_late, existing['id']))
                            logger.info(f"Updated entry time and status for student {student_id} to {final_status}")
                        else:
                            # Only update status if it's an upgrade
                            conn.execute("""
                                UPDATE attendance_summary SET status = ?, is_late = ? WHERE id = ?
                            """, (final_status, is_late, existing['id']))
                            logger.info(f"Upgraded status for student {student_id} to {final_status}")

                    elif action == 'EXIT':
                        # Only update exit time if an entry exists
                        if existing['entry_time']:
                            entry_time = datetime.fromisoformat(existing['entry_time'])
                            duration = (timestamp - entry_time).total_seconds() / 60
                            
                            # Keep existing status on exit
                            status = existing['status']
                            
                            conn.execute("""
                                UPDATE attendance_summary 
                                SET exit_time = ?, total_duration = ?, status = ?
                                WHERE id = ?
                            """, (timestamp.isoformat(), int(duration), status, existing['id']))
                            logger.info(f"Updated exit for student {student_id} in session {session_id}")
                        else:
                            # Handle case where exit is recorded without entry time
                            conn.execute("""
                                UPDATE attendance_summary 
                                SET exit_time = ?, total_duration = 0
                                WHERE id = ?
                            """, (timestamp.isoformat(), existing['id']))
                            logger.info(f"Updated exit time without entry for student {student_id} in session {session_id}")
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating attendance summary: {e}")
            return False
    
    def get_session_attendance(self, session_id: int) -> List[Dict]:
        """Get attendance summary for a session, including all registered students."""
        with self.get_connection() as conn:
            # Get all students first
            all_students = conn.execute("SELECT enrollment_id, name FROM students ORDER BY name").fetchall()
            
            # Get attendance data for this session
            attendance_data = conn.execute("""
                SELECT 
                    student_id,
                    status,
                    entry_time,
                    exit_time,
                    total_duration,
                    is_late
                FROM attendance_summary
                WHERE session_id = ?
            """, (session_id,)).fetchall()
            
            # Create a dictionary for quick lookup of attendance
            attendance_map = {row['student_id']: dict(row) for row in attendance_data}
            
            # Combine the lists
            full_attendance_list = []
            for student in all_students:
                student_id = student['enrollment_id']
                student_name = student['name']
                
                if student_id in attendance_map:
                    # Student has an attendance record for this session
                    record = attendance_map[student_id]
                    full_attendance_list.append({
                        'student_id': student_id,
                        'student_name': student_name,
                        'status': record['status'],
                        'entry_time': record['entry_time'],
                        'exit_time': record['exit_time'],
                        'total_duration': record['total_duration'],
                        'is_late': record['is_late'],
                    })
                else:
                    # Student is registered but has no attendance record for this session (i.e., absent)
                    full_attendance_list.append({
                        'student_id': student_id,
                        'student_name': student_name,
                        'status': 'absent',
                        'entry_time': None,
                        'exit_time': None,
                        'total_duration': 0,
                        'is_late': False,
                    })
            
            return full_attendance_list
    
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
                
                # Delete dependent rows from all tables with foreign key constraints
                # First delete from attendance_logs
                conn.execute("DELETE FROM attendance_logs WHERE student_id = ?", (enrollment_id,))
                
                # Delete from attendance_summary
                conn.execute("DELETE FROM attendance_summary WHERE student_id = ?", (enrollment_id,))
                
                # Delete from attendance_sessions
                conn.execute("DELETE FROM attendance_sessions WHERE student_id = ?", (enrollment_id,))
                
                # Delete from attendance_events
                conn.execute("DELETE FROM attendance_events WHERE student_id = ?", (enrollment_id,))
                
                # Finally delete the student record
                conn.execute("DELETE FROM students WHERE enrollment_id = ?", (enrollment_id,))
                conn.commit()
                logger.info(f"Deleted student {enrollment_id} and related attendance data")
                return True
        except Exception as e:
            logger.error(f"Error deleting student {enrollment_id}: {e}")
            return False
