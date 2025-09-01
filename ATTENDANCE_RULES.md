# Updated Attendance Logic Rules

## New Attendance Timing Rules

### 1. Present Status
- **Time Window**: 15 minutes before class start to 10 minutes after class start
- **Example**: If class starts at 09:00 AM
  - Present window: 08:45 AM to 09:10 AM
  - Students arriving in this window are marked as **PRESENT**

### 2. Absent Status  
- **Time Window**: More than 10 minutes after class start (but before class ends)
- **Example**: If class starts at 09:00 AM and ends at 10:00 AM
  - Absent window: 09:11 AM to 10:00 AM
  - Students arriving in this window are marked as **ABSENT**

### 3. Exit Rules
- **During Class**: Students cannot leave during class hours
  - If a student who is marked PRESENT tries to leave during class, they get **BLOCKED** status
  - Message: "Cannot leave during class hours. Please wait until class ends."
  
- **After Class**: Students can only exit after class officially ends
  - Exit is only allowed for students who were marked **PRESENT**
  - Students marked **ABSENT** don't need exit (already absent)

### 4. Edge Cases

#### Too Early Arrival
- **Before 15-minute window**: Students arriving too early are **BLOCKED**
- **Example**: Arriving before 08:45 AM for a 09:00 AM class
- **Message**: "Too early - Please come back after [time]"

#### Multiple Attempts
- **Already Present**: Students already marked present get **BLOCKED** on subsequent scans
- **Already Absent**: Students already marked absent get **BLOCKED** on subsequent scans

## Status Summary

| Time Period | First Scan | Subsequent Scans | Action |
|-------------|------------|------------------|---------|
| Before -15min | BLOCKED | BLOCKED | Wait |
| -15min to +10min | PRESENT | BLOCKED | Entry recorded |
| +10min to class end | ABSENT | BLOCKED | Marked absent |
| After class end | EXIT (if was present) | BLOCKED | Exit recorded |

## TTS Messages

### Present
- Early: "Good morning [name]! Attendance marked as present. Class starts at [time]."
- On time: "Welcome [name]! Attendance marked as present."

### Absent
- "Unfortunately, you are more than 10 minutes late. You are marked as absent for this class."

### Blocked During Class
- "You cannot leave during class hours. Please wait until the class ends."

### Exit
- "Goodbye [name]! Have a great day. Your exit has been recorded."

## Implementation Details

### Database Changes
- Uses actual status from recognition logic instead of calculating based on grace period
- Supports new status values: 'present', 'absent', 'exit'
- `is_late` field now indicates absence (true for absent status)

### API Changes
- Updated time window calculations using session start time
- Improved status determination logic
- Better TTS message handling for different scenarios

This system ensures strict attendance tracking while preventing students from gaming the system by leaving and re-entering during class hours.
