#!/usr/bin/env python3
"""
Database reset script to apply schema changes.
This will delete the existing database and recreate it with the new schema (without min_duration).
"""

import os
import sys
from database import AttendanceDatabase

def reset_database():
    """Reset the database by deleting and recreating it"""
    db_path = "attendance.db"
    
    print("Resetting database...")
    
    # Delete existing database if it exists
    if os.path.exists(db_path):
        print(f"Deleting existing database: {db_path}")
        os.remove(db_path)
    
    # Create new database with updated schema
    print("Creating new database with updated schema...")
    db = AttendanceDatabase()
    
    print("Database reset complete!")
    print("The new schema no longer includes the 'min_duration' column in sessions.")
    print("You can now create sessions without min_duration criteria.")

if __name__ == "__main__":
    try:
        reset_database()
    except Exception as e:
        print(f"Error resetting database: {e}")
        sys.exit(1)
