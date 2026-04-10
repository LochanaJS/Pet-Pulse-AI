import pyodbc
from fastapi import HTTPException

def get_db_connection():
    conn_str = (
        r'Driver={ODBC Driver 17 for SQL Server};' # Ensure this matches your SSMS setup
        r'Server=LOCHANA\SQLEXPRESS;'
        r'Database=PET_PULSE_AIG;'
        r'Trusted_Connection=yes;'
    )
    try:
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")