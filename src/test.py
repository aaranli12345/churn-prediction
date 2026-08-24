import sqlite3
conn = sqlite3.connect("predictions.db")
total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
distinct = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM predictions").fetchone()
print("Total:", total, "Distinct:", distinct)