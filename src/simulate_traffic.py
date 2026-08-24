import pandas as pd
import requests
import sqlite3
import datetime

conn = sqlite3.connect("predictions.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT,
        churn_probability REAL,
        actual_churn INTEGER,
        timestamp TEXT
    )
""")
conn.commit()

for i in range(1, 13):
    batch = pd.read_csv(f"../batches/batch_{i}.csv")

    for index, row in batch.iterrows():
        row_for_api = row.drop(["customerID", "Churn"], errors="ignore")
        row_dict = row_for_api.to_dict()
        response = requests.post("http://127.0.0.1:8000/predict", json=row_dict)
        prediction = response.json()

        conn.execute(
            "INSERT INTO predictions (customer_id, churn_probability, actual_churn, timestamp) VALUES (?, ?, ?, ?)",
            (
                row["customerID"],
                prediction.get("churn_probability"),
                1 if row["Churn"] == "Yes" else 0,
                datetime.datetime.now().isoformat()
            )
        )
        conn.commit()
        print(f"Batch {i}, row {index}: {prediction}")