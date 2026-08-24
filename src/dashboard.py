import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from data_prep import clean
from evidently import Report
from evidently.presets import DataDriftPreset

DB_PATH = Path(__file__).resolve().parent / "predictions.db"

st.write("Using database:", str(DB_PATH))
st.write("Database exists:", DB_PATH.exists())

conn = sqlite3.connect(DB_PATH)

# Show the tables so we can verify the database
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)
st.write("Tables in database:", tables)

df = pd.read_sql("SELECT * FROM predictions", conn)

df["predicted_churn"] = (df["churn_probability"] > 0.5).astype(int)

batch_size = max(1, len(df) // 12)
df["batch"] = (df.index // batch_size) + 1

st.title("Churn Prediction Dashboard")

st.subheader("High-Risk Customers")
high_risk = df[df["churn_probability"] > 0.5]
st.dataframe(high_risk)

accuracy_by_batch = (
    df.assign(
        correct=df["predicted_churn"] == df["actual_churn"]
    )
    .groupby("batch")["correct"]
    .mean()
)

st.subheader("Model Accuracy Over Time")
st.line_chart(accuracy_by_batch)


def check_batch_drift(batch_number, reference, important_columns):
    current = pd.read_csv(f"../batches/batch_{batch_number}.csv")
    current = clean(current)

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=current, reference_data=reference)
    result_dict = my_eval.dict()

    result = {"batch": batch_number}
    for metric in result_dict["metrics"]:
        name = metric.get("metric_name", "")
        for col in important_columns:
            if f"column={col}," in name:
                value = metric["value"]
                threshold = metric["config"].get("threshold", 0.1)
                result[col] = value > threshold

    return result


important_columns = ["Contract", "MonthlyCharges", "tenure"]
reference = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
reference = clean(reference)

drift_results = [check_batch_drift(i, reference, important_columns) for i in range(1, 13)]
drift_df = pd.DataFrame(drift_results)

st.subheader("Drift Warnings by Batch")
st.dataframe(drift_df)

conn.close()