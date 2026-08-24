import pandas as pd
import numpy as np

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

batch_size = len(df) // 12
discount_months = [10, 11, 0, 1]

for i in range(12):
    start = i * batch_size
    end = start + batch_size
    batch = df.iloc[start:end].copy()
    if i in discount_months:
        batch["MonthlyCharges"] = batch["MonthlyCharges"] * 0.95
        batch["Contract"] = np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            size=len(batch),
            p=[0.1, 0.3, 0.6]
        )
    batch.to_csv(f"../batches/batch_{i+1}.csv", index=False)
    print(f"Batch {i+1}: rows {start} to {end}, size {len(batch)}")