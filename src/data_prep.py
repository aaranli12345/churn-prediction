import pandas as pd


def clean(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop("customerID", axis=1, errors="ignore")
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

