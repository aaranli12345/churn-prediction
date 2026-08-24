import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from data_prep import clean, encode_features
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import recall_score, accuracy_score, precision_score


with mlflow.start_run():
    df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean(df)
    df = encode_features(df)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    mlflow.log_params({
        "test_size": 0.2,
        "random_state": 42,
        "max_iter": 1000
    })
    mlflow.sklearn.log_model(model, "model")
    y_pred = model.predict(X_test_scaled)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    mlflow.log_metrics({
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision
    })
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "../Models/model.joblib")
    joblib.dump(scaler, "../Models/scaler.joblib")
    joblib.dump(X_train.columns.tolist(), "../Models/columns.joblib")
