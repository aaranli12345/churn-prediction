import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from data_prep import clean, encode_features



model = joblib.load("../Models/model.joblib")
scaler = joblib.load("../Models/scaler.joblib")
model_columns = joblib.load("../Models/columns.joblib")



app = FastAPI()

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    df = clean(df)
    df = encode_features(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    X_scaled = scaler.transform(df)
    prediction = model.predict_proba(X_scaled)
    churn_probability = prediction[0][1]
    return {"churn_probability": churn_probability}