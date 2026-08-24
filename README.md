# Customer Churn Prediction — End-to-End MLOps Pipeline

A complete machine learning system that predicts customer churn for a telecom company, built from raw data exploration through to a live, deployed, monitored API. This project was built as a hands-on learning exercise covering the full lifecycle of a real ML system: data cleaning, model training, experiment tracking, API serving, containerization, simulated production traffic, drift detection, and monitoring.

**Live demo:** https://churn-prediction-ehhx.onrender.com/docs
*(Note: hosted on Render's free tier — the service spins down after 15 minutes of inactivity, so the first request after a period of idle time may take 30–50 seconds to respond while it wakes up.)*

---

## What it does

Given a telecom customer's account details (contract type, monthly charges, tenure, services subscribed to, etc.), the API returns a probability that the customer will churn (cancel their service). This kind of model lets a business proactively identify at-risk customers and intervene before they leave.

The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, ~20 features).

---

## Architecture

```
                        ┌─────────────────┐
                        │   Raw CSV data   │
                        └────────┬─────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   data_prep.py          │
                     │   clean() +             │
                     │   encode_features()     │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   train.py              │
                     │   trains 3 models,       │
                     │   logs to MLflow,        │
                     │   saves model/scaler/    │
                     │   columns to disk         │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   predict_api.py         │
                     │   FastAPI service,        │
                     │   loads saved model,      │
                     │   POST /predict           │
                     └───────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                    │
   ┌──────────▼─────────┐ ┌──────▼───────┐ ┌─────────▼─────────┐
   │ simulate_traffic.py │ │  Dockerfile  │ │   dashboard.py     │
   │ sends simulated      │ │  containerizes│ │  Streamlit UI:     │
   │ monthly customer      │ │  the whole    │ │  high-risk list,   │
   │ batches to the API,   │ │  service      │ │  accuracy over     │
   │ logs predictions +    │ │               │ │  time, drift       │
   │ real outcomes to      │ │               │ │  warnings          │
   │ SQLite                │ │               │ │                    │
   └──────────┬───────────┘ └──────────────┘ └─────────┬──────────┘
              │                                          │
              │                                          │
   ┌──────────▼───────────┐                   ┌──────────▼──────────┐
   │   predictions.db      │◄──────────────────┤   detect_drift.py    │
   │   (SQLite)             │                   │   compares batches    │
   │                        │                   │   vs training data     │
   │                        │                   │   using Evidently      │
   └────────────────────────┘                   └────────────────────────┘
```

### Component breakdown

| File | Purpose |
|---|---|
| `src/data_prep.py` | `clean()` and `encode_features()` — shared data cleaning/encoding logic used by both training and serving, so the two never fall out of sync |
| `src/train.py` | Loads data, cleans it, trains a Logistic Regression model, evaluates it, logs the run to MLflow, and saves the model/scaler/column list to disk |
| `src/predict_api.py` | FastAPI service exposing `POST /predict` — validates input with Pydantic, runs it through the same cleaning/encoding pipeline as training, scales it, and returns a churn probability |
| `src/simulate_batches.py` | Splits the dataset into 12 simulated "monthly" batches, artificially shifting 4 of them (Nov/Dec/Jan/Feb) to represent a discount campaign — lower `MonthlyCharges`, more long-term contracts |
| `src/simulate_traffic.py` | Sends every customer in every batch to the live API as if it were real incoming traffic, and logs each prediction alongside the real historical outcome to SQLite |
| `src/detect_drift.py` | Compares a given batch's feature distributions against the original training data using Evidently, to check whether the incoming data still resembles what the model was trained on |
| `src/dashboard.py` | Streamlit dashboard showing current high-risk customers, model accuracy over time, and drift warnings per batch |
| `Dockerfile` | Packages the API (code + trained model + dependencies) into a portable container |

---

## Model

Three models were trained and compared: Logistic Regression, Random Forest, and XGBoost. Because churn is an imbalanced problem (~74% no-churn / ~26% churn), **recall on the churn class** — not raw accuracy — was used as the primary selection metric, since the business cost of *missing* an at-risk customer is generally higher than the cost of a false alarm.

| Model | Accuracy | Precision (churn) | Recall (churn) |
|---|---|---|---|
| **Logistic Regression** ✅ | 0.82 | 0.69 | **0.60** |
| Random Forest | 0.79 | 0.65 | 0.46 |
| XGBoost | 0.79 | 0.63 | 0.52 |

Logistic Regression was selected as the production model based on its higher recall, despite being the simplest of the three.

---

## Simulating drift

To test the drift-detection pipeline against a known, controlled scenario rather than just running it blindly, four of the twelve simulated batches (representing Nov, Dec, Jan, Feb) were deliberately altered to mimic a discount campaign: `MonthlyCharges` reduced by 5%, and `Contract` type re-weighted toward longer commitments. The drift detector correctly flagged `Contract` and `MonthlyCharges` as drifted specifically in those four batches, and correctly reported no drift in the other eight — validating that the detection pipeline works as intended.

---

## Running it locally

### Prerequisites
- Python 3.11
- Docker Desktop (only needed for the containerized version)

### Setup

```bash
git clone https://github.com/aaranli12345/churn-prediction.git
cd churn-prediction

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Train the model

```bash
cd src
python train.py
```

This trains the model and saves `model.joblib`, `scaler.joblib`, and `columns.joblib` to the `models/` folder. View experiment tracking with:

```bash
mlflow ui
```

### Run the API

```bash
uvicorn predict_api:app --reload
```

Visit `http://127.0.0.1:8000/docs` to test the `/predict` endpoint interactively.

### Run the dashboard

```bash
streamlit run dashboard.py
```

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Example request

```bash
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1020.0
}
```

Response:
```json
{
  "churn_probability": 0.636
}
```

---

## What I'd improve next

- Support customers with zero completed billing cycles (currently rejected by the API's input validation, since `TotalCharges` is required as a float)
- Weight business-critical columns (e.g. `Contract`, `MonthlyCharges`) more heavily in the drift-detection threshold, rather than relying solely on Evidently's default aggregate dataset-level score
- Move from SQLite to Postgres for the prediction log, as originally scoped
- Add authentication to the API before any real production use