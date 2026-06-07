from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Churn Prediction API")

model = joblib.load("model.pkl")

class CustomerFeatures(BaseModel):
    recency_days: float
    frequency_180d: float
    monetary_180d: float
    return_rate_180d: float
    avg_discount_pct_180d: float
    avg_rating_180d: float
    category_diversity_180d: float
    ticket_count_90d: float
    negative_ticket_rate_90d: float
    avg_resolution_hours_90d: float
    days_since_signup: float
    sessions_30d: float
    product_views_30d: float
    cart_adds_30d: float
    wishlist_adds_30d: float
    abandoned_carts_30d: float
    email_opens_30d: float
    campaign_clicks_30d: float
    last_visit_days_ago: float

def get_risk_level(prob):
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    else:
        return "low"

def get_explanation(prob, recency, sessions):
    if prob >= 0.7:
        return f"High churn risk. Customer inactive for {recency} days with only {sessions} sessions."
    elif prob >= 0.4:
        return f"Medium churn risk. Recency of {recency} days and {sessions} sessions in last 30 days."
    else:
        return f"Low churn risk. Customer is active with {sessions} sessions recently."

@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest Churn Predictor"}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        features = pd.DataFrame([customer.dict()])
        prob = model.predict_proba(features)[0][1]
        predicted_class = int(prob >= 0.4)
        return {
            "churn_probability": round(float(prob), 4),
            "predicted_class": predicted_class,
            "risk_level": get_risk_level(prob),
            "risk_explanation": get_explanation(prob, customer.recency_days, customer.sessions_30d)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
def batch_predict(customers: List[CustomerFeatures]):
    results = []
    for customer in customers:
        features = pd.DataFrame([customer.dict()])
        prob = model.predict_proba(features)[0][1]
        predicted_class = int(prob >= 0.4)
        results.append({
            "churn_probability": round(float(prob), 4),
            "predicted_class": predicted_class,
            "risk_level": get_risk_level(prob),
            "risk_explanation": get_explanation(prob, customer.recency_days, customer.sessions_30d)
        })
    return {"predictions": results, "total": len(results)}
