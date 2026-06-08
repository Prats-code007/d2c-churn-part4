from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Churn Prediction API", description="D2C Customer Churn Scoring Service")

# Load model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Exact feature order the model expects
FEATURE_COLUMNS = [
    'recency_days', 'frequency_180d', 'monetary_180d', 'return_rate_180d',
    'avg_discount_pct_180d', 'avg_rating_180d', 'category_diversity_180d',
    'ticket_count_90d', 'negative_ticket_rate_90d', 'avg_resolution_hours_90d',
    'days_since_signup', 'sessions_30d', 'product_views_30d', 'cart_adds_30d',
    'wishlist_adds_30d', 'abandoned_carts_30d', 'email_opens_30d',
    'campaign_clicks_30d', 'last_visit_days_ago',
    'city_tier_Tier 2', 'city_tier_Tier 3',
    'age_group_25-34', 'age_group_35-44', 'age_group_45+',
    'acquisition_channel_Influencer', 'acquisition_channel_Instagram',
    'acquisition_channel_Marketplace', 'acquisition_channel_Organic',
    'acquisition_channel_Referral', 'loyalty_tier_Platinum',
    'loyalty_tier_Silver', 'preferred_category_Fragrance',
    'preferred_category_Hair Care', 'preferred_category_Makeup',
    'preferred_category_Skin Care', 'preferred_category_Wellness',
    'marketing_consent_Yes'
]

class CustomerFeatures(BaseModel):
    recency_days: float
    frequency_180d: float
    monetary_180d: float
    return_rate_180d: float = 0.0
    avg_discount_pct_180d: float = 0.0
    avg_rating_180d: float = 4.0
    category_diversity_180d: float = 1.0
    ticket_count_90d: float = 0.0
    negative_ticket_rate_90d: float = 0.0
    avg_resolution_hours_90d: float = 24.0
    days_since_signup: float = 365.0
    sessions_30d: float
    product_views_30d: float = 0.0
    cart_adds_30d: float = 0.0
    wishlist_adds_30d: float = 0.0
    abandoned_carts_30d: float = 0.0
    email_opens_30d: float = 0.0
    campaign_clicks_30d: float = 0.0
    last_visit_days_ago: float
    city_tier: str = "Tier 1"
    age_group: str = "18-24"
    acquisition_channel: str = "Google Search"
    loyalty_tier: str = "Bronze"
    preferred_category: str = "Wellness"
    marketing_consent: str = "No"

def encode_features(customer: CustomerFeatures) -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row['recency_days'] = customer.recency_days
    row['frequency_180d'] = customer.frequency_180d
    row['monetary_180d'] = customer.monetary_180d
    row['return_rate_180d'] = customer.return_rate_180d
    row['avg_discount_pct_180d'] = customer.avg_discount_pct_180d
    row['avg_rating_180d'] = customer.avg_rating_180d
    row['category_diversity_180d'] = customer.category_diversity_180d
    row['ticket_count_90d'] = customer.ticket_count_90d
    row['negative_ticket_rate_90d'] = customer.negative_ticket_rate_90d
    row['avg_resolution_hours_90d'] = customer.avg_resolution_hours_90d
    row['days_since_signup'] = customer.days_since_signup
    row['sessions_30d'] = customer.sessions_30d
    row['product_views_30d'] = customer.product_views_30d
    row['cart_adds_30d'] = customer.cart_adds_30d
    row['wishlist_adds_30d'] = customer.wishlist_adds_30d
    row['abandoned_carts_30d'] = customer.abandoned_carts_30d
    row['email_opens_30d'] = customer.email_opens_30d
    row['campaign_clicks_30d'] = customer.campaign_clicks_30d
    row['last_visit_days_ago'] = customer.last_visit_days_ago
    city_col = f"city_tier_{customer.city_tier}"
    if city_col in row:
        row[city_col] = 1.0
    age_col = f"age_group_{customer.age_group}"
    if age_col in row:
        row[age_col] = 1.0
    acq_col = f"acquisition_channel_{customer.acquisition_channel}"
    if acq_col in row:
        row[acq_col] = 1.0
    loyalty_col = f"loyalty_tier_{customer.loyalty_tier}"
    if loyalty_col in row:
        row[loyalty_col] = 1.0
    cat_col = f"preferred_category_{customer.preferred_category}"
    if cat_col in row:
        row[cat_col] = 1.0
    if customer.marketing_consent == "Yes":
        row['marketing_consent_Yes'] = 1.0
    return pd.DataFrame([row])

def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    else:
        return "low"

def get_explanation(prob: float, recency: float, sessions: float) -> str:
    if prob >= 0.7:
        return f"High churn risk. Customer inactive for {recency:.0f} days with only {sessions:.0f} sessions in last 30 days."
    elif prob >= 0.4:
        return f"Medium churn risk. Recency of {recency:.0f} days and {sessions:.0f} sessions suggest declining engagement."
    else:
        return f"Low churn risk. Customer shows healthy activity with {sessions:.0f} sessions and recency of {recency:.0f} days."

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Random Forest Churn Predictor",
        "model_loaded": model is not None,
        "version": "1.0"
    }

@app.post("/predict")
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        features = encode_features(customer)
        prob = float(model.predict_proba(features)[0][1])
        predicted_class = int(prob >= 0.4)
        return {
            "churn_probability": round(prob, 4),
            "predicted_class": predicted_class,
            "risk_level": get_risk_level(prob),
            "risk_explanation": get_explanation(prob, customer.recency_days, customer.sessions_30d)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
def batch_predict(customers: List[CustomerFeatures]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="No customers provided")
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 customers per batch")
    results = []
    for customer in customers:
        try:
            features = encode_features(customer)
            prob = float(model.predict_proba(features)[0][1])
            predicted_class = int(prob >= 0.4)
            results.append({
                "churn_probability": round(prob, 4),
                "predicted_class": predicted_class,
                "risk_level": get_risk_level(prob),
                "risk_explanation": get_explanation(prob, customer.recency_days, customer.sessions_30d)
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "total": len(results)}
