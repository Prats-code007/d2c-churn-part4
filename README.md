# Part 4: FastAPI Churn Scoring Service

## Project Overview
A FastAPI service that loads a trained Random Forest churn prediction model and returns churn risk scores for customers of a D2C personal care brand. Designed to integrate with an internal CRM tool to help the retention team identify at-risk customers before they churn.

## Setup Instructions

Step 1 - Clone the repository
git clone https://github.com/Prats-code007/d2c-churn-part4
cd d2c-churn-part4

Step 2 - Install dependencies
pip install -r requirements.txt

Step 3 - Run the API
cd app
uvicorn main:app --reload

The API will be available at: http://localhost:8000

## Endpoint Details

GET /health
Returns API health status and confirms model is loaded.

POST /predict
Accepts one customer feature payload and returns churn risk prediction.

POST /batch_predict
Accepts a list of up to 100 customers and returns predictions for each.

## Sample Request

POST http://localhost:8000/predict

{
    "recency_days": 120,
    "frequency_180d": 2,
    "monetary_180d": 800.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.3,
    "avg_rating_180d": 3.0,
    "category_diversity_180d": 1.0,
    "ticket_count_90d": 2.0,
    "negative_ticket_rate_90d": 0.5,
    "avg_resolution_hours_90d": 48.0,
    "days_since_signup": 400.0,
    "sessions_30d": 2,
    "product_views_30d": 5.0,
    "cart_adds_30d": 1.0,
    "wishlist_adds_30d": 0.0,
    "abandoned_carts_30d": 1.0,
    "email_opens_30d": 2.0,
    "campaign_clicks_30d": 0.0,
    "last_visit_days_ago": 30,
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Bronze",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes"
}

## Sample Response

{
    "churn_probability": 0.79,
    "predicted_class": 1,
    "risk_level": "high",
    "risk_explanation": "High churn risk. Customer inactive for 120 days with only 2 sessions in last 30 days."
}

## Test Execution Instructions

Make sure the API is running first, then in a new terminal run:
python tests/test_api.py

Expected output:
PASS: health check
PASS: predict endpoint
PASS: batch predict endpoint
All tests passed!

## Model Notes
- Model type: Random Forest Classifier
- Trained on: rfm_modeling_snapshot.csv
- Snapshot date: 2025-09-30
- Training samples: 1728 customers
- Test samples: 336 customers
- ROC-AUC: 0.878
- Decision threshold: 0.4
- Features: 37 after one-hot encoding
- Top features: recency_days, last_visit_days_ago, monetary_180d
- Model file: model.pkl saved using joblib
- No future data used as input features

## Responsible Use
- This API output is a risk score, not a final decision
- Never deny service to a customer based on churn score alone
- High risk customers should receive retention offers not punishment
- Always combine model output with human judgment
