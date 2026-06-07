# Part 4: FastAPI Churn Scoring Service

## Project Overview
A FastAPI service that loads a trained churn model and returns churn risk predictions for a D2C personal care brand CRM system.

## Repository Structure
- app/main.py - FastAPI application
- tests/test_api.py - API test cases
- model.pkl - Trained Random Forest model
- monitoring_plan.md - Post-deployment monitoring plan
- requirements.txt - Python dependencies

## Setup Instructions

### Install dependencies
pip install -r requirements.txt

### Run the API
cd app
uvicorn main:app --reload

### The API will be available at
http://localhost:8000

## API Endpoints

### GET /health
Returns API health status.
Response: {"status": "ok", "model": "Random Forest Churn Predictor"}

### POST /predict
Accepts one customer feature payload and returns churn risk.
Response includes churn_probability, predicted_class, risk_level, risk_explanation.

### POST /batch_predict
Accepts multiple customers and returns predictions for each.

## Sample Request
POST /predict
{
    "recency_days": 45,
    "frequency_180d": 3,
    "monetary_180d": 1500.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.2,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 2,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 24.0,
    "days_since_signup": 365,
    "sessions_30d": 3,
    "product_views_30d": 10,
    "cart_adds_30d": 2,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 5,
    "campaign_clicks_30d": 1,
    "last_visit_days_ago": 15
}

## Sample Response
{
    "churn_probability": 0.42,
    "predicted_class": 1,
    "risk_level": "medium",
    "risk_explanation": "Medium churn risk. Recency of 45 days and 3 sessions in last 30 days."
}

## Run Tests
python tests/test_api.py

## Model Notes
- Model: Random Forest Classifier
- Trained on: rfm_modeling_snapshot.csv
- ROC-AUC: 0.878
- Threshold: 0.4
