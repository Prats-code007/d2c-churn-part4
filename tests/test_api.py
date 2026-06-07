import requests

BASE_URL = "http://localhost:8000"

sample_customer = {
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

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("PASS: health check")

def test_predict():
    response = requests.post(f"{BASE_URL}/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    print("PASS: predict endpoint")
    print("Result:", data)

def test_batch_predict():
    response = requests.post(f"{BASE_URL}/batch_predict", json=[sample_customer, sample_customer])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    print("PASS: batch predict endpoint")

if __name__ == "__main__":
    test_health()
    test_predict()
    test_batch_predict()
    print("All tests passed!")
