# Monitoring Plan

## What to Track After Deployment

### 1. Data Drift
- Monitor distribution of recency_days and monetary_180d weekly
- Alert if average recency increases by more than 20 days
- Compare incoming feature distributions to training data monthly

### 2. Prediction Distribution
- Track daily ratio of high/medium/low risk predictions
- Alert if more than 70% of predictions are high risk on any day
- Log all predictions with timestamps to a database

### 3. Business Outcomes
- Compare predicted churners vs actual churners monthly
- Track retention campaign conversion rate by risk segment
- Measure revenue saved from successful interventions

### 4. API Health
- Monitor response time - alert if above 500ms
- Track error rate - alert if above 1%
- Log all 400/500 errors with input payload for debugging

### 5. Retraining Triggers
- Retrain if ROC-AUC drops below 0.80 on recent data
- Retrain every 90 days regardless of performance
- Retrain immediately if churn rate shifts by more than 10%

## Responsible Use Note
- The API output is a risk score, not a final decision
- Never deny service to a customer based on churn score alone
- High risk customers should receive help and retention offers
- Do not use churn score to discriminate by city tier or age group
- Always combine model output with human judgment
- Retention team should review edge cases manually
