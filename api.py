from fastapi import FastAPI  
import joblib  
import numpy as np  
import pandas as pd  

# Load model  
model = joblib.load(r"D:\Projects\fraud_detection_pipeline.joblib")  

app = FastAPI()  
@app.post("/predict")
def predict(transaction: dict):
    try:
        # Get feature names directly from the model
        feature_names = model.named_steps['xgboost'].feature_names_in_
        features = pd.DataFrame([transaction])[list(feature_names)]
        
        # Rest of your prediction code...
        xgb_prob = float(model.named_steps['xgboost'].predict_proba(features)[0,1])
        iso_score = float(-model.named_steps['isolation_forest'].score_samples(features)[0])
        combined_prob = 0.7 * xgb_prob + 0.3 * iso_score
        
        return {
            "fraud_probability": combined_prob,
            "is_fraud": bool(combined_prob >= 0.412)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "expected_features": list(model.named_steps['xgboost'].feature_names_in_),
            "received_features": list(transaction.keys())
        }
