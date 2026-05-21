# app/main.py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from app.schemas import CustomerData
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

# Global variable to hold our model pipeline
model_pipeline = None

@app.on_event("startup")
def load_model():
    """Loads the model pipeline when the server starts."""
    global model_pipeline
    model_path = "models/current_model.pkl"
    
    if os.path.exists(model_path):  
        model_pipeline = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {model_path}")

@app.get("/health")
def health_check():
    """Health check endpoint[cite: 70]."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    return {"status": "healthy", "model_version": "1.0"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    """Predicts churn for a single customer[cite: 69, 76]."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    
    # 1. Convert the incoming JSON payload back into a format pandas understands
    # We use by_alias=True so it matches the exact column names your pipeline expects
    input_data = customer.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    
    # 2. Make prediction using the loaded pipeline
    # The pipeline automatically handles the scaling and one-hot encoding! [cite: 75]
    try:
        prediction = model_pipeline.predict(df)[0]
        probability = model_pipeline.predict_proba(df)[0][1]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")