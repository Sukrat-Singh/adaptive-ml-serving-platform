# app/main.py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from app.schemas import CustomerData
import os
import csv
from datetime import datetime

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
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    
    input_data = customer.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    
    try:
        prediction = model_pipeline.predict(df)[0]
        probability = model_pipeline.predict_proba(df)[0][1]
        
        # --- LOGGING LOGIC ---
        log_dir = "data/drift_logs"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "requests.csv")
        
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(input_data.keys()) + ['timestamp', 'prediction', 'probability'])
            if not file_exists:
                writer.writeheader()
            
            log_entry = input_data.copy()
            log_entry.update({
                'timestamp': datetime.now().isoformat(),
                'prediction': int(prediction),
                'probability': float(probability)
            })
            writer.writerow(log_entry)
        # --- END LOGGING ---
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")