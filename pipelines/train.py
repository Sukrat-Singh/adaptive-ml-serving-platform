# pipelines/train.py

import pandas as pd
import joblib # We use this to save the model artifact
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_data(filepath):
    """Loads the raw data and drops leaky/unnecessary columns."""
    df = pd.read_excel(filepath)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)
    
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
        'Churn Reason', 'Churn Score', 'CLTV'
    ]
    df = df.drop(columns=cols_to_drop)
    return df

def build_pipeline(X):
    """Creates the scikit-learn preprocessing and modeling pipeline."""
    numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    return pipeline

def main():
    print("Starting training pipeline...")
    
    # 1. Load data
    df = load_data('../data/raw/Telco_customer_churn.xlsx')
    
    # 2. Split data
    X = df.drop('Churn Value', axis=1)
    y = df['Churn Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Build and train pipeline
    pipeline = build_pipeline(X_train)
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate (Sanity check)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save the artifact to the models/ directory
    # This fulfills the requirement to have a saved model artifact [cite: 162]
    # and matches the project structure expected[cite: 132, 133].
    import os
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/current_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()