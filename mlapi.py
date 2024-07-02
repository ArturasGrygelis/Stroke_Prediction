import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

# Create a FastAPI instance
app = FastAPI()

# Load necessary objects
categorical_features = joblib.load("categorical_features.joblib")
features = joblib.load("features.joblib")
encoder = joblib.load("encoder.joblib")

# Define a Pydantic model for the input data
class ScoringItem(BaseModel):
    gender: str
    work_type: str
    Residence_type: str
    smoking_status: str 
    age: float 
    hypertension: int 
    heart_disease: int 
    ever_married: int 
    avg_glucose_level: float 
    bmi: float

# Load the LightGBM model
with open('lgb1_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the scoring endpoint
@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    try:
        # Convert the Pydantic model to a Pandas DataFrame
        df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())

        # Encode categorical features
        encoded_features = encoder.transform(df[categorical_features])

        # Get the feature names from the encoder
        feature_names = encoder.get_feature_names_out(input_features=categorical_features)

        # Create a DataFrame with the encoded features and feature names
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
        df_encoded = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)

        # Make probability predictions using the LightGBM model
        pred_proba = model.predict_proba(df_encoded)

        # Assuming a binary classification problem, use probabilities for the positive class
        positive_class_probability = pred_proba[:, 1]

        # Prepare the response with SHAP values
        response = {
            "Probability of getting stroke is: ": positive_class_probability[0],
        }

        return response

    except Exception as e:
        # Handle exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))