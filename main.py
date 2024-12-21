import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
try:
    voting_clf = joblib.load("final_model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError as e:
    raise RuntimeError(f"Required model or scaler file is missing: {e}")

# Define feature names
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Define the input schema
class BreastCancerPredictionInput(BaseModel):
    inputs: dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their corresponding values.",
        example={name: 0.0 for name in feature_names},
    )

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Breast Cancer Prediction API!"}

# Endpoint to retrieve feature names
@app.get("/features/")
async def get_features():
    """Returns the feature names expected for the prediction."""
    return {"feature_names": feature_names}

# Prediction endpoint
@app.post("/predict/")
async def predict(input_data: BreastCancerPredictionInput):
    input_features = input_data.inputs

    # Validate missing features
    missing_features = [feature for feature in feature_names if feature not in input_features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing_features}",
        )

    # Validate extra features
    extra_features = [feature for feature in input_features if feature not in feature_names]
    if extra_features:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected features: {extra_features}",
        )

    try:
        # Extract feature values in the correct order
        feature_values = [input_features[feature] for feature in feature_names]
        sample_data = np.array([feature_values])

        # Scale the input
        sample_data_scaled = scaler.transform(sample_data)

        # Predict using the model
        prediction = voting_clf.predict(sample_data_scaled)

        return {"prediction": "Malignant (1)" if prediction[0] == 1 else "Benign (0)"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import nest_asyncio  # Required for running in Jupyter

    # Apply nest_asyncio to allow uvicorn to run in Jupyter's event loop
    nest_asyncio.apply()

    # Start the FastAPI server
    print("Starting FastAPI server... Access the API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
