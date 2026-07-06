"""
Microservices Pattern - Prediction Service
===========================================
Independent, scalable service for real-time placement predictions.
Runs on its own port (8003) and can be scaled independently.
"""

import os
import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Placement Prediction Service",
    description="Microservice for real-time placement prediction with explainability",
    version="1.0.0"
)

# Load model and metadata at startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
model = None
metadata = None
explainer = None


def load_model():
    global model, metadata, explainer
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
        print(f"[Prediction Service] Model loaded: {metadata['model_name']}")
    else:
        print("[Prediction Service] WARNING: No model found. Run pipeline first.")


@app.on_event("startup")
def startup_event():
    load_model()


class StudentFeatures(BaseModel):
    cgpa: float
    aptitude_score: float
    communication_rating: int
    internships: int
    projects: int
    backlog: int
    extracurricular: int
    tenth_pct: float
    twelfth_pct: float
    stream: str = "CS"


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    top_factors: Dict[str, float]


@app.post("/predict", response_model=PredictionResponse)
def predict_placement(student: StudentFeatures):
    """Predict placement status for a student."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    # Prepare features (matching pipeline feature engineering)
    academic_index = (
        student.cgpa * 10 * 0.5 +
        student.tenth_pct * 0.25 +
        student.twelfth_pct * 0.25
    )
    experience_score = student.internships * 2 + student.projects

    features = np.array([[
        student.cgpa, student.aptitude_score,
        student.communication_rating, student.internships,
        student.projects, student.backlog,
        student.extracurricular, student.tenth_pct,
        student.twelfth_pct, academic_index,
        experience_score
    ]])

    # Get prediction and probability
    probability = model.predict_proba(features)[0][1]
    prediction = "Placed" if probability >= 0.5 else "Not Placed"

    # Determine risk level
    if probability >= 0.7:
        risk_level = "Low Risk"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Get SHAP explanations if available
    feature_names = [
        "CGPA", "Aptitude Score", "Communication", "Internships",
        "Projects", "Backlog", "Extracurricular", "10th %", "12th %",
        "Academic Index", "Experience Score"
    ]
    top_factors = {}
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            top_factors = {
                name: round(float(val), 4)
                for name, val in zip(feature_names, shap_values[0])
            }
        except Exception:
            # Fallback: use feature values as importance indicators
            for name, val in zip(feature_names, features[0]):
                top_factors[name] = round(float(val), 4)
    else:
        for name, val in zip(feature_names, features[0]):
            top_factors[name] = round(float(val), 4)

    return PredictionResponse(
        prediction=prediction,
        confidence=round(float(probability), 3),
        risk_level=risk_level,
        top_factors=top_factors
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata["model_name"] if metadata else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
