# Placement Prediction System

ML-based system that predicts whether a student will secure campus placement based on academic performance, aptitude scores, communication skills, internships, and project experience.

## Problem Statement

Campus placements play a crucial role in determining students' career opportunities. However, students often lack insights into their placement readiness, and placement cells struggle to identify candidates who require additional support.

This system helps:
- **Students** understand their placement readiness and get actionable recommendations
- **Placement Officers** identify at-risk students who need additional training or mentoring
- **Recruiters** shortlist potential candidates efficiently

## Architecture

The system implements two architectural patterns:

### 1. Pipe and Filter Pattern (Offline ML Training)
Independent filters connected by pipes — each filter transforms data and passes it along:

```
[CSV Path] →|pipe|→ [Filter: Ingest] →|pipe|→ [Filter: Clean] →|pipe|→ [Filter: Engineer] →|pipe|→ [Filter: Train] →|pipe|→ [Filter: Evaluate] →|pipe|→ [Model]
```

### 2. Microservices Pattern (Online Serving)
Independent, scalable services communicating via REST APIs:

```
API Gateway (8000) → Prediction Service (8003)
                   → User Service (8002)
                   → Notification Service (8005)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| API | FastAPI |
| ML | Scikit-learn, XGBoost |
| Language | Python 3.10+ |

## Live Demo

The Streamlit app is self-contained — it generates synthetic data and trains the model automatically on startup.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Project Structure

```
├── app/
│   └── streamlit_app.py              # Frontend (Student, Officer, Recruiter dashboards)
├── pipeline/                          # Pipe and Filter Architecture Pattern
│   ├── run_pipeline.py                # Pipe and Filter orchestrator
│   ├── data_ingestion.py              # Stage 1: Load data
│   ├── data_cleaning.py               # Stage 2: Clean & handle outliers
│   ├── feature_engineering.py         # Stage 3: Derive features, encode, scale
│   ├── model_training.py              # Stage 4: Train candidate models
│   └── model_evaluation.py            # Stage 5: Evaluate & select best model
├── services/                          # Microservices Architecture Pattern
│   ├── api_gateway/main.py            # Central gateway (port 8000)
│   ├── prediction_service/model.py    # ML inference service (port 8003)
│   ├── notification_service/main.py   # Alert service (port 8005)
│   └── user_service/main.py           # User management (port 8002)
├── data/
│   └── generate_data.py               # Synthetic dataset generator
├── models/                            # Saved trained models
├── run_all.py                         # One-click: generate + train + serve
└── requirements.txt
```

## Running Locally

### Option 1: Streamlit App (Quickest)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Option 2: Full Pipeline + Microservices
```bash
pip install -r requirements.txt

# Generate data and train model
python run_all.py

# Start microservices (each in separate terminal)
python services/prediction_service/model.py
python services/notification_service/main.py
python services/user_service/main.py
python services/api_gateway/main.py
```

### Option 3: Deploy on Streamlit Cloud
1. Fork/push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → Set main file to `app/streamlit_app.py` → Deploy

## Features

| Feature | Description |
|---|---|
| Placement Prediction | Binary classification (Placed / Not Placed) with confidence score |
| Risk Assessment | Low / Medium / High risk categorization |
| Feature Importance | Shows which factors most influence the prediction |
| Recommendations | Actionable suggestions for at-risk students |
| Multi-model Comparison | Logistic Regression, Random Forest, XGBoost |
| Role-based Dashboards | Student, Placement Officer, Recruiter views |
| At-Risk Alerts | Notification service for students below threshold |

## ML Pipeline Details

- **Input Features:** CGPA, Aptitude Score, Communication Rating, Internships, Projects, Backlogs, Extra-curriculars, 10th %, 12th %, Stream
- **Derived Features:** Academic Index, Experience Score
- **Models Compared:** Logistic Regression, Random Forest, XGBoost
- **Evaluation Metric:** F1-Score (primary), Accuracy, AUC-ROC
- **Best Model:** Automatically selected based on highest F1-score

## Course

AIMLCZG546 - Software Engineering for Machine Learning
BITS Pilani (WILP)
