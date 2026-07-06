"""
Streamlit Frontend for Placement Prediction System
====================================================
Self-contained app: generates data + trains model on first run.
Deployable on Streamlit Community Cloud directly from GitHub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from xgboost import XGBClassifier

st.set_page_config(page_title="Placement Prediction System", layout="wide")


# ============================================================
# DATA GENERATION (runs once, cached)
# ============================================================
@st.cache_data
def generate_placement_data():
    """Generate synthetic placement dataset."""
    np.random.seed(42)
    n_students = 1000

    data = {
        "student_id": [f"STU{i:04d}" for i in range(1, n_students + 1)],
        "cgpa": np.round(np.random.uniform(4.0, 10.0, n_students), 2),
        "aptitude_score": np.random.randint(20, 100, n_students),
        "communication_rating": np.random.randint(1, 6, n_students),
        "internships": np.random.randint(0, 4, n_students),
        "projects": np.random.randint(0, 8, n_students),
        "backlog": np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
        "extracurricular": np.random.choice([0, 1], n_students, p=[0.4, 0.6]),
        "tenth_pct": np.round(np.random.uniform(50, 99, n_students), 1),
        "twelfth_pct": np.round(np.random.uniform(45, 98, n_students), 1),
        "stream": np.random.choice(
            ["CS", "EC", "ME", "CE", "EE"], n_students,
            p=[0.35, 0.2, 0.2, 0.15, 0.1]
        ),
    }

    df = pd.DataFrame(data)

    # Generate placement outcome based on features
    placement_score = (
        df["cgpa"] * 8 +
        df["aptitude_score"] * 0.5 +
        df["communication_rating"] * 5 +
        df["internships"] * 10 +
        df["projects"] * 3 +
        df["backlog"] * (-15) +
        df["extracurricular"] * 5 +
        df["tenth_pct"] * 0.2 +
        df["twelfth_pct"] * 0.2
    )
    placement_score += np.random.normal(0, 10, n_students)
    stream_bonus = df["stream"].map({"CS": 10, "EC": 5, "EE": 3, "ME": 0, "CE": -2})
    placement_score += stream_bonus

    threshold = np.percentile(placement_score, 35)
    df["placed"] = (placement_score > threshold).astype(int)

    return df


# ============================================================
# ML PIPELINE (Pipeline Pattern - runs once, cached)
# ============================================================
@st.cache_resource
def run_pipeline(df):
    """
    Pipeline Architecture Pattern Implementation.
    Stages: Ingestion -> Cleaning -> Feature Engineering -> Training -> Evaluation
    """

    # --- Stage 1: Data Ingestion ---
    raw_df = df.copy()

    # --- Stage 2: Data Cleaning ---
    raw_df = raw_df.drop_duplicates()
    numerical_cols = ["cgpa", "aptitude_score", "tenth_pct", "twelfth_pct"]
    for col in numerical_cols:
        raw_df[col] = raw_df[col].fillna(raw_df[col].median())
    for col in numerical_cols:
        Q1, Q3 = raw_df[col].quantile(0.25), raw_df[col].quantile(0.75)
        IQR = Q3 - Q1
        raw_df[col] = raw_df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # --- Stage 3: Feature Engineering ---
    raw_df["academic_index"] = (
        raw_df["cgpa"] * 10 * 0.5 +
        raw_df["tenth_pct"] * 0.25 +
        raw_df["twelfth_pct"] * 0.25
    )
    raw_df["experience_score"] = raw_df["internships"] * 2 + raw_df["projects"]

    le = LabelEncoder()
    raw_df["stream_encoded"] = le.fit_transform(raw_df["stream"])

    feature_cols = [
        "cgpa", "aptitude_score", "communication_rating",
        "internships", "projects", "backlog", "extracurricular",
        "tenth_pct", "twelfth_pct", "academic_index",
        "experience_score", "stream_encoded"
    ]

    X = raw_df[feature_cols]
    y = raw_df["placed"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # --- Stage 4: Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", use_label_encoder=False
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "f1": f1, "accuracy": acc}

    # --- Stage 5: Model Evaluation & Selection ---
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    best_f1 = results[best_name]["f1"]

    return {
        "model": best_model,
        "model_name": best_name,
        "f1_score": best_f1,
        "results": results,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# APP START
# ============================================================
df = generate_placement_data()
pipeline_result = run_pipeline(df)

model = pipeline_result["model"]
metadata = pipeline_result

st.title("Placement Prediction System")
st.markdown("*ML-based system to predict student placement readiness*")
st.success(
    f"Model: **{metadata['model_name']}** | "
    f"F1-Score: **{metadata['f1_score']:.4f}** | "
    f"Dataset: **{len(df)} students**"
)

# Sidebar
role = st.sidebar.selectbox("Select Role", ["Student", "Placement Officer", "Recruiter"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture Patterns")
st.sidebar.markdown("1. **Pipeline Pattern** - Data flows through sequential stages")
st.sidebar.markdown("2. **Microservices Pattern** - Independent services for prediction, notifications, user mgmt")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Comparison")
for name, res in metadata["results"].items():
    icon = " ✅" if name == metadata["model_name"] else ""
    st.sidebar.markdown(f"- {name}: F1={res['f1']:.4f}{icon}")


# ============================================================
# STUDENT VIEW
# ============================================================
if role == "Student":
    st.header("Student Placement Readiness Check")
    st.markdown("Enter your details below to check your placement readiness.")

    col1, col2, col3 = st.columns(3)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        aptitude_score = st.slider("Aptitude Score", 0, 100, 65)
        communication_rating = st.slider("Communication Rating", 1, 5, 3)
        stream = st.selectbox("Stream", ["CS", "EC", "ME", "CE", "EE"])

    with col2:
        internships = st.number_input("Number of Internships", 0, 5, 1)
        projects = st.number_input("Number of Projects", 0, 10, 2)
        backlog = st.selectbox("Any Backlogs?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col3:
        extracurricular = st.selectbox(
            "Extra-curricular Activities?", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        tenth_pct = st.slider("10th Percentage", 30.0, 100.0, 75.0, 0.5)
        twelfth_pct = st.slider("12th Percentage", 30.0, 100.0, 72.0, 0.5)

    if st.button("Predict My Placement Chances", type="primary"):
        # Feature engineering
        academic_index = cgpa * 10 * 0.5 + tenth_pct * 0.25 + twelfth_pct * 0.25
        experience_score = internships * 2 + projects
        stream_map = {"CE": 0, "CS": 1, "EC": 2, "EE": 3, "ME": 4}
        stream_encoded = stream_map.get(stream, 0)

        features = np.array([[
            cgpa, aptitude_score, communication_rating,
            internships, projects, backlog, extracurricular,
            tenth_pct, twelfth_pct, academic_index,
            experience_score, stream_encoded
        ]])

        # Scale features
        features_scaled = metadata["scaler"].transform(features)

        probability = model.predict_proba(features_scaled)[0][1]
        prediction = "Placed" if probability >= 0.5 else "Not Placed"

        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if prediction == "Placed":
                st.metric("Prediction", "✅ Placed")
            else:
                st.metric("Prediction", "❌ Not Placed")
        with col_res2:
            st.metric("Confidence", f"{probability:.1%}")
        with col_res3:
            if probability >= 0.7:
                st.metric("Risk Level", "🟢 Low Risk")
            elif probability >= 0.4:
                st.metric("Risk Level", "🟡 Medium Risk")
            else:
                st.metric("Risk Level", "🔴 High Risk")

        st.progress(float(probability))

        # Feature importance (for tree-based models)
        st.markdown("### Key Factors (Feature Importance)")
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": metadata["feature_cols"],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance.set_index("Feature")["Importance"])

        # Recommendations
        st.markdown("### Recommendations")
        if probability < 0.5:
            recommendations = []
            if cgpa < 7.0:
                recommendations.append("📚 Improve your CGPA - focus on core subjects")
            if aptitude_score < 60:
                recommendations.append("🧮 Practice aptitude questions regularly")
            if communication_rating < 3:
                recommendations.append("🗣️ Work on communication skills - join speaking clubs")
            if internships < 1:
                recommendations.append("💼 Get at least one internship experience")
            if projects < 2:
                recommendations.append("🛠️ Build more projects to showcase skills")
            if backlog == 1:
                recommendations.append("📖 Clear backlogs as priority")

            if recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.info("Keep working on improving all areas consistently.")
        else:
            st.success("You are on the right track! Keep up the good work.")


# ============================================================
# PLACEMENT OFFICER VIEW
# ============================================================
elif role == "Placement Officer":
    st.header("Placement Officer Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Placement Rate", f"{df['placed'].mean():.1%}")
    with col3:
        st.metric("Avg CGPA", f"{df['cgpa'].mean():.2f}")
    with col4:
        st.metric("Students with Backlogs", int(df["backlog"].sum()))

    st.markdown("### At-Risk Students")
    st.markdown("Students flagged based on low CGPA, low aptitude, or no experience.")

    at_risk = df[
        (df["cgpa"] < 6.5) |
        (df["aptitude_score"] < 40) |
        ((df["internships"] == 0) & (df["projects"] < 2))
    ].copy()
    at_risk = at_risk.sort_values("cgpa")

    st.dataframe(
        at_risk[["student_id", "cgpa", "aptitude_score", "communication_rating",
                 "internships", "projects", "backlog", "stream", "placed"]].head(25),
        use_container_width=True
    )
    st.info(f"Total at-risk students: **{len(at_risk)}** out of {len(df)}")

    st.markdown("### Stream-wise Placement Statistics")
    stream_stats = df.groupby("stream")["placed"].agg(["count", "sum", "mean"]).reset_index()
    stream_stats.columns = ["Stream", "Total Students", "Placed", "Placement Rate"]
    stream_stats["Not Placed"] = stream_stats["Total Students"] - stream_stats["Placed"]
    stream_stats["Placement Rate"] = stream_stats["Placement Rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(stream_stats, use_container_width=True)

    st.markdown("### Batch CGPA Distribution")
    st.bar_chart(df["cgpa"].value_counts(bins=10).sort_index())


# ============================================================
# RECRUITER VIEW
# ============================================================
elif role == "Recruiter":
    st.header("Recruiter Dashboard")
    st.markdown("Filter and shortlist candidates based on your requirements.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_cgpa = st.slider("Minimum CGPA", 0.0, 10.0, 7.0, 0.5)
    with col2:
        min_aptitude = st.slider("Minimum Aptitude Score", 0, 100, 60)
    with col3:
        selected_streams = st.multiselect(
            "Streams", df["stream"].unique().tolist(),
            default=["CS", "EC"]
        )

    col4, col5 = st.columns(2)
    with col4:
        min_internships = st.number_input("Minimum Internships", 0, 5, 0)
    with col5:
        no_backlogs = st.checkbox("No Backlogs Only", value=False)

    # Apply filters
    filtered = df[
        (df["cgpa"] >= min_cgpa) &
        (df["aptitude_score"] >= min_aptitude) &
        (df["stream"].isin(selected_streams)) &
        (df["internships"] >= min_internships)
    ]
    if no_backlogs:
        filtered = filtered[filtered["backlog"] == 0]

    filtered = filtered.sort_values("cgpa", ascending=False)

    st.metric("Matching Candidates", len(filtered))
    st.dataframe(
        filtered[["student_id", "cgpa", "aptitude_score", "communication_rating",
                  "internships", "projects", "stream", "backlog"]].head(50),
        use_container_width=True
    )
