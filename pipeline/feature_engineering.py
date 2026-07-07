import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter: Feature engineering - create derived features, encode, scale."""

    # Create derived features
    if "cgpa" in df.columns and "tenth_pct" in df.columns and "twelfth_pct" in df.columns:
        df["academic_index"] = (
            df["cgpa"] * 10 * 0.5 +
            df["tenth_pct"] * 0.25 +
            df["twelfth_pct"] * 0.25
        )
        print("[Feature Engineering] Created 'academic_index'")

    if "internships" in df.columns and "projects" in df.columns:
        df["experience_score"] = df["internships"] * 2 + df["projects"]
        print("[Feature Engineering] Created 'experience_score'")

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"[Feature Engineering] Encoded '{col}' -> '{col}_encoded'")

    # Drop original categorical columns
    df = df.drop(columns=categorical_cols)

    # Scale numerical features (exclude target)
    target_col = "placed"
    feature_cols = [c for c in df.columns if c != target_col]
    numerical_features = df[feature_cols].select_dtypes(include=["float64", "int64"]).columns.tolist()

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print(f"[Feature Engineering] Scaled {len(numerical_features)} numerical features")

    print(f"[Feature Engineering] Final features: {list(df.columns)}")
    return df
