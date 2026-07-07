import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter: Data cleaning - handle missing values, duplicates, outliers."""

    initial_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"[Data Cleaning] Removed {initial_rows - len(df)} duplicate rows")

    # Identify column types
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from numerical cols if present
    if "placed" in numerical_cols:
        numerical_cols.remove("placed")

    # Handle missing values - numerical: median, categorical: mode
    for col in numerical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"[Data Cleaning] Filled {missing} missing values in '{col}' with median")

    for col in categorical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"[Data Cleaning] Filled {missing} missing values in '{col}' with mode")

    # Outlier detection and capping using IQR for numerical columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"[Data Cleaning] Capped {outliers} outliers in '{col}'")

    print(f"[Data Cleaning] Final dataset: {len(df)} rows")
    return df
