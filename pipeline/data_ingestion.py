import pandas as pd


def ingest_data(data_path: str) -> pd.DataFrame:
    """Stage 1: Load raw data from CSV source."""
    df = pd.read_csv(data_path)
    print(f"[Data Ingestion] Loaded {len(df)} records, {len(df.columns)} features")
    print(f"[Data Ingestion] Columns: {list(df.columns)}")
    return df
