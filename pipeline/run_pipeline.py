"""
Pipeline Architecture Pattern Implementation
=============================================
Sequential stages with clear input/output contracts.
Each stage is independently testable and replaceable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.data_ingestion import ingest_data
from pipeline.data_cleaning import clean_data
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import train_models
from pipeline.model_evaluation import evaluate_and_select


class PlacementPipeline:
    """Pipeline architecture: sequential stages with clear contracts.

    Each stage:
    - Has a defined input type and output type
    - Can be independently tested
    - Can be replaced without affecting other stages
    - Logs its progress for monitoring
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.stages = [
            ("Data Ingestion", ingest_data),
            ("Data Cleaning", clean_data),
            ("Feature Engineering", engineer_features),
            ("Model Training", train_models),
            ("Model Evaluation", evaluate_and_select),
        ]

    def run(self):
        """Execute pipeline stages sequentially."""
        print("\n" + "=" * 60)
        print("PLACEMENT PREDICTION PIPELINE")
        print("=" * 60)

        data = self.data_path
        for i, (stage_name, stage_fn) in enumerate(self.stages, 1):
            print(f"\n{'─' * 60}")
            print(f"Stage {i}/{len(self.stages)}: {stage_name}")
            print(f"{'─' * 60}")
            data = stage_fn(data)

        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}\n")
        return data


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "placement_data.csv"
    )
    pipeline = PlacementPipeline(data_path)
    result = pipeline.run()
