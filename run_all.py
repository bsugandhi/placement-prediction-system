"""
Main entry point to run the complete Placement Prediction System.
=================================================================
1. Generates sample data
2. Runs the ML pipeline (Pipeline Pattern)
3. Provides instructions to start microservices (Microservices Pattern)
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)


def generate_data():
    """Generate synthetic placement data."""
    print("\n[Step 1] Generating synthetic placement data...")
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Run data generation
    exec(open(os.path.join(data_dir, "generate_data.py")).read())


def run_pipeline():
    """Run the ML training pipeline."""
    print("\n[Step 2] Running ML Pipeline...")
    from pipeline.run_pipeline import PlacementPipeline

    data_path = os.path.join(BASE_DIR, "data", "placement_data.csv")
    pipeline = PlacementPipeline(data_path)
    pipeline.run()


def print_service_instructions():
    """Print instructions to start microservices."""
    print("\n" + "=" * 60)
    print("MICROSERVICES READY TO START")
    print("=" * 60)
    print("""
To start the microservices (run each in a separate terminal):

1. Prediction Service (Port 8003):
   python services/prediction_service/model.py

2. Notification Service (Port 8005):
   python services/notification_service/main.py

3. User Service (Port 8002):
   python services/user_service/main.py

4. API Gateway (Port 8000):
   python services/api_gateway/main.py

5. Streamlit Frontend:
   streamlit run app/streamlit_app.py

API Documentation:
- Gateway:    http://localhost:8000/docs
- Prediction: http://localhost:8003/docs
- Users:      http://localhost:8002/docs
- Notifications: http://localhost:8005/docs
""")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    generate_data()
    run_pipeline()
    print_service_instructions()
