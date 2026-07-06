"""Generate synthetic placement dataset for demo purposes."""

import pandas as pd
import numpy as np

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

# Generate placement outcome based on features (realistic logic)
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

# Add noise
placement_score += np.random.normal(0, 10, n_students)

# Stream bonus
stream_bonus = df["stream"].map({"CS": 10, "EC": 5, "EE": 3, "ME": 0, "CE": -2})
placement_score += stream_bonus

# Convert to binary with threshold
threshold = np.percentile(placement_score, 35)  # ~65% placement rate
df["placed"] = (placement_score > threshold).astype(int)

print(f"Dataset generated: {len(df)} records")
print(f"Placement rate: {df['placed'].mean():.1%}")
print(f"Columns: {list(df.columns)}")

# Save
df.to_csv("data/placement_data.csv", index=False)
print("Saved to data/placement_data.csv")
