import joblib
import os
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report
)


def evaluate_and_select(training_result: dict) -> str:
    """Filter: Evaluate all models and save the best one."""

    models = training_result["models"]
    X_test = training_result["X_test"]
    y_test = training_result["y_test"]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    best_model_name = None
    best_f1 = 0
    best_model = None

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("=" * 60)

    # Print detailed classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, target_names=["Not Placed", "Placed"]))

    # Save best model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"[Evaluation] Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "f1_score": best_f1,
        "feature_cols": training_result["feature_cols"]
    }
    metadata_path = os.path.join(model_dir, "model_metadata.joblib")
    joblib.dump(metadata, metadata_path)

    return model_path
