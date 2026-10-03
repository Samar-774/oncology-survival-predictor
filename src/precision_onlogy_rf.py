import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')


def run_precision_oncology_model1():
    print("=" * 80)
    print("PRECISION ONCOLOGY ENGINE – MODEL 1")
    print("Histology Classification + Explainability")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load preprocessed data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv("processed_data.csv")
        print(f"✓ Loaded preprocessed data: {df.shape}")
    except FileNotFoundError:
        print("❌ processed_data.csv not found. Run preprocessing first.")
        return

    # ------------------------------------------------------------------
    # Define target
    # ------------------------------------------------------------------
    class_target = "Histology_Encoded"

    if class_target not in df.columns:
        raise ValueError("Histology_Encoded not found in dataset.")

    # ------------------------------------------------------------------
    # Feature selection (SAFE: no survival leakage)
    # ------------------------------------------------------------------
    X = df.drop(columns=[class_target, "Survival_Rate"], errors="ignore")
    y = df[class_target]

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")

    # ------------------------------------------------------------------
    # Train / Test Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------------
    # Model 1: Random Forest Classifier
    # ------------------------------------------------------------------
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    rf_classifier.fit(X_train, y_train)
    print("✓ Histology Classification Model Trained")

    # ------------------------------------------------------------------
    # SHAP Explainability (Diagnosis)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INITIALIZING SHAP EXPLAINABILITY (MODEL 1)")
    print("=" * 80)

    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test)

    # ------------------------------------------------------------------
    # Global Feature Importance (RF)
    # ------------------------------------------------------------------
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_classifier.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Global Diagnostic Features:")
    print("-" * 50)
    print(feature_importance.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Patient-Level Explainability (3 Samples)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PATIENT-LEVEL DIAGNOSTIC EXPLANATIONS")
    print("=" * 80)

    sample_patients = X_test.iloc[:3]

    for i in range(len(sample_patients)):
        patient_data = sample_patients.iloc[i:i + 1]
        patient_id = sample_patients.index[i]

        predicted_class = rf_classifier.predict(patient_data)[0]

        # SHAP values for predicted class
        patient_shap = shap_values[int(predicted_class)][i]
        top_feature_idx = np.argmax(np.abs(patient_shap))
        top_feature = X.columns[top_feature_idx]
        direction = "Positive" if patient_shap[top_feature_idx] > 0 else "Negative"

        print(f"PATIENT ID: {patient_id}")
        print(f"  - Predicted Histology Class: {int(predicted_class)}")
        print(f"  - Primary Diagnostic Driver: {top_feature} ({direction} Influence)")
        print("-" * 40)

    # ------------------------------------------------------------------
    # Save predicted histology for Model 2 (HAND-OFF)
    # ------------------------------------------------------------------
    df["Predicted_Histology"] = rf_classifier.predict(X)

    df[["Predicted_Histology"]].to_csv(
        "predicted_histology_for_model2.csv",
        index=False
    )

    print("\n✓ Predicted histology saved for Model 2")
    print("=" * 80)
    print("MODEL 1 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_precision_oncology_model1()