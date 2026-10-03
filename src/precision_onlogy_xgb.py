import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import shap
import warnings

warnings.filterwarnings("ignore")


def run_precision_oncology_model2():
    print("=" * 80)
    print("PRECISION ONCOLOGY ENGINE – MODEL 2")
    print("Survival Prognostication (XGBoost + SHAP)")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    df = pd.read_csv("processed_data.csv")
    hist_pred = pd.read_csv("predicted_histology_for_model2.csv")

    # Attach Model-1 output
    df["Predicted_Histology"] = hist_pred["Predicted_Histology"]

    # ---------------------------------------------------------
    # Define target
    # ---------------------------------------------------------
    target = "Survival_Rate"

    X = df.drop(columns=["Histology_Encoded", target], errors="ignore")
    y = df[target]

    print(f"✓ Feature matrix: {X.shape}")
    print(f"✓ Target range: {y.min():.1f}% – {y.max():.1f}%")

    # ---------------------------------------------------------
    # Train / Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # XGBoost Survival Regressor
    # ---------------------------------------------------------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("✓ Survival Prognostication Model Trained")

    # ---------------------------------------------------------
    # SHAP Explainability (Survival)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SHAP EXPLAINABILITY – SURVIVAL PREDICTION")
    print("=" * 80)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global importance
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean_SHAP_Impact": importance
    }).sort_values(by="Mean_SHAP_Impact", ascending=False)

    print("\nTop Global Survival Drivers:")
    print(importance_df.head(5).to_string(index=False))

    # ---------------------------------------------------------
    # Patient-Level Survival Explanation
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PATIENT-LEVEL SURVIVAL EXPLANATIONS")
    print("=" * 80)

    sample_patients = X_test.iloc[:3]

    for i in range(len(sample_patients)):
        patient = sample_patients.iloc[i:i+1]
        patient_id = patient.index[0]

        survival_pct = model.predict(patient)[0]
        survival_pct = np.clip(survival_pct, 0, 100)

        shap_row = shap_values[i]
        top_idx = np.argmax(np.abs(shap_row))
        top_feature = X.columns[top_idx]
        direction = "Positive" if shap_row[top_idx] > 0 else "Negative"

        print(f"PATIENT ID: {patient_id}")
        print(f"  - Predicted Survival Probability: {survival_pct:.1f}%")
        print(f"  - Primary Survival Driver: {top_feature} ({direction})")
        print("-" * 40)

    print("\n" + "=" * 80)
    print("MODEL 2 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_precision_oncology_model2()