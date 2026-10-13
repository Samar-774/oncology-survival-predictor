import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

warnings.filterwarnings("ignore")

def run_precision_oncology_model2():
    print("=" * 80)
    print("PRECISION ONCOLOGY ENGINE – MODEL 2")
    print("Survival Prognostication (XGBoost + SHAP + Ablation)")
    print("=" * 80)


    try:
        df = pd.read_csv("data/processed_data.csv")
        prob_feat = pd.read_csv("data/predicted_histology_for_model2.csv")
        df = pd.concat([df, prob_feat], axis=1)
        print(f"✓ Integrated Model 1 Probabilities. Total Features: {df.shape[1]}")
    except FileNotFoundError:
        print("❌ Data files not found. Ensure Model 1 has run successfully.")
        return

    target = "Survival_Rate"
    X = df.drop(columns=["Histology_Encoded", target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')

    X_rf = df.drop(columns=["Histology_Encoded", "Survival_Rate", "Predicted_Histology"], errors="ignore")
    y_rf = df["Histology_Encoded"]
    
    _, X_rf_test, _, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)
    
    y_rf_pred = rf.predict(X_rf_test)
    
    tumor_labels = ["Astrocytoma", "Glioblastoma", "Meningioma"]
    print("\nMODEL 1 — CLASSIFICATION REPORT")
    print(classification_report(y_rf_test, y_rf_pred, target_names=tumor_labels))
    
    cm = confusion_matrix(y_rf_test, y_rf_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tumor_labels, yticklabels=tumor_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Tumor Histology Classification")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix_histology.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Confusion matrix saved to plots/")


    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r_sq = r2_score(y_test, y_pred)

    print("\n" + "=" * 80)
    print("MODEL 2 PERFORMANCE METRICS")
    print("=" * 80)
    print(f"RMSE:     {rmse:.2f}%")
    print(f"MAE:      {mae:.2f}%")
    print(f"R² Score: {r_sq:.3f}")


    print("\n" + "=" * 80)
    print("ABLATION STUDY: STACKED VS BASELINE")
    print("=" * 80)
    
    prob_cols = ['Predicted_Histology']
    X_train_base = X_train.drop(columns=prob_cols, errors='ignore')
    X_test_base = X_test.drop(columns=prob_cols, errors='ignore')

    base_model = XGBRegressor(random_state=42)
    base_model.fit(X_train_base, y_train)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_model.predict(X_test_base)))

    print(f"Baseline RMSE (Clinical Only): {base_rmse:.2f}%")
    print(f"Stacked RMSE (With Model 1):   {rmse:.2f}%")
    print(f"✓ Improvement: {base_rmse - rmse:.2f} percentage points")


    print("\n" + "=" * 80)
    print("GENERATING SHAP VISUALS")
    print("=" * 80)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # --- Global Beeswarm Plot ---
    plt.figure(figsize=(12, 8))  # Wider figure to help with labels
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Global Feature Importance – Survival Prediction", pad=20)
    plt.subplots_adjust(left=0.3) # Shift plot right to make room for labels
    plt.savefig("plots/shap_beeswarm_survival.png", dpi=300, bbox_inches="tight")
    plt.close()




    high_risk_idx = np.argmin(y_pred)
    low_risk_idx = np.argmax(y_pred)

    for idx, label in zip([high_risk_idx, low_risk_idx], ["high_risk", "low_risk"]):
        # Fix for potential array format in expected_value
        base_val = explainer.expected_value
        if isinstance(base_val, (np.ndarray, list)):
            base_val = base_val[0]

        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test.iloc[idx],
            feature_names=X_test.columns
        )

        plt.figure(figsize=(14, 7)) 
        shap.plots.waterfall(exp, show=False)
        plt.title(f"{label.replace('_', ' ').title()} Patient Explanation", pad=25)
        

        plt.subplots_adjust(left=0.4) 
        
        plt.savefig(f"plots/shap_waterfall_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("✓ All SHAP visuals and Waterfall plots saved with corrected margins.")
    print("=" * 80)
    print("MODEL 2 EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_precision_oncology_model2()