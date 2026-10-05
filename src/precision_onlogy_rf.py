import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


def run_precision_oncology_model1():

    try:
        df = pd.read_csv("data/processed_data.csv")
    except FileNotFoundError:
        print("❌ processed_data.csv not found. Run preprocessing first.")
        return

    class_target = "Histology_Encoded"

    if class_target not in df.columns:
        raise ValueError("Histology_Encoded not found in dataset.")

    X = df.drop(columns=[class_target, "Survival_Rate"], errors="ignore")
    y = df[class_target]

    
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    rf_classifier.fit(X_train, y_train)

    joblib.dump(rf_classifier, 'models/rf_model.pkl')
    joblib.dump(list(X.columns), 'models/rf_features.pkl')
    print("✓ RF model saved")
    print("✓ RF features saved")


    df["Predicted_Histology"] = rf_classifier.predict(X)

    df[["Predicted_Histology"]].to_csv("data/predicted_histology_for_model2.csv", index=False)
    print("✓ Predicted histology saved for Model 2")
    print("✓ Model 1 complete")

if __name__ == "__main__":
    run_precision_oncology_model1()