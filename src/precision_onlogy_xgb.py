import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore")


def run_precision_oncology_model2():
    df = pd.read_csv("data/processed_data.csv")
    hist_pred = pd.read_csv("data/predicted_histology_for_model2.csv")

    df["Predicted_Histology"] = hist_pred["Predicted_Histology"]

    target = "Survival_Rate"

    X = df.drop(columns=["Histology_Encoded", target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, 'models/xgb_model.pkl')
    joblib.dump(list(X.columns), 'models/xgb_features.pkl')
    print("✓ XGB model saved")
    print("✓ XGB features saved")
    print("✓ Model 2 complete")


if __name__ == "__main__":
    run_precision_oncology_model2()