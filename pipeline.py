import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import shap
import joblib

def predict_patient(age, gender, tumor_grade, tumor_location, 
                    treatment, treatment_outcome, recurrence_site, 
                    recurrence_time, idh1, idh2, tp53, atrx, pten, egfr):
    
    # Load models and artifacts
    rf = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    rf_features = joblib.load('models/rf_features.pkl')
    xgb_features = joblib.load('models/xgb_features.pkl')
    encoders = joblib.load('models/encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    histology_mapping = joblib.load('models/histology_mapping.pkl')
    max_survival = joblib.load('models/max_survival.pkl')
    
    GRADE_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    # Scale numerical inputs
    scaled_vals = scaler.transform([[age, recurrence_time]])[0]
    scaled_age = scaled_vals[0]
    scaled_recurrence = scaled_vals[1]

    # Encode categorical inputs
    gender_enc = encoders['Gender'].transform([gender])[0]
    location_enc = encoders['Tumor Location'].transform([tumor_location])[0]
    treatment_enc = encoders['Treatment'].transform([treatment])[0]
    outcome_enc = encoders['Treatment Outcome'].transform([treatment_outcome])[0]
    recurrence_site_enc = encoders['Recurrence Site'].transform([recurrence_site])[0]
    stage_enc = GRADE_MAP[tumor_grade]

    # Build RF patient row
    patient_rf = pd.DataFrame([[
        scaled_age, scaled_recurrence,
        int(idh1), int(tp53), int(atrx), int(pten), int(egfr), int(idh2),
        stage_enc, gender_enc, location_enc, treatment_enc, outcome_enc, recurrence_site_enc
    ]], columns=rf_features)

    # RF prediction
    predicted_class = rf.predict(patient_rf)[0]
    tumor_type = histology_mapping[predicted_class]
    predicted_proba = rf.predict_proba(patient_rf)[0]

    # XGB patient row
    patient_xgb = patient_rf.copy()
    patient_xgb['Predicted_Histology'] = predicted_class
    patient_xgb = patient_xgb[xgb_features]

    # XGB survival prediction
    survival_rate = xgb_model.predict(patient_xgb)[0]
    survival_months = (np.clip(survival_rate, 0, 100) / 100) * max_survival

    # SHAP for RF classification
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap_values = rf_explainer.shap_values(patient_rf)
    if isinstance(rf_shap_values, list):
        patient_rf_shap = rf_shap_values[int(predicted_class)][0]
    elif rf_shap_values.ndim == 3:
        patient_rf_shap = rf_shap_values[0, :, int(predicted_class)]
    else:
        patient_rf_shap = rf_shap_values[0]

    # SHAP for XGB survival
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(patient_xgb)

    # Primary driver from XGB SHAP
    top_idx = np.argmax(np.abs(xgb_shap_values[0]))
    top_feature = xgb_features[top_idx]
    direction = "Positive" if xgb_shap_values[0][top_idx] > 0 else "Negative"
    top_feature_clean = top_feature.replace('_Encoded', '').replace('_', ' ').title()

    # Risk assessment
    if survival_months < 20:
        risk = "High"
    elif survival_months < 36:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        'tumor_type': tumor_type,
        'survival_months': survival_months,
        'predicted_proba': predicted_proba,
        'histology_mapping': histology_mapping,
        'rf_shap': patient_rf_shap,
        'xgb_shap': xgb_shap_values[0],
        'rf_features': rf_features,
        'xgb_features': xgb_features,
        'top_feature': top_feature_clean,
        'direction': direction,
        'risk': risk
    }