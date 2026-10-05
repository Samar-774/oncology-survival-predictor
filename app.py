import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Oncology Survival Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Oncology Survival Predictor")
st.markdown("A clinical ML pipeline for **brain tumor classification** and **survival prediction**, powered by Explainable AI (SHAP).")
st.divider()

# ─── Constants (matching dataset exactly) ────────────────────────────
GRADE_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
TUMOR_TYPES = {0: 'Astrocytoma', 1: 'Glioblastoma', 2: 'Meningioma'}

GENDER_OPTIONS = ['Female', 'Male']
LOCATION_OPTIONS = ['Frontal lobe', 'Occipital lobe', 'Parietal lobe', 'Temporal lobe']
TREATMENT_OPTIONS = [
    'Chemotherapy', 'Chemotherapy + Radiation', 'Radiation',
    'Surgery', 'Surgery + Chemotherapy', 'Surgery + Radiation',
    'Surgery + Radiation therapy'
]
OUTCOME_OPTIONS = ['Complete response', 'Partial response', 'Progressive disease', 'Stable disease']
RECURRENCE_OPTIONS = ['Frontal lobe', 'None', 'Occipital lobe', 'Parietal lobe', 'Temporal lobe']

# ─── Load & preprocess ───────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("data/brain_tumor_augmented.csv")
    df = df.drop(columns=['Patient ID'])

    # Handle missing recurrence site
    df['Recurrence Site'] = df['Recurrence Site'].fillna('None')

    # Target 1: Histology classification
    histology_encoder = LabelEncoder()
    df['Histology_Encoded'] = histology_encoder.fit_transform(df['Tumor Type'])
    histology_mapping = dict(enumerate(histology_encoder.classes_))
    df = df.drop(columns=['Tumor Type'])

    # Target 2: Survival in months (keep raw, not normalized)
    max_survival = df['Survival Time (months)'].max()
    df['Survival_Rate'] = (df['Survival Time (months)'] / max_survival) * 100
    df = df.drop(columns=['Survival Time (months)'])

    # Stage encoding - dataset uses 'I','II','III','IV' without Grade prefix
    df['Stage_Encoded'] = df['Tumor Grade'].map(GRADE_MAP).fillna(0).astype(int)
    df = df.drop(columns=['Tumor Grade'])

    # Categorical encoding
    categorical_columns = ['Gender', 'Tumor Location', 'Treatment', 'Treatment Outcome', 'Recurrence Site']
    encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col.replace(" ", "_")}_Encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df = df.drop(columns=[col])

    # Fill missing recurrence time
    if 'Time to Recurrence (months)' in df.columns:
        df['Time to Recurrence (months)'] = df['Time to Recurrence (months)'].fillna(0)

    # Scale numerical columns — fit scaler and return it
    numerical_columns = ['Age', 'Time to Recurrence (months)']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, histology_mapping, encoders, scaler, max_survival

@st.cache_resource
def load_models():
    rf = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    rf_features = joblib.load('models/rf_features.pkl')
    xgb_features = joblib.load('models/xgb_features.pkl')
    return rf, xgb_model, rf_features, xgb_features

# ─── Load everything ─────────────────────────────────────────────────
with st.spinner("Loading data and models..."):
    df, histology_mapping, encoders, scaler, max_survival = load_and_preprocess()
    rf, xgb_model, rf_features, xgb_features = load_models()

st.success("✅ Models ready! Fill in patient details below.")
st.divider()

# ─── Patient Input Form ──────────────────────────────────────────────
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=90, value=50)
    gender = st.selectbox("Gender", GENDER_OPTIONS)
    tumor_grade = st.selectbox("Tumor Grade", ['I', 'II', 'III', 'IV'])
    tumor_location = st.selectbox("Tumor Location", LOCATION_OPTIONS)

with col2:
    treatment = st.selectbox("Treatment", TREATMENT_OPTIONS)
    treatment_outcome = st.selectbox("Treatment Outcome", OUTCOME_OPTIONS)
    recurrence_site = st.selectbox("Recurrence Site (None if no recurrence)", RECURRENCE_OPTIONS)
    recurrence_time = st.slider("Time to Recurrence (months, 0 if none)", min_value=0, max_value=60, value=0)


st.subheader("🧬 Genomic Mutation Status")
st.caption("Select mutations present in the patient's tumor biopsy report")

mut_col1, mut_col2, mut_col3 = st.columns(3)

with mut_col1:
    idh1 = st.toggle("IDH1 Mutated", value=False)
    idh2 = st.toggle("IDH2 Mutated", value=False)

with mut_col2:
    tp53 = st.toggle("TP53 Mutated", value=False)
    atrx = st.toggle("ATRX Mutated", value=False)

with mut_col3:
    pten = st.toggle("PTEN Mutated", value=False)
    egfr = st.toggle("EGFR Mutated", value=False)

st.divider()

if st.button("🔬 Run Prediction", use_container_width=True):
    with st.spinner("Running prediction..."):

        # Scale age and recurrence time using the SAME scaler fit on training data
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

        # Build RF patient row (exact column order matters)
        patient_rf = pd.DataFrame([[
            scaled_age, scaled_recurrence,
            int(idh1), int(tp53), int(atrx), int(pten), int(egfr), int(idh2),
            stage_enc, gender_enc, location_enc, treatment_enc, outcome_enc, recurrence_site_enc
        ]], columns=rf_features)

        # RF prediction
        predicted_class = rf.predict(patient_rf)[0]
        tumor_type = histology_mapping[predicted_class]
        predicted_proba = rf.predict_proba(patient_rf)[0]

        # XGB patient row (includes predicted histology)
        patient_xgb = patient_rf.copy()
        patient_xgb['Predicted_Histology'] = predicted_class
        patient_xgb = patient_xgb[xgb_features]

        # XGB survival prediction
        survival_rate = xgb_model.predict(patient_xgb)[0]
        survival_months = (np.clip(survival_rate, 0, 100) / 100) * max_survival

        # SHAP for RF
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(patient_rf)

        # Handle both 3D and list formats
        if isinstance(shap_values, list):
            patient_shap = shap_values[int(predicted_class)][0]
        elif shap_values.ndim == 3:
            patient_shap = shap_values[0, :, int(predicted_class)]
        else:
            patient_shap = shap_values[0]

        # Use XGB SHAP for primary driver (survival is the main output)
        xgb_explainer = shap.TreeExplainer(xgb_model)
        xgb_shap_values = xgb_explainer.shap_values(patient_xgb)

        top_idx = np.argmax(np.abs(xgb_shap_values[0]))
        top_feature = xgb_features[top_idx]
        direction = "Positive" if xgb_shap_values[0][top_idx] > 0 else "Negative"
        top_feature_clean = top_feature.replace('_Encoded', '').replace('_', ' ').title()

    # ─── Results ─────────────────────────────────────────────────
    st.subheader("📊 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("🧬 Tumor Type", tumor_type)
    r2.metric("📅 Survival Horizon", f"{survival_months:.1f} months")
    r3.metric("🔍 Primary Driver", top_feature_clean)

    st.divider()

    # Confidence breakdown
    st.subheader("🎯 Classification Confidence")
    conf_df = pd.DataFrame({
        'Tumor Type': list(histology_mapping.values()),
        'Confidence': [f"{p*100:.1f}%" for p in predicted_proba]
    })
    st.dataframe(conf_df, hide_index=True, use_container_width=True)

    st.divider()

    # SHAP explanation
    st.subheader("🧠 SHAP Explanation")
    st.markdown(f"""
> **For this patient**, the model predicts **{tumor_type}** with an estimated survival horizon of **{survival_months:.1f} months**.
>
> The primary driver behind this prediction is **{top_feature_clean}**, which has a **{direction.lower()} influence** on the outcome.
    """)

    if direction == "Positive":
        st.info(f"ℹ️ **{top_feature_clean}** pushed the classification **toward {tumor_type}**.")
    else:
        st.info(f"ℹ️ **{top_feature_clean}** pushed the classification **away from {tumor_type}**.")

    # SHAP bar chart
    st.divider()
    st.subheader("📊 SHAP Feature Impact")
    st.caption("How much each feature pushed the prediction — positive = toward this tumor type, negative = away from it.")

    clean_features = [f.replace('_Encoded','').replace('_',' ').title() for f in rf_features]
    shap_df = pd.DataFrame({
        'Feature': clean_features,
        'SHAP Value': patient_shap
    }).sort_values('SHAP Value', key=abs, ascending=True)

    colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df['SHAP Value']]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value (impact on prediction)')
    ax.set_title(f'Feature Impact for Predicted Class: {tumor_type}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("🔴 Red = pushes toward this tumor type | 🔵 Blue = pushes away from this tumor type")

    # XGB SHAP — Survival explanation
    st.divider()
    st.subheader("📈 Survival Prediction Drivers")
    st.caption("How much each feature influenced the survival prediction specifically.")

    xgb_clean_features = [f.replace('_Encoded','').replace('_',' ').title() for f in xgb_features]
    xgb_shap_df = pd.DataFrame({
        'Feature': xgb_clean_features,
        'SHAP Value': xgb_shap_values[0]
    }).sort_values('SHAP Value', key=abs, ascending=True)

    xgb_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in xgb_shap_df['SHAP Value']]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(xgb_shap_df['Feature'], xgb_shap_df['SHAP Value'], color=xgb_colors)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('SHAP Value (impact on survival prediction)')
    ax2.set_title('Survival Prediction — Feature Impact')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("🟢 Green = increases survival | 🔴 Red = decreases survival")

    # Risk indicator
    st.divider()
    st.subheader("⚠️ Risk Assessment")
    if survival_months < 20:
        st.error(f"🔴 High Risk — Predicted survival of {survival_months:.1f} months. Aggressive intervention recommended.")
    elif survival_months < 36:
        st.warning(f"🟡 Moderate Risk — Predicted survival of {survival_months:.1f} months. Close monitoring advised.")
    else:
        st.success(f"🟢 Lower Risk — Predicted survival of {survival_months:.1f} months. Standard care protocol.")

st.divider()
st.caption("Built with Scikit-learn · XGBoost · SHAP · Streamlit | Data: Brain Tumor Clinical Dataset")