import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

from pipeline import predict_patient

st.set_page_config(page_title="Oncology Survival Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Oncology Survival Predictor")
st.markdown("A clinical ML pipeline for **brain tumor classification** and **survival prediction**, powered by Explainable AI (SHAP).")
st.divider()

# ─── Constants ───────────────────────────────────────────────────────
GENDER_OPTIONS = ['Female', 'Male']
LOCATION_OPTIONS = ['Frontal lobe', 'Occipital lobe', 'Parietal lobe', 'Temporal lobe']
TREATMENT_OPTIONS = [
    'Chemotherapy', 'Chemotherapy + Radiation', 'Radiation',
    'Surgery', 'Surgery + Chemotherapy', 'Surgery + Radiation',
    'Surgery + Radiation therapy'
]
OUTCOME_OPTIONS = ['Complete response', 'Partial response', 'Progressive disease', 'Stable disease']
RECURRENCE_OPTIONS = ['Frontal lobe', 'None', 'Occipital lobe', 'Parietal lobe', 'Temporal lobe']

# ─── Load artifacts for encoders/scaler (needed for form validation) ─
@st.cache_resource
def load_artifacts():
    encoders = joblib.load('models/encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    histology_mapping = joblib.load('models/histology_mapping.pkl')
    max_survival = joblib.load('models/max_survival.pkl')
    return encoders, scaler, histology_mapping, max_survival

with st.spinner("Loading models..."):
    encoders, scaler, histology_mapping, max_survival = load_artifacts()

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
        result = predict_patient(
            age, gender, tumor_grade, tumor_location,
            treatment, treatment_outcome, recurrence_site,
            recurrence_time, idh1, idh2, tp53, atrx, pten, egfr
        )

    # ─── Results ─────────────────────────────────────────────────
    st.subheader("📊 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("🧬 Tumor Type", result['tumor_type'])
    r2.metric("📅 Survival Horizon", f"{result['survival_months']:.1f} months")
    r3.metric("🔍 Primary Driver", result['top_feature'])

    st.divider()

    # Confidence breakdown
    st.subheader("🎯 Classification Confidence")
    conf_df = pd.DataFrame({
        'Tumor Type': list(result['histology_mapping'].values()),
        'Confidence': [f"{p*100:.1f}%" for p in result['predicted_proba']]
    })
    st.dataframe(conf_df, hide_index=True, use_container_width=True)

    st.divider()

    # SHAP explanation text
    st.subheader("🧠 SHAP Explanation")
    st.markdown(f"""
> **For this patient**, the model predicts **{result['tumor_type']}** with an estimated survival horizon of **{result['survival_months']:.1f} months**.
>
> The primary survival driver is **{result['top_feature']}**, which has a **{result['direction'].lower()} influence** on survival.
    """)
    st.info(f"ℹ️ **{result['top_feature']}** {'increases' if result['direction'] == 'Positive' else 'decreases'} predicted survival for this patient.")

    # RF SHAP bar chart
    st.divider()
    st.subheader("📊 Tumor Classification — Feature Impact")
    st.caption("How much each feature pushed the tumor type prediction.")

    clean_rf_features = [f.replace('_Encoded','').replace('_',' ').title() for f in result['rf_features']]
    rf_shap_df = pd.DataFrame({
        'Feature': clean_rf_features,
        'SHAP Value': result['rf_shap']
    }).sort_values('SHAP Value', key=abs, ascending=True)

    rf_colors = ['#e74c3c' if v > 0 else '#3498db' for v in rf_shap_df['SHAP Value']]
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.barh(rf_shap_df['Feature'], rf_shap_df['SHAP Value'], color=rf_colors)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_xlabel('SHAP Value (impact on classification)')
    ax1.set_title(f'Feature Impact for Predicted Class: {result["tumor_type"]}')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)
    st.caption("🔴 Red = pushes toward this tumor type | 🔵 Blue = pushes away")

    # XGB SHAP bar chart
    st.divider()
    st.subheader("📈 Survival Prediction — Feature Impact")
    st.caption("How much each feature influenced the survival prediction.")

    clean_xgb_features = [f.replace('_Encoded','').replace('_',' ').title() for f in result['xgb_features']]
    xgb_shap_df = pd.DataFrame({
        'Feature': clean_xgb_features,
        'SHAP Value': result['xgb_shap']
    }).sort_values('SHAP Value', key=abs, ascending=True)

    xgb_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in xgb_shap_df['SHAP Value']]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(xgb_shap_df['Feature'], xgb_shap_df['SHAP Value'], color=xgb_colors)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('SHAP Value (impact on survival)')
    ax2.set_title('Survival Prediction — Feature Impact')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("🟢 Green = increases survival | 🔴 Red = decreases survival")

    # Risk assessment
    st.divider()
    st.subheader("⚠️ Risk Assessment")
    if result['risk'] == "High":
        st.error(f"🔴 High Risk — Predicted survival of {result['survival_months']:.1f} months. Aggressive intervention recommended.")
    elif result['risk'] == "Moderate":
        st.warning(f"🟡 Moderate Risk — Predicted survival of {result['survival_months']:.1f} months. Close monitoring advised.")
    else:
        st.success(f"🟢 Lower Risk — Predicted survival of {result['survival_months']:.1f} months. Standard care protocol.")

st.divider()
st.caption("Built with Scikit-learn · XGBoost · SHAP · Streamlit | Data: Brain Tumor Clinical Dataset")