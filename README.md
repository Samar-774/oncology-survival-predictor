# 🧠 Oncology Survival Predictor

A clinical Machine Learning pipeline that assists oncologists in **brain tumor classification** and **survival rate prediction**, powered by Explainable AI (SHAP).

---

## 🔍 What It Does

Given a patient's clinical data (age, tumor stage, treatment history), the engine:

1. **Classifies** the tumor histology type (Astrocytoma, Glioblastoma, or Meningioma)
2. **Predicts** the patient's survival horizon in months
3. **Explains** the prediction using SHAP — identifying the primary driver for that specific patient

---

## 🏗️ Pipeline Architecture

```
Raw Clinical Data
       ↓
  Preprocessing
       ↓
Model 1: Random Forest (Histology Classification)
       ↓
Model 2: XGBoost (Survival Prediction) ← uses Model 1 output as a feature
       ↓
SHAP Explainability Layer
       ↓
Patient Report
```

---

## 📊 Model Performance

| Model | Task | Metric | Score |
|---|---|---|---|
| Random Forest | Histology Classification | Accuracy | 99% |
| XGBoost | Survival Prediction | R² | 0.813 |
| XGBoost | Survival Prediction | RMSE | 5.25% |

### Stacked Architecture Advantage
| Approach | RMSE |
|---|---|
| Clinical features only | 6.07% |
| Stacked (with Model 1 output) | 5.25% |

The 2-stage stacked pipeline reduces error by **0.81 percentage points** over a baseline clinical-only model.

---

## 🔬 Explainability — SHAP

Standard ML models are black boxes. In oncology, an unexplained prediction is dangerous.

SHAP (SHapley Additive exPlanations) makes the model a **glass box**:

- **Global:** Which features matter most across the entire dataset
- **Local:** For a specific patient, which feature was the actual primary driver

### Example Patient Report
```
REPORT FOR PATIENT ID: 1333
├── Histology:  Group 1 (Glioblastoma)
├── Survival:   37.8 months
└── XAI Driver: Age (Positive Influence)
```
*"While tumor stage is generally the strongest predictor, for this patient, age is a protective factor — the model is optimistic because of it."*

---

## 📈 Visual Results

| Plot | Description |
|---|---|
| `confusion_matrix_histology.png` | Model 1 classification accuracy per tumor type |
| `shap_beeswarm_survival.png` | Global feature importance for survival prediction |
| `shap_waterfall_high_risk.png` | SHAP explanation for a high-risk patient |
| `shap_waterfall_low_risk.png` | SHAP explanation for a low-risk patient |

---

## 🚀 How to Run

```bash
# Step 1 — Preprocess raw data
python preprocess_clinical_data.py

# Step 2 — Run Random Forest (Histology Classification)
python precision_onlogy_rf.py

# Step 3 — Run XGBoost (Survival Prediction)
python precision_onlogy_xgb.py
```

---

## 🛠️ Tech Stack

- **ML Models:** Scikit-learn (Random Forest), XGBoost
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---