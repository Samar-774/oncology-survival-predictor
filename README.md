# 🧠 Oncology Survival Predictor

A clinical Machine Learning pipeline that assists oncologists in **brain tumor classification** and **survival rate prediction**, powered by Explainable AI (SHAP).

🚀 **Live App:** [oncology-survival-predictor-project.streamlit.app](https://oncology-survival-predictor-project.streamlit.app)

---

## 🔍 What It Does

Given a patient's clinical and genomic data, the engine:

1. **Classifies** the tumor histology type (Astrocytoma, Glioblastoma, or Meningioma)
2. **Predicts** the patient's survival horizon in months
3. **Explains** the prediction using SHAP — identifying the primary driver for that specific patient
4. **Assesses risk** — High / Moderate / Low with clinical recommendations

---

## 🏗️ Pipeline Architecture

```
Raw Clinical Data + Genomic Mutation Features
               ↓
         Preprocessing
               ↓
Model 1: Random Forest (Histology Classification)
               ↓
Model 2: XGBoost (Survival Prediction) ← uses Model 1 output as a feature
               ↓
      SHAP Explainability Layer
               ↓
    Patient Report + Risk Assessment
```

---

## 🧬 Dataset

The dataset combines two sources:

- **Synthetic clinical dataset** — 2000 patients with age, tumor grade, treatment, recurrence, and survival time
- **TCGA GBM/LGG dataset** — 862 real patients used to derive genomic mutation rates

Mutation features (IDH1, IDH2, TP53, ATRX, PTEN, EGFR) were added to the clinical dataset using real TCGA-derived probabilities per tumor grade. Survival times were adjusted based on known mutation impact:

| Gene | Effect | Monthly Impact |
|---|---|---|
| IDH1 | Protective | +8 months |
| ATRX | Slightly protective | +3 months |
| IDH2 | Slightly protective | +3 months |
| TP53 | Risk factor | -2 months |
| PTEN | Risk factor | -4 months |
| EGFR | Risk factor | -6 months |

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
├── Histology:  Glioblastoma
├── Survival:   37.8 months
├── Risk:       🟡 Moderate
└── XAI Driver: IDH1 Mutation (Protective Influence)
```
*"While tumor stage is generally the strongest predictor, for this patient, IDH1 mutation is a protective factor — the model is optimistic because of it."*

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
# Step 1 — Augment dataset with mutation features
python augment_dataset.py

# Step 2 — Preprocess augmented data
python src/preprocess_clinical_data.py

# Step 3 — Run Random Forest (Histology Classification)
python src/precision_onlogy_rf.py

# Step 4 — Run XGBoost (Survival Prediction)
python src/precision_onlogy_xgb.py

# Step 5 — Launch Streamlit app
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **ML Models:** Scikit-learn (Random Forest), XGBoost
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
oncology-survival-predictor/
├── data/                          # Datasets (gitignored)
├── src/
│   ├── preprocess_clinical_data.py
│   ├── precision_onlogy_rf.py
│   ├── precision_onlogy_xgb.py
│   └── evaluate_models.py
├── plots/                         # Generated SHAP visualizations
├── augment_dataset.py             # Dataset augmentation script
├── app.py                         # Streamlit application
├── requirements.txt
└── README.md
```

---