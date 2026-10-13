# 🧠 Oncology Survival Predictor

A clinical Machine Learning pipeline that classifies brain tumor histology and predicts patient survival horizons, powered by Explainable AI (SHAP).

🚀 **Live App:** [oncology-survival-predictor-project.streamlit.app](https://oncology-survival-predictor-project.streamlit.app)

---

## 🔍 What It Does

Given a patient's clinical and genomic data, the engine:

1. **Classifies** tumor histology (Astrocytoma, Glioblastoma, or Meningioma) using Random Forest
2. **Predicts** survival horizon in months using XGBoost (stacked on RF output)
3. **Explains** both predictions using SHAP — two separate charts for classification and survival
4. **Assesses risk** — High / Moderate / Low with clinical recommendations

---

## 🏗️ Pipeline Architecture

```
brain_tumor_dataset.csv
        ↓
augment_dataset.py               → adds TCGA-derived mutation features + survival adjustments
        ↓
brain_tumor_augmented.csv
        ↓
src/preprocess_clinical_data.py  → encodes, scales, saves processed_data.csv
        ↓
src/precision_onlogy_rf.py       → trains Random Forest, saves rf_model.pkl + rf_features.pkl
        ↓
src/precision_onlogy_xgb.py      → trains XGBoost, saves xgb_model.pkl + xgb_features.pkl
        ↓
save_artifacts.py                → saves encoders, scaler, histology_mapping, max_survival
        ↓
app.py                           → loads all pkl files, serves predictions via Streamlit
```

---

## 🧬 Dataset

The dataset combines two sources:

- **Synthetic clinical dataset** — 2000 patients with age, tumor grade, treatment, recurrence and survival time
- **TCGA GBM/LGG dataset** — 862 real patients used to derive genomic mutation rates per grade

Mutation features (IDH1, IDH2, TP53, ATRX, PTEN, EGFR) were added using real TCGA-derived probabilities. Survival times were adjusted based on known clinical factors:

| Factor | Impact |
|---|---|
| IDH1 mutated | +8 months (protective) |
| ATRX mutated | +3 months (protective) |
| IDH2 mutated | +3 months (protective) |
| TP53 mutated | -2 months (risk) |
| PTEN mutated | -4 months (risk) |
| EGFR mutated | -6 months (risk) |
| Age (per year above mean) | -0.8 months |
| Grade I | +8 months |
| Grade II | +4 months |
| Grade III | -4 months |
| Grade IV | -12 months |
| No recurrence | +5 months |
| Early recurrence (<12m) | -8 months |

---

## 📊 Model Performance

| Model | Task | Metric | Score |
|---|---|---|---|
| Random Forest | Histology Classification | Accuracy | 99.75% |
| XGBoost | Survival Prediction | R² | 0.954 |
| XGBoost | Survival Prediction | RMSE | 5.46% |
| XGBoost | Survival Prediction | MAE | 3.21% |

### Stacked Architecture Advantage
| Approach | RMSE |
|---|---|
| Clinical features only | 5.82% |
| Stacked (with RF output) | 5.46% |

The 2-stage stacked pipeline reduces RMSE by **0.36 percentage points** over a baseline clinical-only model.

---

## 🔬 Explainability — SHAP

Two SHAP charts per prediction:

- **Classification SHAP** — which features drove the tumor type prediction
- **Survival SHAP** — which features drove the survival estimate (green = increases survival, red = decreases)

### Example Output
```
Patient: 35 year old, Grade II, IDH1 mutated
├── Tumor Type:      Astrocytoma (87% confidence)
├── Survival:        48.3 months
├── Primary Driver:  IDH1 Mutation (protective)
└── Risk:            🟢 Lower Risk
```

---

## 📈 SHAP Visualizations

| Plot | Description |
|---|---|
| `plots/confusion_matrix_histology.png` | RF classification accuracy per tumor type |
| `plots/shap_beeswarm_survival.png` | Global feature importance for survival prediction |
| `plots/shap_waterfall_high_risk.png` | SHAP explanation for highest risk patient |
| `plots/shap_waterfall_low_risk.png` | SHAP explanation for lowest risk patient |

---

## 🚀 How to Run Locally

```bash
# Step 1 — Augment dataset with TCGA mutation features
python augment_dataset.py

# Step 2 — Preprocess augmented data
python src/preprocess_clinical_data.py

# Step 3 — Train and save models
python src/precision_onlogy_rf.py
python src/precision_onlogy_xgb.py

# Step 4 — Save encoders and artifacts
python save_artifacts.py

# Step 5 — (Optional) Evaluate models and generate SHAP plots
python src/evaluate_models.py

# Step 6 — Launch app
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
- **Model Serialization:** Joblib

---

## 📁 Project Structure

```
oncology-survival-predictor/
├── data/                          # Datasets (gitignored)
├── models/                        # Trained model pkl files
├── src/
│   ├── preprocess_clinical_data.py
│   ├── precision_onlogy_rf.py
│   ├── precision_onlogy_xgb.py
│   └── evaluate_models.py
├── plots/                         # Generated SHAP visualizations
├── augment_dataset.py
├── save_artifacts.py
├── app.py
├── requirements.txt
└── README.md
├── pipeline.py                    # Prediction logic called by app.py
```

---

## ⚠️ Known Limitations

- The base dataset is synthetic — real clinical data (TCGA/SEER) would improve model generalizability
- Mutation features are probabilistically assigned based on TCGA rates, not sequenced from real patients
- A production system would require validation on real hospital data before clinical use

---