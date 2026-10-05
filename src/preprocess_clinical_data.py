import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/brain_tumor_augmented.csv')

df = df.drop(columns=['Patient ID'])

histology_encoder = LabelEncoder()
df['Histology_Encoded'] = histology_encoder.fit_transform(df['Tumor Type'])
histology_mapping = dict(enumerate(histology_encoder.classes_))


survival_col = 'Survival Time (months)'
max_survival = df[survival_col].max()
df['Survival_Rate'] = (df[survival_col] / max_survival) * 100

df = df.drop(columns=['Survival Time (months)'])
df = df.drop(columns=['Tumor Type'])

stage_mapping = {'Grade I': 1, 'Grade II': 2, 'Grade III': 3, 'Grade IV': 4, 'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df['Stage_Encoded'] = df['Tumor Grade'].map(stage_mapping).fillna(0)

df = df.drop(columns=['Tumor Grade'])

categorical_columns = [
    'Gender', 
    'Tumor Location',
    'Treatment',
    'Treatment Outcome',
    'Recurrence Site'
]
encoders={}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col.replace(" ", "_")}_Encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        df = df.drop(columns=[col])

if 'Time to Recurrence (months)' in df.columns:
    df['Time to Recurrence (months)'] = df['Time to Recurrence (months)'].fillna(0)

numerical_columns = ['Age', 'Time to Recurrence (months)']
numerical_columns = [c for c in numerical_columns if c in df.columns]

scaler = StandardScaler()

if numerical_columns:
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

output_path = 'data/processed_data.csv'
df.to_csv(output_path, index=False)