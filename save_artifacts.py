import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv('data/brain_tumor_augmented.csv')
df = df.drop(columns=['Patient ID'])
df['Recurrence Site'] = df['Recurrence Site'].fillna('None')

histology_encoder = LabelEncoder()
df['Histology_Encoded'] = histology_encoder.fit_transform(df['Tumor Type'])
histology_mapping = dict(enumerate(histology_encoder.classes_))

max_survival = df['Survival Time (months)'].max()

GRADE_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df['Stage_Encoded'] = df['Tumor Grade'].map(GRADE_MAP)
df = df.drop(columns=['Tumor Grade', 'Tumor Type', 'Survival Time (months)'])

categorical_columns = ['Gender', 'Tumor Location', 'Treatment', 'Treatment Outcome', 'Recurrence Site']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    col_encoded = col.replace(' ', '_') + '_Encoded'
    df[col_encoded] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    df = df.drop(columns=[col])

df['Time to Recurrence (months)'] = df['Time to Recurrence (months)'].fillna(0)
scaler = StandardScaler()
scaler.fit(df[['Age', 'Time to Recurrence (months)']])

joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(histology_mapping, 'models/histology_mapping.pkl')
joblib.dump(max_survival, 'models/max_survival.pkl')
print('All artifacts saved!')