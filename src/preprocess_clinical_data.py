import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOADING DATA...")
print("="*80)
df = pd.read_csv('brain_tumor_dataset.csv')
print(f"Original dataset shape: {df.shape}")
print(f"\nOriginal columns: {list(df.columns)}\n")

print("="*80)
print("CLEANING DATA...")
print("="*80)
# Fixed: Changed 'Patient_ID' to 'Patient ID' to match your CSV
df = df.drop(columns=['Patient ID'])
print("✓ Dropped Patient ID column")
print(f"Updated shape: {df.shape}\n")

print("="*80)
print("TARGET ENCODING...")
print("="*80)

# Fixed: Changed 'Histology' to 'Tumor Type' based on your dataset columns
histology_encoder = LabelEncoder()
df['Histology_Encoded'] = histology_encoder.fit_transform(df['Tumor Type'])

print("CLASSIFICATION TARGET - Histology Encoding Mapping (from Tumor Type):")
print("-" * 60)
histology_mapping = dict(enumerate(histology_encoder.classes_))
for code, label in histology_mapping.items():
    count = (df['Histology_Encoded'] == code).sum()
    print(f"  {code} = {label:20s} (n={count})")
print()

print("REGRESSION TARGET - Survival_Rate (%)")
print("-" * 60)

survival_col = 'Survival Time (months)'
max_survival = df[survival_col].max()

df['Survival_Rate'] = (df[survival_col] / max_survival) * 100

print(f"  Survival_Rate Range: [{df['Survival_Rate'].min():.2f}, {df['Survival_Rate'].max():.2f}]")
print(f"  Mean: {df['Survival_Rate'].mean():.2f}")
print(f"  Std: {df['Survival_Rate'].std():.2f}")
print()
df = df.drop(columns=['Survival Time (months)'])

df = df.drop(columns=['Tumor Type'])

print("="*80)
print("FEATURE ENGINEERING...")
print("="*80)

# Note: Your dataset log did not show Symptom columns. 
# I am keeping this logic as requested, but added a check to prevent errors if they are missing.
symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3']
if all(col in df.columns for col in symptom_cols):
    print("\n4.1 SYMPTOM AGGREGATION:")
    print("-" * 60)

    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(df[col].unique())

    all_symptoms = sorted(list(all_symptoms))
    print(f"Unique symptoms found: {all_symptoms}\n")

    for symptom in all_symptoms:
        col_name = f'Has_{symptom.replace(" ", "_")}'
        df[col_name] = (
            (df['Symptom_1'] == symptom) | 
            (df['Symptom_2'] == symptom) | 
            (df['Symptom_3'] == symptom)
        ).astype(int)
        print(f"  ✓ Created {col_name:25s} - Positive cases: {df[col_name].sum()}")

    df = df.drop(columns=symptom_cols)
    print("\n✓ Dropped original Symptom columns\n")

print("4.2 ORDINAL ENCODING - Stage (from Tumor Grade):")
print("-" * 60)
# Fixed: Changed 'Stage' to 'Tumor Grade'
stage_mapping = {'Grade I': 1, 'Grade II': 2, 'Grade III': 3, 'Grade IV': 4, 'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df['Stage_Encoded'] = df['Tumor Grade'].map(stage_mapping).fillna(0)

print("Stage Mapping:")
for stage, code in stage_mapping.items():
    count = (df['Stage_Encoded'] == code).sum()
    if count > 0:
        print(f"  {stage:3s} → {code}  (n={count})")
df = df.drop(columns=['Tumor Grade'])
print()

print("4.3 BINARY/CATEGORICAL ENCODING:")
print("-" * 60)

# Updated list to match columns found in your print log
categorical_columns = [
    'Gender', 
    'Tumor Location',
    'Treatment',
    'Treatment Outcome',
    'Recurrence Site'
]

encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col.replace(" ", "_")}_Encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        mapping = dict(enumerate(le.classes_))
        print(f"\n{col}:")
        for code, label in mapping.items():
            count = (df[f'{col.replace(" ", "_")}_Encoded'] == code).sum()
            print(f"  {code} = {label:20s} (n={count})")

        df = df.drop(columns=[col])
print()
# HANDLE MISSING VALUES
print("4.3.1 HANDLING MISSING VALUES:")
print("-" * 60)

if 'Time to Recurrence (months)' in df.columns:
    missing_count = df['Time to Recurrence (months)'].isnull().sum()
    print(f"Missing values in Time to Recurrence: {missing_count}")

    # Replace missing recurrence time with 0 (means no recurrence)
    df['Time to Recurrence (months)'] = df['Time to Recurrence (months)'].fillna(0)

    print("✓ Filled missing recurrence times with 0\n")
print()

print("4.4 NORMALIZATION - Numerical Features:")
print("-" * 60)

# Updated to match your actual numeric columns
numerical_columns = ['Age', 'Time to Recurrence (months)']
numerical_columns = [c for c in numerical_columns if c in df.columns]

scaler = StandardScaler()

if numerical_columns:
    print("Before normalization:")
    for col in numerical_columns:
        print(f"  {col:20s} - Mean: {df[col].mean():8.3f}, Std: {df[col].std():8.3f}")
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    print("\nAfter normalization (StandardScaler):")
    for col in numerical_columns:
        print(f"  {col:20s} - Mean: {df[col].mean():8.3f}, Std: {df[col].std():8.3f}")

print()
print("="*80)
print("SAVING PROCESSED DATA...")
print("="*80)

output_path = 'processed_data.csv'
df.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")
print(f"Final shape: {df.shape}\n")

print("="*80)
print("VERIFICATION - FINAL DATASET INFO")
print("="*80)
print("\nDataFrame Info:")
print("-" * 60)
df.info()

print("\n" + "="*80)
print("FIRST 5 ROWS OF PROCESSED DATA:")
print("="*80)
print(df.head())

print("\n" + "="*80)
print("DATA TYPES CHECK:")
print("="*80)
print(df.dtypes.value_counts())

print("\n" + "="*80)
print("NULL VALUES CHECK:")
print("="*80)
print(f"Total null values: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() == 0:
    print("✓ NO NULL VALUES - Data is clean!")
else:
    print("⚠ Warning: Null values detected")
    print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n" + "="*80)
print("NON-NUMERIC COLUMNS CHECK:")
print("="*80)
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric) == 0:
    print("✓ ALL COLUMNS ARE NUMERIC - Ready for ML models!")
else:
    print(f"⚠ Warning: Non-numeric columns found: {non_numeric}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("\nSummary:")
print(f"  - Original rows: {df.shape[0]}")
print(f"  - Final features: {df.shape[1]}")
print(f"  - Classification target: Histology_Encoded")
print(f"  - Regression target: Survival_Rate (%)")
print(f"  - All features are numeric and ready for modeling")
print("="*80)


print(df[['Survival_Rate']].describe())
