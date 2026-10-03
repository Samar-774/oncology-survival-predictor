import pandas as pd
import numpy as np

df=pd.read_csv("data/brain_tumor_dataset.csv")

# Mutation rates derived from TCGA GBM/LGG dataset (862 real patients)
# Source: kaggle.com/datasets/tanshihjen/clinical-gliomagrading
# Calculated using: df.groupby('Grade')['GENE'].value_counts(normalize=True)
MUTATION_RATES={
    'I':{

        'IDH1': 0.784,
        'TP53': 0.477,
        'ATRX': 0.371,
        'PTEN': 0.050,
        'EGFR': 0.062,
        'IDH2': 0.042,
    },

    'II':{
        'IDH1': 0.784,
        'TP53': 0.477,
        'ATRX': 0.371,
        'PTEN': 0.050,
        'EGFR': 0.062,
        'IDH2': 0.042,
    },
    'III':{
        'IDH1': 0.704,
        'TP53': 0.415,
        'ATRX': 0.180,
        'PTEN': 0.250,
        'EGFR': 0.162,
        'IDH2': 0.022,        
    },
    'IV':{
        'IDH1': 0.063,
        'TP53': 0.320,
        'ATRX': 0.096,
        'PTEN': 0.328,
        'EGFR': 0.229,
        'IDH2': 0.006,
    }
}
GENES = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'IDH2']
for gene in GENES:
    mutation_col = []
    for index,row in df.iterrows():
        grade=row['Tumor Grade']
        rate = MUTATION_RATES[grade][gene]
        mutated= np.random.random() < rate
        mutation_col.append(1 if mutated else 0)

    df[f'{gene}_Mutated']=mutation_col

df['Survival Time (months)'] = df['Survival Time (months)'] + (df['IDH1_Mutated']*8) - (df['EGFR_Mutated'] * 6) - (df['PTEN_Mutated']*4) + (df['ATRX_Mutated']*3) - (df['TP53_Mutated']*2) + (df['IDH2_Mutated']*3)

df['Survival Time (months)'] = df['Survival Time (months)'].clip(6, 72)

# print(df['Survival Time (months)'].describe())

df.to_csv('data/brain_tumor_augmented.csv', index=False)

print('Operstion succesful')