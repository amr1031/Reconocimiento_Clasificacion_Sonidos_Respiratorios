#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train_and_dump.py

import os, glob
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1) Rutas ---
AUDIO_DIR = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
CSV_PATH  = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
OUT_DIR   = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

# --- 2) Cargo diagnósticos y encoder ---
df = pd.read_csv(CSV_PATH, header=None, names=['patient_id','diagnosis'])
le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])

# --- 3) Función para extraer los 4 "pct" del CNN previamente entrenado ---
# (aquí debes tener sample2MelSpectrum y load_cnn_model importados si tu CNN ya está listo,
#  o bien puedes omitir el CNN y quedarte solo con pct_crackle y pct_wheeze basados en tu lógica previa).
# Para mantenerlo simple voy a simular que sólo uso las proporciones que ya tenías:
def extract_pct_feats_for_patient(pid):
    # Busca todos los wav
    wavs = glob.glob(os.path.join(AUDIO_DIR, f"{pid}_*.wav"))
    pct_list = []
    for w in wavs:
        # aquí iría tu CNN + load_cycles + extract_mel_spec_fixed → p = cnn.predict(...) 
        # pct_list.append(p)  
        # **POR EJEMPLO** haré un vector fijo para simular:
        pct_list.append(np.random.dirichlet([1,1,1,1]))  

    if not pct_list:
        # paciente sin audio, devolver ceros
        return np.zeros(4)
    P = np.vstack(pct_list)
    return P.mean(axis=0)

# --- 4) Construyo X y y ---
records = []
for pid in df['patient_id']:
    pct = extract_pct_feats_for_patient(pid)  # shape (4,)
    records.append( [pid, *pct] )

feat_cols = ['patient_id','pct_none','pct_crackle','pct_wheeze','pct_both']
df_feat = pd.DataFrame(records, columns=feat_cols)

# Me quedo sólo con los pacientes que tengo label
df_full = df_feat.merge(df, on='patient_id')
X = df_full[['pct_none','pct_crackle','pct_wheeze','pct_both']].values
y = le.transform(df_full['diagnosis'])

# --- 5) Entreno el RandomForest REALMENTE FITEADO ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# --- 6) Serializo TODO ---
joblib.dump(clf,         os.path.join(OUT_DIR, 'rf_3class.pkl'))
joblib.dump(le,          os.path.join(OUT_DIR, 'le_group3.pkl'))
joblib.dump(['pct_none','pct_crackle','pct_wheeze','pct_both'],
             os.path.join(OUT_DIR, 'feature_cols.pkl'))

print("✅ Modelo entrenado y guardado en 'models/'")
