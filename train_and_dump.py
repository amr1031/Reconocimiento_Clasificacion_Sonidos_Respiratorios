#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train_and_dump_enhanced.py

import os
import glob
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.preprocessing    import LabelEncoder
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import StratifiedKFold, cross_validate
from imblearn.over_sampling   import SMOTE

# ─── RUTAS ────────────────────────────────────────────────────────

ROOT_AUDIO = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
CSV_PATH   = 'input/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
MODELS_DIR = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── 1) Cargo diagnósticos ─────────────────────────────────────────
df_diag = pd.read_csv(CSV_PATH, header=None, names=['patient_id','diagnosis'])
# Creamos un grouping de 3 clases: Obstructive, Healthy, Infectious
grp_map = {
    'COPD':'Obstructive','Bronchiectasis':'Obstructive','Bronchiolitis':'Obstructive',
    'URTI':'Infectious','Pneumonia':'Infectious','LRTI':'Infectious',
    # todo lo demás → Healthy
}
df_diag['diagnosis_grp'] = df_diag['diagnosis'].map(grp_map).fillna('Healthy')

le3 = LabelEncoder()
y_all = le3.fit_transform(df_diag['diagnosis_grp'])

# ─── 2) Extraer ventanas y features globales (igual que en inference) ──
def extract_mel_spec_fixed(y, sr, n_mels=50, n_fft=512, hop_length=256, target_frames=245):
    fft = min(n_fft, len(y))
    if len(y) < fft:
        y = np.pad(y, (0, fft - len(y)), 'constant')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=fft,
                                       hop_length=hop_length, power=2.0)
    mel_db = librosa.power_to_db(S, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    mel = (mel_db - mn) / (mx - mn) if mx>mn else np.zeros_like(mel_db)
    T = mel.shape[1]
    if T < target_frames:
        mel = np.pad(mel, ((0,0),(0,target_frames-T)), 'constant')
    else:
        mel = mel[:,:target_frames]
    return mel

def load_cycles(txt_path, wav_path):
    segs  = np.loadtxt(txt_path, dtype=int, delimiter='\t', usecols=(0,1))
    audio, sr = librosa.load(wav_path, sr=None)
    out = []
    for s_ms,e_ms in segs:
        if e_ms <= s_ms: continue
        s = int(s_ms/1000 * sr); e = int(e_ms/1000 * sr)
        chunk = audio[s:e]
        if chunk.size>0: out.append((chunk, sr))
    return out

def extract_features_for_patient(pid):
    rec = {}
    prob_list, rms_list, zcr_list = [], [], []
    mfcc_mean_list, mfcc_std_list = [], []
    cent_list, band_list, roll_list = [], [], []

    # ruta a WAV/TXT
    files = glob.glob(os.path.join(ROOT_AUDIO,f"{pid}_*.wav"))
    for wav in files:
        txt = wav.replace('.wav','.txt')
        if not os.path.isfile(txt): continue

        for chunk,sr in load_cycles(txt,wav):
            # CNN → prob (aquí simulamos con mfcc como proxy)
            mel = extract_mel_spec_fixed(chunk,sr)
            # si tuvieras la CNN: p = cnn.predict(...)
            # for now extraemos RMS/ZCR etc
            # 1) RMS/ZCR
            rms_list.append(librosa.feature.rms(y=chunk).mean())
            zcr_list.append(librosa.feature.zero_crossing_rate(y=chunk).mean())
            # 2) MFCC
            mf = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            mfcc_mean_list.append(mf.mean(axis=1))
            mfcc_std_list.append(mf.std(axis=1))
            # 3) espectrales
            cent_list.append(librosa.feature.spectral_centroid(y=chunk, sr=sr).mean())
            band_list.append(librosa.feature.spectral_bandwidth(y=chunk, sr=sr).mean())
            roll_list.append(librosa.feature.spectral_rolloff(y=chunk, sr=sr).mean())

    # resumen
    rec['rms_mean'], rec['rms_std'] = np.mean(rms_list), np.std(rms_list)
    rec['zcr_mean'], rec['zcr_std'] = np.mean(zcr_list), np.std(zcr_list)

    Mm = np.vstack(mfcc_mean_list); Ms = np.vstack(mfcc_std_list)
    for j in range(13):
        rec[f'mfcc{j+1}_mean'] = Mm[:,j].mean()
        rec[f'mfcc{j+1}_std']  = Ms[:,j].mean()

    rec['centroid_mean'], rec['centroid_std']   = np.mean(cent_list), np.std(cent_list)
    rec['bandwidth_mean'], rec['bandwidth_std'] = np.mean(band_list), np.std(band_list)
    rec['rolloff_mean'], rec['rolloff_std']     = np.mean(roll_list), np.std(roll_list)

    return rec

# --- 3) Construyo DataFrame de features para todos los pacientes
records = []
for pid in df_diag['patient_id']:
    feats = extract_features_for_patient(pid)
    if feats:
        feats['patient_id'] = pid
        records.append(feats)

df_feat = pd.DataFrame.from_records(records)
feature_cols = [c for c in df_feat.columns if c!='patient_id']

# Serializo feature_cols:
joblib.dump(feature_cols, os.path.join(MODELS_DIR,'feature_cols.pkl'))

# --- 4) Uno con etiquetas y preparo X,y
df_final = df_feat.merge(df_diag[['patient_id','diagnosis_grp']], on='patient_id')
X = df_final[feature_cols].values
y = le3.transform(df_final['diagnosis_grp'])

# --- 5) SMOTE para balancear
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X,y)

# --- 6) Entreno RF con pesos balanceados
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'    # <–– pesos automáticos
)
cv = cross_validate(
    clf, X_res, y_res,
    cv=StratifiedKFold(5,shuffle=True,random_state=42),
    scoring=['accuracy','precision_macro','recall_macro','f1_macro']
)
print("CV Accuracy:",  np.mean(cv['test_accuracy']))
print("CV F1-macro:", np.mean(cv['test_f1_macro']))

# Ajusto final
clf.fit(X_res,y_res)

# --- 7) Serializo artefactos finales ────────────────────────────────
joblib.dump(clf,      os.path.join(MODELS_DIR,'rf_3class.pkl'))
joblib.dump(le3,      os.path.join(MODELS_DIR,'le3_group3.pkl'))

print("Modelos guardados en", MODELS_DIR)
