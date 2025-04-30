#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train_and_dump.py

import os
import glob
import warnings
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ――― Evitamos sólo los warnings de n_fft demasiado grande ―――
warnings.filterwarnings(
    "ignore",
    message="n_fft.* is too large for input signal.*"
)

# ――― RUTAS ―――
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(
    BASE_DIR,
    'input',
    'Respiratory_Sound_Database',
    'Respiratory_Sound_Database'
)
AUDIO_DIR  = os.path.join(INPUT_DIR, 'audio_and_txt_files')
CSV_PATH   = os.path.join(INPUT_DIR, 'patient_diagnosis.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Lectura de CSV en:", CSV_PATH)
print("Buscando audios en:", AUDIO_DIR)

# ――― 1) Cargo y agrupo diagnósticos a 3 clases ―――
df = pd.read_csv(CSV_PATH, header=None, names=['patient_id','diagnosis'])
mapping = {
    'COPD'           : 'Obstructive',
    'Asthma'         : 'Obstructive',
    'Bronchiectasis' : 'Obstructive',
    'Bronchiolitis'  : 'Obstructive',
    'Healthy'        : 'Healthy'
}
df['group3'] = df['diagnosis'].map(mapping).fillna('Infectious')
le3 = LabelEncoder()
df['label'] = le3.fit_transform(df['group3'])

# ――― 2) Cargo CNN para extraer probabilidades ―――
cnn = load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))

# ――― 3) Funciones auxiliares ―――
def extract_mel_spec_fixed(y, sr,
                           n_mels=50, n_fft=512,
                           hop_length=256, target_frames=245):
    # ajustamos n_fft a la longitud de la señal
    fft = min(n_fft, len(y))
    if len(y) < fft:
        y = np.pad(y, (0, fft - len(y)), mode='constant')
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(S, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    mel = (mel_db - mn) / (mx - mn) if mx > mn else np.zeros_like(mel_db)
    T = mel.shape[1]
    if T < target_frames:
        mel = np.pad(mel, ((0,0),(0,target_frames - T)), mode='constant')
    else:
        mel = mel[:, :target_frames]
    return mel

def load_cycles(txt_path, wav_path):
    segs = np.loadtxt(txt_path, dtype=int, delimiter='\t', usecols=(0,1))
    audio, sr = librosa.load(wav_path, sr=None)
    chunks = []
    for s_ms, e_ms in segs:
        if e_ms <= s_ms:
            continue
        s = int(s_ms / 1000 * sr)
        e = int(e_ms / 1000 * sr)
        chunk = audio[s:e]
        if chunk.size > 0:
            chunks.append((chunk, sr))
    return chunks

# ――― 4) Extraigo features por paciente ―――
records = []
for _, row in df.iterrows():
    pid = row['patient_id']
    rec = {'patient_id': pid, 'label': row['label']}
    probs, rms_list, zcr_list = [], [], []
    mf_means, mf_stds = [], []
    cents, bands, rolls = [], [], []

    wavs = glob.glob(os.path.join(AUDIO_DIR, f"{pid}_*.wav"))
    for wav in wavs:
        txt = wav[:-4] + '.txt'
        if not os.path.isfile(txt):
            continue
        for chunk, sr in load_cycles(txt, wav):
            # 1) probabilidades CNN
            mel = extract_mel_spec_fixed(chunk, sr)
            p   = cnn.predict(mel[np.newaxis, ..., np.newaxis], verbose=0)[0]
            probs.append(p)
            # 2) RMS & ZCR
            rms_list.append(librosa.feature.rms(y=chunk).mean())
            zcr_list.append(librosa.feature.zero_crossing_rate(y=chunk).mean())
            # 3) MFCCs
            mf = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            mf_means.append(mf.mean(axis=1))
            mf_stds.append(mf.std(axis=1))
            # 4) descriptores espectrales
            cents.append(librosa.feature.spectral_centroid(y=chunk, sr=sr).mean())
            bands.append(librosa.feature.spectral_bandwidth(y=chunk, sr=sr).mean())
            rolls.append(librosa.feature.spectral_rolloff(y=chunk, sr=sr).mean())

    if not probs:
        continue

    P = np.vstack(probs)
    for i, lab in enumerate(['none','crackle','wheeze','both']):
        rec[f'mean_p{i}_{lab}'] = P[:, i].mean()

    rec['rms_mean'], rec['rms_std'] = np.mean(rms_list), np.std(rms_list)
    rec['zcr_mean'], rec['zcr_std'] = np.mean(zcr_list), np.std(zcr_list)

    mm, ms = np.vstack(mf_means), np.vstack(mf_stds)
    for j in range(13):
        rec[f'mfcc{j+1}_mean'] = mm[:, j].mean()
        rec[f'mfcc{j+1}_std']  = ms[:, j].mean()

    rec['centroid_mean'], rec['centroid_std']   = np.mean(cents), np.std(cents)
    rec['bandwidth_mean'], rec['bandwidth_std'] = np.mean(bands), np.std(bands)
    rec['rolloff_mean'], rec['rolloff_std']     = np.mean(rolls), np.std(rolls)

    records.append(rec)

# ――― 5) Construyo DataFrame y entreno RandomForest ―――
df_feat      = pd.DataFrame(records)
feature_cols = [c for c in df_feat.columns if c not in ('patient_id','label')]
X            = df_feat[feature_cols].values
y            = df_feat['label'].values

clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3.fit(X, y)

# ――― 6) Serializo artefactos en models/ ―――
joblib.dump(clf3,         os.path.join(MODEL_DIR, 'rf_3class.pkl'))
joblib.dump(le3,          os.path.join(MODEL_DIR, 'le3_group3.pkl'))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.pkl'))

print("✅ Artefactos guardados en", MODEL_DIR, ":", os.listdir(MODEL_DIR))

