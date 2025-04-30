#!/usr/bin/env python
# coding: utf-8

# In[2]:


# inference.py

import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# ─── RUTAS ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ─── CARGO MODELOS Y ARTEFACTOS───────────────────────────────────
cnn          = load_model(os.path.join(MODELS_DIR, 'cnn_model.h5'))
clf3         = joblib.load(os.path.join(MODELS_DIR, 'rf_3class.pkl'))
le3          = joblib.load(os.path.join(MODELS_DIR, 'le3_group3.pkl'))
feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))

# ─── MEL-SPECTROGRAM FIJO 50×245 ──────────────────────────────────
def extract_mel_spec_fixed(y, sr,
                           n_mels=50, n_fft=512,
                           hop_length=256, target_frames=245):
    fft = min(n_fft, len(y))
    if len(y) < fft:
        y = np.pad(y, (0, fft - len(y)), 'constant')
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels, n_fft=fft,
        hop_length=hop_length,
        power=2.0
    )
    mel_db = librosa.power_to_db(S, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    mel = (mel_db - mn) / (mx - mn) if mx > mn else np.zeros_like(mel_db)
    T = mel.shape[1]
    if T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), 'constant')
    else:
        mel = mel[:, :target_frames]
    return mel

# ─── EXTRACCIÓN GLOBAL DE FEATURES AUDIO-ONLY──────────────────────
def extract_global_features(audio, sr):
    """
    Divide el audio en ventanas de 5s (hop=1s) y para cada ventana extrae:
      - probabilidades CNN (4)
      - rms_mean, rms_std (2)
      - zcr_mean, zcr_std (2)
      - mfcc1..13_mean, mfcc1..13_std (26)
      - centroid_mean, centroid_std (2)
      - bandwidth_mean, bandwidth_std (2)
      - rolloff_mean, rolloff_std (2)
    Total = 4 + 2 + 2 + 26 + 2 + 2 + 2 = 40
    Devuelve un array de 1×40 en el orden de feature_cols.
    """
    win_s, hop_s = 5.0, 1.0
    wlen = int(win_s * sr)

    p_list = []
    rms_list = []
    zcr_list = []
    mfcc_mean_list = []
    mfcc_std_list = []
    cent_list = []
    band_list = []
    roll_list = []

    # 1) Extrae caracteristicas por ventana
    for start in np.arange(0, len(audio) / sr - win_s + 1e-6, hop_s):
        i0 = int(start * sr)
        seg = audio[i0: i0 + wlen]
        if len(seg) < wlen:
            seg = np.pad(seg, (0, wlen - len(seg)), 'constant')

        # a) CNN
        mel = extract_mel_spec_fixed(seg, sr)
        p   = cnn.predict(mel[np.newaxis, ..., np.newaxis], verbose=0)[0]
        p_list.append(p)

        # b) RMS y ZCR
        rms_list.append(librosa.feature.rms(y=seg).mean())
        zcr_list.append(librosa.feature.zero_crossing_rate(y=seg).mean())

        # c) MFCC
        mf = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        mfcc_mean_list.append(mf.mean(axis=1))
        mfcc_std_list.append(mf.std(axis=1))

        # d) Espectrales
        cent_list.append( librosa.feature.spectral_centroid(y=seg, sr=sr).mean() )
        band_list.append( librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean() )
        roll_list.append( librosa.feature.spectral_rolloff(y=seg, sr=sr).mean() )

    # 2) Agrega media y std para cada característica
    rec = {}
    P = np.vstack(p_list)  # ventana×4
    labels = ['none','crackle','wheeze','both']
    for i,lab in enumerate(labels):
        rec[f'mean_p{i}_{lab}'] = P[:,i].mean()

    rec['rms_mean'], rec['rms_std'] = np.mean(rms_list), np.std(rms_list)
    rec['zcr_mean'], rec['zcr_std'] = np.mean(zcr_list), np.std(zcr_list)

    M_mean = np.vstack(mfcc_mean_list)  # ventana×13
    M_std  = np.vstack(mfcc_std_list)
    for j in range(13):
        rec[f'mfcc{j+1}_mean'] = M_mean[:,j].mean()
        rec[f'mfcc{j+1}_std']  = M_std[:,j].mean()

    rec['centroid_mean'], rec['centroid_std']   = np.mean(cent_list), np.std(cent_list)
    rec['bandwidth_mean'], rec['bandwidth_std'] = np.mean(band_list), np.std(band_list)
    rec['rolloff_mean'], rec['rolloff_std']     = np.mean(roll_list), np.std(roll_list)

    # 3) Monta el array final 1×40 según feature_cols
    import pandas as pd
    df = pd.DataFrame([rec])
    return df[feature_cols].values

# ─── PREDICCIÓN FINAL───────────────────────────────────────────────
def predict_disease_from_audio(audio, sr):
    """
    Recibe el array de audio y su sr, extrae las 40 features
    y devuelve la etiqueta final (Healthy/Obstructive/Infectious).
    """
    Xg = extract_global_features(audio, sr)
    y  = clf3.predict(Xg)[0]
    return le3.inverse_transform([y])[0]

