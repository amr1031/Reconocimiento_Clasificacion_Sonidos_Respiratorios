#!/usr/bin/env python
# coding: utf-8

# In[2]:


# inference.py
import os
import joblib
import numpy as np
import librosa

# ─── 1) Carga de artefactos ─────────────────────────────────────────────────────
MODEL_DIR      = os.path.join(os.path.dirname(__file__), 'models')
RF_PATH        = os.path.join(MODEL_DIR, 'rf_3class.pkl')
LE_PATH        = os.path.join(MODEL_DIR, 'le_group3.pkl')
FEATCOLS_PATH  = os.path.join(MODEL_DIR, 'feature_cols.pkl')
CNN_PATH       = os.path.join(MODEL_DIR, 'cnn_model.h5')

# RandomForest, LabelEncoder y lista de columnas
clf         = joblib.load(RF_PATH)
le          = joblib.load(LE_PATH)
feature_cols= joblib.load(FEATCOLS_PATH)

# CNN para crackles/wheezes usada como feature extractor
from tensorflow.keras.models import load_model
cnn = load_model(CNN_PATH)

# ─── 2) Funciones auxiliares ────────────────────────────────────────────────────
def extract_mel_spec_fixed(y, sr, n_mels=50, n_fft=512, hop_length=256, target_frames=245):
    """Como en train: mel-gram 50×245 normalizado."""
    S       = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=n_fft, hop_length=hop_length,
                                              power=2.0)
    mel_db  = librosa.power_to_db(S, ref=np.max)
    mn, mx  = mel_db.min(), mel_db.max()
    mel     = (mel_db - mn)/(mx - mn) if mx>mn else np.zeros_like(mel_db)
    # pad/cut
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0,0),(0,target_frames-mel.shape[1])), 'constant')
    else:
        mel = mel[:, :target_frames]
    return mel

def extract_pct_features_from_signal(data, sr, window_sec=5.0, step_sec=1.0):
    """
    Divide data en ventanas de window_sec reaplicando cada step_sec,
    predice con el CNN y devuelve pct_none, pct_crackle, pct_wheeze, pct_both.
    """
    wlen   = int(window_sec * sr)
    step   = int(step_sec   * sr)
    probs  = []  # lista de [p0,p1,p2,p3] por ventana

    for start in range(0, len(data)-wlen+1, step):
        seg = data[start:start+wlen]
        if len(seg)<wlen:
            seg = np.pad(seg, (0, wlen-len(seg)), 'constant')
        mel = extract_mel_spec_fixed(seg, sr)
        x   = mel[np.newaxis, ..., np.newaxis]
        p   = cnn.predict(x, verbose=0)[0]
        probs.append(p)

    if not probs:
        raise RuntimeError("Audio demasiado corto para extraer features.")

    P = np.vstack(probs)
    # media de probabilidades en todas las ventanas
    pct = P.mean(axis=0)  
    # devolvemos pct_none, pct_crackle, pct_wheeze, pct_both
    return pct.tolist()

# ─── 3) Función principal de inferencia ─────────────────────────────────────────
def predict_disease_from_audio(data: np.ndarray, sr: int):
    """
    Dado un array de audio (normalizado [-1,1]) y su sample rate,
    extrae pct-features (4 valores) y predice la clase de enfermedad.
    Devuelve (label_str, {label:prob}).
    """
    # 1) extraer los 4 pct-features
    pct_feats = extract_pct_features_from_signal(data, sr)
    # 2) montar el vector X en el orden correcto
    X = np.array([pct_feats])[...,]  # shape (1,4)
    # 3) predecir con el RF
    proba = clf.predict_proba(X)[0]
    num   = np.argmax(proba)
    label = le.inverse_transform([num])[0]
    # 4) mapear feature names si quieres debug
    feat_dict = {c:v for c,v in zip(feature_cols, pct_feats)}
    return label, {cls: float(f"{p:.3f}") for cls,p in zip(le.classes_, proba)}

# ─── 4) Prueba rápida (si se ejecuta directamente) ─────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv)!=2:
        print("Uso: python inference.py ruta/al/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    y,sr = librosa.load(wav_path, sr=None)
    disease, probs = predict_disease_from_audio(y, sr)
    print("Predicción:", disease)
    print("Probabilidades:", probs)
