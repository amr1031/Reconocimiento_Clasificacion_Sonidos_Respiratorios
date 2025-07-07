#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# train_and_dump_bueno_tuned.py

#!/usr/bin/env python
# coding: utf-8

import os, glob
import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model

# 1) Rutas y directorios
WAV_DIR    = 'INPUT/WAV'
META_CSV   = 'metadata/diagnostics.csv'
OUT_DIR    = 'models_group3'
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Carga de metadata
df = pd.read_csv(META_CSV)

# 3) Mapeo a 3 categorías clínicas
mapping = {
    'Healthy': 'Sano',
    'URTI':      'Infecciosa',
    'COPD':      'Obstructiva',
    # agrega más mapeos si es necesario
}
df['group3'] = df['diagnostic'].map(mapping)

# 4) Extracción de probabilidades promedio por paciente
records = []
for pid in df['patient_id'].unique():
    path = os.path.join(WAV_DIR, f"{pid}.wav")
    y, sr = librosa.load(path, sr=22050)
    cnn = load_model('models/cnn_model.h5')
    feats = []
    # ventanas de 5 s con salto de 1 s
    win_len = 5 * sr
    hop_len = sr
    for start in range(0, len(y), hop_len):
        segment = y[start:start+win_len]
        if len(segment) < win_len:
            break
        # mel-espectrograma normalizado
        melspec = librosa.feature.melspectrogram(segment, sr=sr, n_mels=50, n_fft=512, hop_length=256)
        logspec = librosa.power_to_db(melspec)
        logspec = (logspec - logspec.mean()) / (logspec.std() + 1e-6)
        logspec = np.expand_dims(logspec, axis=(0,-1))
        probs = cnn.predict(logspec, verbose=0)[0]
        feats.append(probs)
    if not feats:
        feats = [[0.,0.,0.,0.]]
    mean_feats = np.mean(np.vstack(feats), axis=0)
    records.append([pid, *mean_feats])

feat_cols = ['patient_id','pct_none','pct_crackle','pct_wheeze','pct_both']
df_feat = pd.DataFrame(records, columns=feat_cols)

# 5) Preparación de X, y y codificador
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(df['group3'])
df_full = df_feat.merge(df[['patient_id','group3']], on='patient_id')
X = df_full[feat_cols[1:]].values
y = le.transform(df_full['group3'])

# 6) Definición del Random Forest\clf = RandomForestClassifier(
    n_estimators=300,
    criterion='entropy',
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# 7) Validación cruzada estratificada (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"F1–Macro CV (5 pliegues): {scores.mean():.4f} ± {scores.std():.4f}")

# 8) Entrenamiento final sobre todo el conjunto y métricas OOB
clf.fit(X, y)
print(f"OOB score: {clf.oob_score_:.4f}\n")

# Métricas sobre todo el conjunto para ver consistencia
y_pr = clf.predict(X)
print("=== Métricas finales ===")
print(classification_report(
    le.inverse_transform(y),
    le.inverse_transform(y_pr),
    zero_division=0
))
print("Matriz de confusión:")
print(confusion_matrix(y, y_pr))

# 9) Serialización de artefactos
joblib.dump(clf, os.path.join(OUT_DIR, 'rf_group3_tuned.pkl'))
joblib.dump(le, os.path.join(OUT_DIR, 'le_group3.pkl'))
joblib.dump(feat_cols[1:], os.path.join(OUT_DIR, 'feature_cols_group3.pkl'))
print(f"Modelo final entrenado y guardado en '{OUT_DIR}/rf_group3_tuned.pkl'")
