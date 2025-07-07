#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model

# 1) Rutas y directorios
WAV_DIR  = 'INPUT/WAV'
META_CSV = 'metadata/diagnostics.csv'
OUT_DIR  = 'models_group3'
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Carga de metadata
df = pd.read_csv(META_CSV)

# 3) Mapeo a 3 categorías clínicas
mapping = {
    'Healthy': 'Sano',
    'URTI':      'Infecciosa',
    'COPD':      'Obstructiva',
}
df['group3'] = df['diagnostic'].map(mapping)

# 4) Extracción de probabilidades promedio por paciente
records = []
cnn = load_model('models/cnn_model.h5')
for pid in df['patient_id'].unique():
    path = os.path.join(WAV_DIR, f"{pid}.wav")
    y, sr = librosa.load(path, sr=22050)
    feats = []
    win_len = 5 * sr
    hop_len = sr
    for start in range(0, len(y), hop_len):
        segment = y[start:start+win_len]
        if len(segment) < win_len:
            break
        melspec = librosa.feature.melspectrogram(segment, sr=sr, n_mels=50, n_fft=512, hop_length=256)
        logspec = librosa.power_to_db(melspec)
        logspec = (logspec - logspec.mean()) / (logspec.std() + 1e-6)
        logspec = np.expand_dims(logspec, axis=(0,-1))
        probs = cnn.predict(logspec, verbose=0)[0]
        feats.append(probs)
    if not feats:
        feats = [[0., 0., 0., 0.]]
    mean_feats = np.mean(np.vstack(feats), axis=0)
    records.append([pid, *mean_feats])

feat_cols = ['patient_id', 'pct_none', 'pct_crackle', 'pct_wheeze', 'pct_both']
df_feat = pd.DataFrame(records, columns=feat_cols)

# 5) Preparación de X, y y codificador
le = LabelEncoder().fit(df['group3'])
df_full = df_feat.merge(df[['patient_id', 'group3']], on='patient_id')
X = df_full[feat_cols[1:]].values
y = le.transform(df_full['group3'])

# 6) Configuración inicial del Random Forest
base_clf = RandomForestClassifier(
    criterion='entropy',
    class_weight='balanced_subsample',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# 7) Búsqueda de hiperparámetros con GridSearchCV y validación cruzada estratificada
param_grid = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    refit=True
)
grid.fit(X, y)
clf = grid.best_estimator_
print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor F1–macro (CV): {grid.best_score_:.4f}\n")

# 8) Entrenamiento final y métricas OOB
clf.fit(X, y)
print(f"OOB score: {clf.oob_score_:.4f}\n")

# 9) Métricas finales en todo el conjunto
y_pred = clf.predict(X)
print("=== Métricas finales ===")
print(classification_report(
    le.inverse_transform(y),
    le.inverse_transform(y_pred),
    zero_division=0
))
print("Matriz de confusión:")
print(confusion_matrix(y, y_pred))

# 10) Serialización de artefactos
joblib.dump(clf, os.path.join(OUT_DIR, 'rf_group3_tuned.pkl'))
joblib.dump(le, os.path.join(OUT_DIR, 'le_group3.pkl'))
joblib.dump(feat_cols[1:], os.path.join(OUT_DIR, 'feature_cols_group3.pkl'))
print(f"Modelo final entrenado y guardado en '{OUT_DIR}/rf_group3_tuned.pkl'")
