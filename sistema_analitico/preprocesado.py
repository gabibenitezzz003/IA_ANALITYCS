"""Preprocesado avanzado: normalización, codificación de categóricas y pipeline.

Provee utilidades simples y seguras para convertir un DataFrame en matrices
listas para modelos. Todo en español y con API limpia.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os


def guardar_pipeline(ruta: str, objeto) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'wb') as f:
        pickle.dump(objeto, f)


def cargar_pipeline(ruta: str):
    with open(ruta, 'rb') as f:
        return pickle.load(f)


def preparar_matriz(df: pd.DataFrame, columnas_numericas: List[str], columnas_categoricas: List[str]) -> Tuple[np.ndarray, Dict[str, Dict]]:
    """Convierte el DataFrame en una matriz numérica y devuelve metadatos.

    - Rellena nulos con la mediana para numéricos.
    - Codifica categóricas con índices (label encoding) de forma reproducible.
    """
    df = df.copy()
    metadatos = {'encoders': {}}
    # Numericos
    X_num = []
    scaler = StandardScaler()
    for c in columnas_numericas:
        if c in df.columns:
            col = df[c].fillna(df[c].median()).astype(float)
        else:
            col = pd.Series([0.0] * len(df))
        X_num.append(col.values.astype(float))
    if X_num:
        X_num_arr = np.vstack(X_num).T
        X_num_arr = scaler.fit_transform(X_num_arr)
    else:
        X_num_arr = np.zeros((len(df), 0), dtype=float)
    # Categóricas
    # Categóricas: One-hot
    if columnas_categoricas:
        cat_vals = df[columnas_categoricas].fillna('').astype(str)
        # Compatibilidad con distintas versiones de scikit-learn
        try:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        except TypeError:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = ohe.fit_transform(cat_vals)
        metadatos['encoders'] = {'onehot': ohe, 'categoricas': columnas_categoricas}
    else:
        X_cat = np.zeros((len(df), 0), dtype=float)

    X = np.hstack([X_num_arr.astype('float32'), X_cat.astype('float32')])
    return X, metadatos
