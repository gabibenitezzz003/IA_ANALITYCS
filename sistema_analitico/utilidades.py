"""Funciones utilitarias: carga de datos, preprocesado y helpers.

Todo en español y con nombres únicos.
"""
import pandas as pd
import numpy as np
import re
from typing import Any


def cargar_datos_csv(ruta_o_buffer) -> pd.DataFrame:
    df = pd.read_csv(ruta_o_buffer)
    return df


def preprocesar_texto(texto: str) -> str:
    if texto is None:
        return ""
    texto = texto.lower()
    texto = re.sub(r"https?://\S+", "", texto)
    texto = re.sub(r"[^a-z0-9áéíóúñü\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def vector_aleatorio(dimension: int, semilla: int = 42) -> np.ndarray:
    rng = np.random.RandomState(semilla)
    return rng.randn(dimension).astype(np.float32)
