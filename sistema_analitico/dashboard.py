"""Módulo de dashboards y reportes exportables.

Provee utilidades para crear tablas interactivas, mapas de calor y exportar
reportes en HTML/CSV desde la app.
"""
import pandas as pd
import numpy as np
import os


def tabla_interactiva(df: pd.DataFrame):
    # Devuelve el DataFrame (Streamlit lo mostrará interactivo)
    return df


def mapa_calor(df: pd.DataFrame, columnas=None):
    import plotly.express as px
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[columnas].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
    return fig


def exportar_reporte_html(df: pd.DataFrame, ruta: str):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    html = df.to_html(classes='table table-striped')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(html)
    return ruta
