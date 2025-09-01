"""Módulo de visualizaciones interactivas para dashboards.

Provee funciones que crean gráficos interactivos usando Plotly si está
disponible, o Matplotlib como respaldo. Nombres y variables en español.
"""
from typing import Optional
import numpy as np


def grafico_series_temporales(x, y, titulo: str = "Serie temporal"):
    try:
        import plotly.graph_objs as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='valor'))
        fig.update_layout(title=titulo, xaxis_title='x', yaxis_title='y')
        return fig
    except Exception:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='o')
        ax.set_title(titulo)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig


class VisualizadorInteractivo:
    def mostrar_serie(self, eje_x, eje_y, titulo: str = "Serie"):
        fig = grafico_series_temporales(eje_x, eje_y, titulo)
        try:
            import streamlit as st
            st.plotly_chart(fig) if hasattr(fig, 'to_plotly_json') else st.pyplot(fig)
        except Exception:
            # Si no hay Streamlit, intentar guardar a HTML/PNG
            try:
                if hasattr(fig, 'write_html'):
                    fig.write_html('salida_serie.html')
                else:
                    fig.savefig('salida_serie.png')
            except Exception:
                pass
