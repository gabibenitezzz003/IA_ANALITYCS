"""Módulo de visualizaciones interactivas para dashboards.

Provee funciones que crean gráficos interactivos usando Plotly si está
disponible, o Matplotlib como respaldo. Nombres y variables en español.
"""
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA


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

    def mostrar_proyector_embeddings(self, embeddings: np.ndarray, etiquetas=None, titulo: str = 'Embeddings'):
        try:
            import plotly.express as px
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            df = {'x': coords[:, 0], 'y': coords[:, 1]}
            if etiquetas is not None:
                df['label'] = etiquetas
            fig = px.scatter(df, x='x', y='y', color='label' if 'label' in df else None, title=titulo)
            try:
                import streamlit as st
                st.plotly_chart(fig)
            except Exception:
                fig.write_html('embeddings_proyector.html')
        except Exception:
            # Fallback matplotlib
            try:
                import matplotlib.pyplot as plt
                pca = PCA(n_components=2)
                coords = pca.fit_transform(embeddings)
                fig, ax = plt.subplots()
                ax.scatter(coords[:, 0], coords[:, 1])
                fig.savefig('embeddings_proyector.png')
            except Exception:
                pass
