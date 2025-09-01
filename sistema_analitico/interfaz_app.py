"""Aplicación de interfaz gráfica (Streamlit) para el sistema analítico.

Este archivo define un punto de entrada ligero que intenta usar Streamlit
si está disponible; si no, ofrece instrucciones para ejecutar la app.
Todos los nombres están en español y son descriptivos.
"""
from typing import Optional


def ejecutar_interfaz():
    """Inicia la interfaz gráfica si Streamlit está instalado.

    Si Streamlit no está disponible, imprime instrucciones.
    """
    try:
        import streamlit as st
        from sistema_analitico.ia_analitica import MotorLLMAnalitico, MotorDecisionAnalitico
        from sistema_analitico.modelos_predictivos import ModeloBasicoPredictivo
        from sistema_analitico.visualizaciones import VisualizadorInteractivo
        from sistema_analitico.utilidades import cargar_datos_csv, preprocesar_texto
    except Exception as e:
        print("Streamlit no está instalado o hay un error en imports:", e)
        print("Para ejecutar la interfaz instale dependencias y ejecute:\n  streamlit run sistema_analitico/interfaz_app.py")
        return

    st.set_page_config(page_title="Sistema Analítico IA", layout="wide")

    st.sidebar.title("Panel de control")
    opcion = st.sidebar.selectbox("Acción", ["Cargar datos", "Análisis", "Modelado", "Visualizaciones"])

    if opcion == "Cargar datos":
        archivo = st.file_uploader("Suba un CSV con los datos", type=["csv"])
        if archivo is not None:
            df = cargar_datos_csv(archivo)
            st.write(df.head())
            st.success("Datos cargados correctamente. Use la pestaña 'Análisis' para continuar.")

    elif opcion == "Análisis":
        st.header("Análisis automático con IA")
        texto = st.text_area("Proporcione conocimiento o contexto (texto libre)")
        if st.button("Extraer contexto y resumen"):
            motor = MotorLLMAnalitico()
            contexto = motor.extraer_contexto(texto)
            st.write("Contexto numérico (vector):")
            st.write(contexto[:10])
            motor_dec = MotorDecisionAnalitico()
            decisiones = motor_dec.evaluar_decisiones(contexto)
            st.write("Decisiones sugeridas:")
            st.json(decisiones)

    elif opcion == "Modelado":
        st.header("Entrenar y predecir")
        st.write("Este módulo permite entrenar modelos predictivos simples con los datos cargados.")
        st.info("Nota: en esta demo el entrenado es local y rápido; para producción utilice infraestructuras dedicadas.")

    elif opcion == "Visualizaciones":
        st.header("Visualizaciones interactivas")
        st.write("Use la pestaña de 'Cargar datos' para cargar un dataset y luego explore aquí.")


if __name__ == '__main__':
    ejecutar_interfaz()
