"""Aplicación de interfaz gráfica (Streamlit) para el sistema analítico.

Este archivo define un punto de entrada ligero que intenta usar Streamlit
si está disponible; si no, ofrece instrucciones para ejecutar la app.
Todos los nombres están en español y son descriptivos.
"""
from typing import Optional
import numpy as np
import threading
import time
import multiprocessing

# Metrics: expose a small Prometheus metrics server in background if prometheus_client is available
_METRICS_STARTED = False
try:
    from prometheus_client import Counter, Histogram, start_http_server
    REQ_COUNTER = Counter('sistema_analitico_requests_total', 'Total de solicitudes a operaciones críticas', ['operacion'])
    INF_LATENCY = Histogram('sistema_analitico_inferencia_latency_seconds', 'Latencia de inferencia en segundos', ['operacion'])
    def _start_metrics_server(port: int = 8000):
        """Start metrics server in a separate process so Streamlit re-execs don't kill it."""
        global _METRICS_STARTED
        if _METRICS_STARTED:
            return
        def _serve_proc(p):
            try:
                start_http_server(p)
                with open('/tmp/metrics_start.log', 'a') as fh:
                    fh.write(f'metrics_started on port {p}\n')
                # block to keep process alive
                while True:
                    time.sleep(3600)
            except Exception as e:
                try:
                    with open('/tmp/metrics_start.log', 'a') as fh:
                        fh.write(f'metrics_start_error: {e!r}\n')
                except Exception:
                    pass

        proc = multiprocessing.Process(target=_serve_proc, args=(port,), daemon=True)
        proc.start()
        _METRICS_STARTED = True
except Exception:
    # prometheus_client not available; metrics are disabled
    REQ_COUNTER = None
    INF_LATENCY = None


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
    # Start prometheus metrics server (best-effort)
    try:
        _start_metrics_server(8000)
    except Exception:
        pass
    # Aplicar CSS personalizado si existe
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

    st.sidebar.title("Panel de control")
    opcion = st.sidebar.selectbox("Acción", ["Login", "Cargar datos", "Análisis", "Modelado", "Visualizaciones", "Decisiones"])

    # Login simple
    if opcion == 'Login':
        try:
            from sistema_analitico.auth import crear_usuario, verificar_usuario
        except Exception:
            st.warning('Módulo de autenticación no disponible')
            return
        usuario = st.text_input('Usuario')
        password = st.text_input('Contraseña', type='password')
        if st.button('Registrar (demo)'):
            crear_usuario(usuario, password)
            st.success('Usuario creado (demo)')
        if st.button('Iniciar sesión'):
            if verificar_usuario(usuario, password):
                st.success('Login correcto')
            else:
                st.error('Usuario o contraseña incorrectos')
        return
    # Encabezado principal
    st.markdown(
        "<div class='cabecera'><div class='logo'>IA</div><div><div class='titulo'>Sistema Analítico IA — Dashboard</div><div class='subtitulo'>Analítica, generación y decisiones asistidas</div></div></div>",
        unsafe_allow_html=True,
    )

    # Organización por pestañas para una UI más limpia
    tab_cargar, tab_analisis, tab_modelado, tab_visual, tab_decisiones = st.tabs(["Cargar datos", "Análisis", "Modelado", "Visualizaciones", "Decisiones"])

    if opcion == "Cargar datos" or True:
        with tab_cargar:
            archivo = st.file_uploader("Suba un CSV con los datos", type=["csv"] )
            if archivo is not None:
                df = cargar_datos_csv(archivo)
                st.session_state['df_actual'] = df
                st.write(df.head())
                st.success("Datos cargados correctamente. Use las pestañas para continuar.")

    if opcion == "Análisis" or True:
        with tab_analisis:
            st.header("Análisis automático con IA")
            texto = st.text_area("Proporcione conocimiento o contexto (texto libre)")
            if st.button("Extraer contexto y resumen"):
                motor = MotorLLMAnalitico()
                # instrument: measure inference latency and count
                if REQ_COUNTER is not None and INF_LATENCY is not None:
                    REQ_COUNTER.labels(operacion='extraer_contexto').inc()
                    start = time.time()
                    contexto = motor.extraer_contexto(texto)
                    INF_LATENCY.labels(operacion='extraer_contexto').observe(time.time() - start)
                else:
                    contexto = motor.extraer_contexto(texto)
                st.write("Contexto numérico (vector):")
                st.write(contexto[:10])
                motor_dec = MotorDecisionAnalitico()
                decisiones = motor_dec.evaluar_decisiones(contexto)
                st.write("Decisiones sugeridas:")
                st.json(decisiones)
            st.markdown("---")
            st.subheader('Generación y resumen')
            from sistema_analitico.generativo import MotorGenerativo
            mg = MotorGenerativo()
            modelo_hf = st.selectbox('Modelo Hugging Face para generación (si está instalado)', ['gpt2', 'distilgpt2', 'EleutherAI/gpt-neo-125M'])
            mg.modelo_texto = modelo_hf
            prompt = st.text_area('Prompt para generación')
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Generar texto'):
                    salida = mg.generar_texto(prompt or texto, max_length=128)
                    st.write(salida)
            with col2:
                if st.button('Resumir texto'):
                    resumen = mg.resumir_texto(texto or prompt, max_length=80)
                    st.write(resumen)

            # Visualizaciones rápidas
            st.markdown('---')
            st.subheader('Dashboard rápido')
            from sistema_analitico.dashboard import mapa_calor, exportar_reporte_html
            df = st.session_state.get('df_actual')
            if df is not None:
                if st.button('Mostrar mapa de calor (correlaciones)'):
                    fig = mapa_calor(df)
                    st.plotly_chart(fig)
                if st.button('Exportar reporte HTML'):
                    ruta = exportar_reporte_html(df, 'artifacts/reporte.html')
                    st.success(f'Reporte exportado a {ruta}')
    
    # (bloque duplicado eliminado)

    elif opcion == "Modelado":
        st.header("Entrenar y predecir")
        st.write("Este módulo permite entrenar modelos predictivos simples con los datos cargados.")
        st.info("Nota: en esta demo el entrenado es local y rápido; para producción utilice infraestructuras dedicadas.")
        df = st.session_state.get('df_actual')
        if df is None:
            st.warning("No hay dataset cargado. Cargue un CSV en la pestaña 'Cargar datos'.")
        else:
            columnas = list(df.columns)
            col_target = st.selectbox('Seleccione columna objetivo (target)', columnas)
            col_features = st.multiselect('Seleccione columnas de entrada (features)', [c for c in columnas if c != col_target])
            if st.button('Entrenar modelo con Motor IA avanzado'):
                from sistema_analitico.ia_avanzada import MotorIAAvanzada
                X = df[col_features].fillna(0).values.astype('float32')
                y = df[col_target].fillna(0).values.astype('float32')
                motor = MotorIAAvanzada(dimension_entrada=X.shape[1], dimension_oculta=64)
                motor.inicializar_modelo()
                with st.spinner('Entrenando...'):
                    motor.entrenar(X, y, pasos=50)
                st.success('Entrenamiento finalizado')
                muestra = X[0:5]
                preds = [motor.predecir(m) for m in muestra]
                st.write('Predicciones de ejemplo:', preds)
                # Ofrecer guardar el modelo
                if st.button('Guardar modelo entrenado'):
                    from sistema_analitico.persistencia import guardar_modelo_numpy
                    try:
                        # Si el motor tiene pesos numpy
                        guardar_modelo_numpy('artifacts/modelo_params.npy', motor.pesos)
                        st.success('Modelo guardado en artifacts/modelo_params.npy')
                    except Exception:
                        st.error('No se pudo guardar el modelo con el método actual')
                if st.button('Guardar modelo PyTorch (state_dict)'):
                    try:
                        from sistema_analitico.persistencia import guardar_modelo_pytorch
                        # Solo funcionará si el motor interno es PyTorch
                        if hasattr(motor, 'modelo') and motor._inicializado_con_torch:
                            guardar_modelo_pytorch('artifacts/modelo_state.pt', motor.modelo)
                            st.success('State dict guardado en artifacts/modelo_state.pt')
                        else:
                            st.error('El motor no tiene modelo PyTorch inicializado')
                    except Exception as e:
                        st.error(f'Error guardando modelo PyTorch: {e}')
                if st.button('Cargar modelo PyTorch (state_dict)'):
                    try:
                        from sistema_analitico.persistencia import cargar_modelo_pytorch
                        if hasattr(motor, 'modelo') and motor._inicializado_con_torch:
                            cargar_modelo_pytorch('artifacts/modelo_state.pt', motor.modelo)
                            st.success('Modelo cargado desde artifacts/modelo_state.pt')
                        else:
                            st.error('No hay modelo PyTorch inicializado para cargar los pesos')
                    except Exception as e:
                        st.error(f'Error cargando modelo PyTorch: {e}')

        st.markdown('---')
        st.subheader('AutoML - búsqueda de hiperparámetros')
        if st.button('Ejecutar búsqueda aleatoria (demo)'):
            from sistema_analitico.auto_ml import busqueda_aleatoria
            # evaluador demo: escogemos lr y pasos para simular una métrica
            def evaluador(params):
                lr = float(params.get('lr', 0.01))
                pasos = int(params.get('pasos', 10))
                # métrica sintética: preferir lr cercano a 0.01 y más pasos
                score = -abs(lr - 0.01) + 0.01 * pasos
                return score

            espacio = {'lr': [0.001, 0.005, 0.01, 0.05], 'pasos': [5, 10, 20, 50]}
            mejor, score = busqueda_aleatoria(evaluador, espacio, iteraciones=20)
            st.write('Mejor configuración encontrada:', mejor, 'score:', score)

        # tabla interactiva con AgGrid si está instalado
        try:
            from st_aggrid import AgGrid
            df = st.session_state.get('df_actual')
            if df is not None:
                AgGrid(df.head(50))
        except Exception:
            pass

    elif opcion == "Visualizaciones":
        st.header("Visualizaciones interactivas")
        st.write("Use la pestaña de 'Cargar datos' para cargar un dataset y luego explore aquí.")
        df = st.session_state.get('df_actual')
        if df is None:
            st.warning('No hay dataset cargado para visualizar')
        else:
            from sistema_analitico.visualizaciones import VisualizadorInteractivo
            vis = VisualizadorInteractivo()
            if st.button('Mostrar proyector de embeddings (demo)'):
                # demo embeddings
                emb = np.random.randn(len(df), 16).astype('float32')
                vis.mostrar_proyector_embeddings(emb, etiquetas=df.index.astype(str).tolist())

    elif opcion == 'Decisiones':
        st.header('Motor de decisiones')
        from sistema_analitico.decisiones import MotorDecisionAvanzado, ReglaAnalitica
        motor_dec = MotorDecisionAvanzado()
        # Regla demo
        def cond_demo(ctx):
            return True
        motor_dec.agregar_regla(ReglaAnalitica('Regla demo', cond_demo, {'accion': 'Monitorear', 'detalle': 'demo'}, peso=0.5))
        ctx = {}
        res = motor_dec.evaluar(ctx)
        st.write(res)


if __name__ == '__main__':
    ejecutar_interfaz()
