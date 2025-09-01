Sistema Analítico IA (demo)

Este repositorio contiene una estructura inicial en español para un sistema
analítico impulsado por IA con una interfaz gráfica interactiva (Streamlit).

Estructura:
- sistema_analitico/: módulos en español para IA, modelos, visualizaciones y utilidades.
- README.md: esta guía.

Cómo ejecutar la interfaz (recomendado en entorno virtual):

1. Instalar dependencias:
   pip install -r requirements.txt
2. Ejecutar:
   streamlit run sistema_analitico/interfaz_app.py

Notas:
- Esta es una demo local y determinista. Para producción, integre modelos reales
  (PyTorch, TensorFlow, OpenAI, etc.) y orquestación.

Arquitectura propuesta
---------------------
El proyecto está organizado por responsabilidades: lógica de IA, modelos,
visualizaciones, persistencia y la interfaz. Puntos clave:

- `sistema_analitico/utilidades.py`: funciones de ayuda y preprocesado ligero.
- `sistema_analitico/preprocesado.py`: pipeline de preprocesado para datasets (numericos/categoricos).
- `sistema_analitico/persistencia.py`: guardar/cargar parámetros y memoria; soporta PyTorch si está instalado.
- `sistema_analitico/generativo.py`: motor generativo con integración opcional a Hugging Face.
- `sistema_analitico/ia_avanzada.py`: motor de entrenamiento (usa PyTorch si está disponible).
- `sistema_analitico/interfaz_app.py`: Streamlit app para interacción y dashboards.

Ejecutar tests
--------------
Instale dependencias y luego:

   pip install -r requirements.txt
   pip install pytest
   pytest -q

