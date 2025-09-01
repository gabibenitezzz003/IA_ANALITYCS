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

Ejecutar con Docker (recomendado para pruebas locales)
-----------------------------------------------

1) Construir y ejecutar con docker-compose:

   sudo docker-compose up -d --build

2) Verificar contenedor y logs:

   sudo docker ps
   sudo docker logs -f nuevo_proyecto-sistema_analitico-1

3) Acceder a la interfaz web en:

   http://localhost:8501

Persistencia local (artifacts):

   El volumen `./artifacts` se monta dentro del contenedor en `/app/artifacts`.

CI / Publicación de imagen:

 - El workflow de GitHub Actions en `.github/workflows/ci.yml` ejecuta tests y,
   en la rama `main`, construye y publica la imagen a Docker Hub si existen los
   secretos `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN` en el repositorio.

Notas de seguridad y producción:

 - Esta es una demo. No use las credenciales ni el sistema de auth en producción
   sin revisiones de seguridad.
 - Para producción considere añadir un reverse-proxy (nginx/Caddy) y TLS.

