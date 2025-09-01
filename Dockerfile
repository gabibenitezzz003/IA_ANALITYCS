### Multi-stage Dockerfile: build a small runtime image and run as non-root user
FROM python:3.11-slim as build
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
	&& pip wheel --wheel-dir=/wheels -r requirements.txt

FROM python:3.11-slim as runtime
# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser -s /sbin/nologin appuser
WORKDIR /app
# Copy pre-built wheels and install them without network access (faster, reproducible)
COPY --from=build /wheels /wheels
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy application code
COPY . /app
RUN chown -R appuser:appuser /app /home/appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
EXPOSE 8501
# Simple healthcheck that queries the local Streamlit server (single-line)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD python -c "import urllib.request,sys; resp=urllib.request.urlopen('http://127.0.0.1:8501', timeout=3); sys.exit(0 if resp.getcode()==200 else 1)"
ENTRYPOINT ["streamlit", "run", "sistema_analitico/interfaz_app.py"]
CMD ["--server.port", "8501", "--server.address", "0.0.0.0"]
