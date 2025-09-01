"""Metrics sidecar server for sistema_analitico.

Starts a prometheus_client HTTP server on port 8000 and exposes a few demo metrics.
This runs as a separate container/service so Prometheus can scrape reliably.
"""
import time
from prometheus_client import start_http_server, Gauge, Counter


def main():
    # Demo metrics
    UP = Gauge('sistema_analitico_sidecar_up', 'Indicador que el sidecar de métricas está arriba')
    COUNTER = Counter('sistema_analitico_sidecar_heartbeat_total', 'Heartbeats del sidecar')

    start_http_server(8000)
    UP.set(1)
    # simple heartbeat to show changing metric
    while True:
        COUNTER.inc()
        time.sleep(5)


if __name__ == '__main__':
    main()
