"""Módulos de inteligencia analítica: extracción de contexto, decisiones y flujo.

Incluye implementaciones limpias y deterministas que simulan pipelines complejos
para integración con la interfaz y modelos predictivos.
"""
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import hashlib


@dataclass
class MotorLLMAnalitico:
    """Motor que extrae representaciones desde texto (simulado).

    Proporciona métodos para convertir conocimiento textual en vectores numéricos
    reproducibles que alimentan los modelos predictivos y el motor de decisiones.
    """
    dimensiones: int = 512

    def extraer_contexto(self, texto: str) -> np.ndarray:
        if texto is None:
            texto = ""
        vec = np.zeros((self.dimensiones,), dtype=np.float32)
        for i in range(self.dimensiones):
            h = hashlib.blake2b(digest_size=8)
            h.update(texto.encode('utf-8'))
            h.update(f':{i}'.encode('utf-8'))
            val = int.from_bytes(h.digest(), 'big') % 10000
            vec[i] = (val / 10000.0) * 2 - 1
        return vec


@dataclass
class MotorDecisionAnalitico:
    """Motor que recibe vectores y propone decisiones analíticas.

    Implementa reglas heurísticas y scoring simple para priorizar acciones.
    """

    def evaluar_decisiones(self, vector_contexto: np.ndarray) -> List[Dict[str, Any]]:
        norm = float(np.linalg.norm(vector_contexto) + 1e-9)
        media = float(np.mean(vector_contexto))
        score_base = media / (norm + 1e-6)
        acciones = []
        if score_base > 0.1:
            acciones.append({"accion": "Aumentar inversión", "puntuacion": float(score_base)})
        if score_base < -0.1:
            acciones.append({"accion": "Reducir exposición", "puntuacion": float(-score_base)})
        acciones.append({"accion": "Monitorear métricas", "puntuacion": 0.5})
        return acciones
