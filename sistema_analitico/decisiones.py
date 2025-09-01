"""Motor de decisiones: reglas, scoring y búsqueda simple.

Incluye una pequeña infraestructura para definir reglas y ejecutar una búsqueda
por políticas que maximicen una métrica (grid search como placeholder para RL).
"""
from typing import List, Dict, Any
import numpy as np


class ReglaAnalitica:
    def __init__(self, nombre: str, condicion, accion: Dict[str, Any], peso: float = 1.0):
        self.nombre = nombre
        self.condicion = condicion  # función que recibe contexto y devuelve bool
        self.accion = accion
        self.peso = peso


class MotorDecisionAvanzado:
    def __init__(self, reglas: List[ReglaAnalitica] = None):
        self.reglas = reglas or []

    def agregar_regla(self, regla: ReglaAnalitica):
        self.reglas.append(regla)

    def evaluar(self, contexto) -> List[Dict[str, Any]]:
        salidas = []
        for r in self.reglas:
            try:
                if r.condicion(contexto):
                    sal = r.accion.copy()
                    sal['puntuacion'] = r.peso
                    sal['regla'] = r.nombre
                    salidas.append(sal)
            except Exception:
                continue
        # ordenar por puntuacion
        salidas.sort(key=lambda x: x.get('puntuacion', 0), reverse=True)
        return salidas

    def buscar_politica(self, espacio_acciones: List[Dict[str, Any]], objetivo, evaluador, max_iter=50):
        """Búsqueda simple por rejilla aleatorizada: devuelve la mejor acción.

        - espacio_acciones: lista de acciones posibles (dictionaries)
        - objetivo: función que toma acción y contexto y devuelve score
        - evaluador: función que ejecuta la acción y devuelve métricas
        """
        mejor = None
        mejor_score = -float('inf')
        for _ in range(max_iter):
            idx = np.random.randint(0, len(espacio_acciones))
            accion = espacio_acciones[idx]
            score = objetivo(accion)
            if score > mejor_score:
                mejor_score = score
                mejor = accion
        return mejor, mejor_score
