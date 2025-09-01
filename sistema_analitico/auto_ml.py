"""AutoML ligero: búsqueda aleatoria y búsqueda por rejilla para optimizar modelos.

Este módulo ofrece una API simple que interactúa con los modelos existentes
para buscar hiperparámetros que maximicen una métrica.
"""
import numpy as np
from typing import Callable, Dict, Any, Tuple


def busqueda_aleatoria(evaluador: Callable[[Dict[str, Any]], float], espacio: Dict[str, list], iteraciones: int = 50) -> Tuple[Dict[str, Any], float]:
    mejor = None
    mejor_score = -float('inf')
    claves = list(espacio.keys())
    for _ in range(iteraciones):
        candidato = {k: np.random.choice(espacio[k]) for k in claves}
        score = evaluador(candidato)
        if score > mejor_score:
            mejor_score = score
            mejor = candidato
    return mejor, mejor_score


def busqueda_rejilla(evaluador: Callable[[Dict[str, Any]], float], espacio: Dict[str, list]) -> Tuple[Dict[str, Any], float]:
    # Simple grid search (can explode en combinaciones, usado con cuidado)
    from itertools import product
    keys = list(espacio.keys())
    mejor = None
    mejor_score = -float('inf')
    for vals in product(*[espacio[k] for k in keys]):
        cand = {k: v for k, v in zip(keys, vals)}
        score = evaluador(cand)
        if score > mejor_score:
            melhor_score = score
            mejor = cand
    return mejor, mejor_score
