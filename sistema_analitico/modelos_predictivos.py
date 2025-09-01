"""Modelos predictivos y pipeline de entrenamiento ligero.

Incluye una clase base y un modelo predictivo sencillo (regresión lineal por lotes)
para demostración e integración con los vectores del MotorLLMAnalitico.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class ModeloBasicoPredictivo:
    """Modelo de regresión lineal trivial con inicialización determinista.

    Este modelo funciona con entradas vectoriales y produce una salida escalar.
    """
    dimensiones_entrada: int = 512
    parametros: Optional[np.ndarray] = None

    def inicializar_parametros(self):
        rng = np.random.RandomState(12345)
        self.parametros = rng.randn(self.dimensiones_entrada).astype(np.float32) * 0.01

    def predecir(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if self.parametros is None:
            self.inicializar_parametros()
        # Alinea dimensiones si es necesario
        if x.shape[0] != self.dimensiones_entrada:
            if x.shape[0] > self.dimensiones_entrada:
                x = x[: self.dimensiones_entrada]
            else:
                x = np.pad(x, (0, self.dimensiones_entrada - x.shape[0]))
        return float(x @ self.parametros)

    def entrenar_por_lotes(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-2, pasos: int = 100):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, d = X.shape
        if self.parametros is None:
            self.inicializar_parametros()
        for _ in range(pasos):
            preds = X @ self.parametros
            grad = (preds - y).dot(X) / n
            self.parametros -= lr * grad
