"""IA avanzada: integración opcional con PyTorch y fallback determinista.

Provee una API en español para inicializar, entrenar y predecir con un
modelo sencillo. Si PyTorch no está instalado, usa implementaciones
deterministas para mantener la reproductibilidad.
"""
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class MotorIAAvanzada:
    """Motor que encapsula un encoder + predictor.

    Métodos:
    - inicializar_modelo: prepara la arquitectura (PyTorch si disponible)
    - entrenar: entrenamiento sencillo (si PyTorch, usa optimizador)
    - generar_embedding: obtiene embedding de texto o vector
    - predecir: devuelve una predicción escalar
    """

    def __init__(self, dimension_entrada: int = 512, dimension_oculta: int = 256):
        self.dimension_entrada = int(dimension_entrada)
        self.dimension_oculta = int(dimension_oculta)
        self.modelo = None
        self._inicializado_con_torch = False

    def inicializar_modelo(self):
        if TORCH_AVAILABLE:
            class RedSimple(nn.Module):
                def __init__(self, d_in, d_h):
                    super().__init__()
                    self.fc1 = nn.Linear(d_in, d_h)
                    self.act = nn.ReLU()
                    self.fc2 = nn.Linear(d_h, 1)

                def forward(self, x):
                    x = self.act(self.fc1(x))
                    return self.fc2(x).squeeze(-1)

            self.modelo = RedSimple(self.dimension_entrada, self.dimension_oculta)
            # Inicialización determinista si se desea
            for p in self.modelo.parameters():
                nn.init.uniform_(p, -0.01, 0.01)
            self._inicializado_con_torch = True
        else:
            # Fallback simple: parámetros numpy
            rng = np.random.RandomState(123)
            self.pesos = rng.randn(self.dimension_entrada).astype(np.float32) * 0.01
            self._inicializado_con_torch = False

    def generar_embedding(self, vector_entrada: np.ndarray) -> np.ndarray:
        x = np.asarray(vector_entrada, dtype=np.float32)
        if x.shape[0] != self.dimension_entrada:
            if x.shape[0] > self.dimension_entrada:
                x = x[: self.dimension_entrada]
            else:
                x = np.pad(x, (0, self.dimension_entrada - x.shape[0]))
        # En el modo PyTorch, devolvemos el mismo vector; el encoder real sería más complejo
        return x

    def entrenar(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-2, pasos: int = 100):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if self.modelo is None and not hasattr(self, 'pesos'):
            self.inicializar_modelo()

        if self._inicializado_con_torch:
            device = torch.device('cpu')
            self.modelo.to(device)
            opt = torch.optim.SGD(self.modelo.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            Xt = torch.from_numpy(X)
            yt = torch.from_numpy(y)
            for _ in range(pasos):
                opt.zero_grad()
                preds = self.modelo(Xt)
                loss = loss_fn(preds, yt)
                loss.backward()
                opt.step()
        else:
            # Descenso de gradiente simple para el vector de parámetros
            if not hasattr(self, 'pesos'):
                self.inicializar_modelo()
            n = X.shape[0]
            for _ in range(pasos):
                preds = X.dot(self.pesos)
                grad = ((preds - y).dot(X)) / max(1, n)
                self.pesos -= lr * grad

    def predecir(self, vector_entrada: np.ndarray) -> float:
        x = self.generar_embedding(vector_entrada)
        if self._inicializado_con_torch:
            import torch
            xt = torch.from_numpy(x).float()
            self.modelo.eval()
            with torch.no_grad():
                return float(self.modelo(xt))
        else:
            return float(x.dot(self.pesos))
