import os
import numpy as np
try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

from sistema_analitico.persistencia import guardar_modelo_numpy, cargar_modelo_numpy


def test_guardar_numpy_tmp(tmp_path):
    arr = np.array([1.0, 2.0], dtype='float32')
    ruta = tmp_path / 'params.npy'
    guardar_modelo_numpy(str(ruta), arr)
    carg = cargar_modelo_numpy(str(ruta))
    assert np.allclose(carg, arr)


def test_pytorch_guardar_cargar(tmp_path):
    if not TORCH:
        return
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 1)
    m = M()
    ruta = tmp_path / 's.pt'
    from sistema_analitico.persistencia import guardar_modelo_pytorch, cargar_modelo_pytorch
    guardar_modelo_pytorch(str(ruta), m)
    m2 = M()
    cargar_modelo_pytorch(str(ruta), m2)
    # if no exception, assume success
