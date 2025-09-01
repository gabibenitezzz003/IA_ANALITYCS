"""Persistencia de modelos y memoria de embeddings.

Soporta guardar modelos PyTorch (si están presentes) y parámetros numpy.
También maneja una 'memoria' simple basada en archivos NPZ.
"""
import os
import numpy as np
try:
    import torch
    TORCH_INSTALADO = True
except Exception:
    TORCH_INSTALADO = False

def guardar_modelo_numpy(ruta: str, parametros: np.ndarray):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.save(ruta, parametros)

def cargar_modelo_numpy(ruta: str) -> np.ndarray:
    return np.load(ruta + '.npy') if not ruta.endswith('.npy') else np.load(ruta)

def guardar_memoria_npz(ruta: str, claves: list, embeddings: np.ndarray):
    # Normalize ruta to end with .npz
    if not ruta.endswith('.npz'):
        ruta = ruta + '.npz'
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.savez_compressed(ruta, claves=claves, embeddings=embeddings)

def cargar_memoria_npz(ruta: str):
    # Try several variants: exact path, with .npz, without .npz
    candidates = [ruta]
    if not ruta.endswith('.npz'):
        candidates.append(ruta + '.npz')
    else:
        candidates.append(ruta[:-4])
    for p in candidates:
        try:
            data = np.load(p)
            return data['claves'], data['embeddings']
        except Exception:
            continue
    raise FileNotFoundError(f"No se encontró memoria en ninguna de las rutas: {candidates}")


def guardar_modelo_pytorch(ruta: str, modelo) -> None:
    if not TORCH_INSTALADO:
        raise RuntimeError('PyTorch no está instalado en este entorno')
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    torch.save(modelo.state_dict(), ruta)


def cargar_modelo_pytorch(ruta: str, modelo) -> None:
    if not TORCH_INSTALADO:
        raise RuntimeError('PyTorch no está instalado en este entorno')
    modelo.load_state_dict(torch.load(ruta, map_location='cpu'))
    return modelo
