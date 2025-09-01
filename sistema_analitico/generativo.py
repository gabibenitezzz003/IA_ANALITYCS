"""Motor generativo: integración opcional con Hugging Face Transformers y fallback.

Provee generación de texto, resúmenes y embeddings. Intenta cargar `transformers`
dinámicamente; si no está disponible usa implementaciones deterministas.
"""
from typing import List, Optional
import hashlib
import numpy as np
import os

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'hf_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)


class MotorGenerativo:
    """Encapsula generación y embeddings.

    Si `transformers` está instalado usa pipelines; si no, usa métodos
    deterministas para demos y pruebas.
    """

    def __init__(self, modelo_texto: str = "gpt2"):
        self.modelo_texto = modelo_texto
        self._hf_generador = None
        self._hf_embedder = None
        self._soporta_hf = None

    def _intentar_cargar_hf(self):
        if self._soporta_hf is not None:
            return self._soporta_hf
        try:
            from transformers import pipeline
            # carga perezosa cuando se pida generación o embeddings
            self._hf_pipeline = pipeline
            # también intentamos cargar sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self._hf_sentence_model = SentenceTransformer
            except Exception:
                self._hf_sentence_model = None
            self._soporta_hf = True
        except Exception:
            self._soporta_hf = False
        return self._soporta_hf

    def generar_texto(self, prompt: str, max_length: int = 128) -> str:
        if self._intentar_cargar_hf():
            pipeline = self._hf_pipeline
            try:
                if self._hf_generador is None:
                    self._hf_generador = pipeline('text-generation', model=self.modelo_texto)
                out = self._hf_generador(prompt, max_length=max_length, do_sample=True)
                return out[0]['generated_text']
            except Exception:
                pass
        # Fallback determinista: construir texto combinando palabras del prompt usando hash
        palabras = (prompt or "").split()
        base = "palabra"
        resultado = prompt + ' '
        rng = hashlib.blake2b(digest_size=8)
        for i in range(max_length // 8):
            rng.update(f":{i}".encode('utf-8'))
            token = int.from_bytes(rng.digest(), 'big') % 1000
            resultado += f"{base}{token} "
        return resultado.strip()

    def resumir_texto(self, texto: str, max_length: int = 128) -> str:
        # Si hay transformers, usar summarization; si no, usar truncamiento inteligente
        if self._intentar_cargar_hf():
            try:
                pipeline = self._hf_pipeline
                summarizer = pipeline('summarization')
                out = summarizer(texto, max_length=max_length)
                return out[0]['summary_text']
            except Exception:
                pass
        # Fallback simple: devolver las oraciones principales (primeras N palabras)
        palabras = texto.split()
        return ' '.join(palabras[:max_length])

    def embedding(self, texto: str, dimensiones: int = 512) -> np.ndarray:
        # Try transformers feature-extraction, else deterministic hashing to vector
        if self._intentar_cargar_hf():
            try:
                # use sentence-transformers if available for fast cached embeddings
                if getattr(self, '_hf_sentence_model', None) is not None:
                    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                    cache_path = os.path.join(_CACHE_DIR, model_name.replace('/', '_'))
                    if self._hf_embedder is None:
                        self._hf_embedder = self._hf_sentence_model(model_name)
                    arr = self._hf_embedder.encode(texto, show_progress_bar=False)
                    arr = np.asarray(arr, dtype=np.float32)
                else:
                    pipeline = self._hf_pipeline
                    if self._hf_embedder is None:
                        self._hf_embedder = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')
                    emb = self._hf_embedder(texto)
                    # pipeline returns nested lists -> average to fixed vector
                    arr = np.asarray(emb).mean(axis=0).astype(np.float32)
                if arr.size >= dimensiones:
                    return arr[:dimensiones]
                else:
                    return np.pad(arr, (0, dimensiones - arr.size))
            except Exception:
                pass
        # Deterministic hash-based vector
        vec = np.zeros((dimensiones,), dtype=np.float32)
        for i in range(dimensiones):
            h = hashlib.blake2b(digest_size=4)
            h.update((texto or "").encode('utf-8'))
            h.update(f":{i}".encode('utf-8'))
            val = int.from_bytes(h.digest(), 'big') % 10000
            vec[i] = (val / 10000.0) * 2 - 1
        return vec
