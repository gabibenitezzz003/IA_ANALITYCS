import numpy as np
from sistema_analitico.generativo import MotorGenerativo
from sistema_analitico.persistencia import guardar_memoria_npz, cargar_memoria_npz


def test_generador_y_memoria(tmp_path):
    mg = MotorGenerativo()
    t = 'texto prueba'
    g = mg.generar_texto(t, max_length=20)
    assert isinstance(g, str) and len(g) > 0

    claves = ['uno', 'dos']
    embs = np.random.randn(2, 8).astype('float32')
    ruta = tmp_path / 'mem.npz'
    guardar_memoria_npz(str(ruta), claves, embs)
    k, e = cargar_memoria_npz(str(ruta) + '.npz')
    assert list(k) == claves
    assert e.shape == (2, 8)
