import pandas as pd
import numpy as np
from sistema_analitico.preprocesado import preparar_matriz


def test_preparar_matriz_basico():
    df = pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', None]})
    X, meta = preparar_matriz(df, ['a'], ['b'])
    assert X.shape[0] == 3
    assert 'onehot' in meta['encoders']
