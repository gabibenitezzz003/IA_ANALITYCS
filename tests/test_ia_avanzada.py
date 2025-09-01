import numpy as np
from sistema_analitico.ia_avanzada import MotorIAAvanzada


def test_entrenar_y_predecir():
    X = np.random.RandomState(1).randn(50, 10).astype('float32')
    y = (X.sum(axis=1) + 0.01 * np.random.randn(50)).astype('float32')
    motor = MotorIAAvanzada(dimension_entrada=10, dimension_oculta=4)
    motor.inicializar_modelo()
    motor.entrenar(X, y, pasos=5)
    p = motor.predecir(X[0])
    assert isinstance(p, float)
