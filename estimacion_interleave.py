# estimacion_interleave.py
# Estima d mirando qué filas y columnas del mosaico deberían ser cero.

import numpy as np


def indicesDeCeros(d, tamCelda, cantidad):
    """
    Índices (en toda la imagen) de las filas o columnas que deben ser cero
    si el interleave es d.
    En cada celda: q líneas al inicio y (d - q) líneas al final.
    """
    q = d // 2
    indices = []
    for i in range(cantidad):
        inicio = i * tamCelda
        for k in range(q):
            indices.append(inicio + k)                  # ceros al inicio
        for k in range(d - q):
            indices.append(inicio + tamCelda - 1 - k)   # ceros al final
    return indices


def cumpleD(mosaico, R, C, d):
    """True si todas las filas y columnas esperadas para d son completamente negras."""
    alto, ancho = mosaico.shape[:2]
    sh = alto // R
    sw = ancho // C

    filas = indicesDeCeros(d, sh, R)
    columnas = indicesDeCeros(d, sw, C)

    filasNegras = np.all(mosaico[filas, :, :] == 0)
    columnasNegras = np.all(mosaico[:, columnas, :] == 0)
    return filasNegras and columnasNegras


def estimarInterleave(mosaico, R, C):
    """
    Devuelve (dEstimado, diagnostico).
    Se prueba de mayor a menor: si d=5 se cumple, d=3 y d=1 también,
    así que nos quedamos con el primero (el más grande) que calce.
    """
    alto, ancho = mosaico.shape[:2]
    if alto % R != 0 or ancho % C != 0:
        raise ValueError("El mosaico no es compatible con R y C")

    diagnostico = {}
    dEstimado = 0   # si ninguno calza, d = 0
    for d in (5, 3, 1):
        diagnostico[d] = cumpleD(mosaico, R, C, d)

    for d in (5, 3, 1):
        if diagnostico[d]:
            dEstimado = d
            break

    return dEstimado, diagnostico