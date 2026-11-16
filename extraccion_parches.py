# extraccion_parches.py
# Recorta los parches útiles del mosaico usando el d estimado.

import numpy as np


def extraerParches(mosaico, R, C, d):
    """
    Devuelve un arreglo de forma (R*C, ph, pw, 3).
    El parche k estaba en la fila r = k // C, columna c = k % C
    de la grilla de ENTRADA (índices desde 0, fila por fila).
    """
    alto, ancho = mosaico.shape[:2]
    sh = alto // R
    sw = ancho // C
    ph = sh - d
    pw = sw - d
    q = d // 2

    parches = []
    for r in range(R):
        for c in range(C):
            fila = r * sh + q
            col = c * sw + q
            parches.append(mosaico[fila:fila + ph, col:col + pw])
    return np.array(parches)