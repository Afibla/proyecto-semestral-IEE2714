import numpy as np
from compatibilidad import costo_borde_horizontal, costo_borde_vertical

def calcular_matrices_compatibilidad(parches, alfa=0.5):
    """
    Construye las matrices de compatibilidad N x N.
    
    D_horiz[i, j]: Costo de ubicar parche[i] a la izquierda de parche[j].
    D_vert[i, j]: Costo de ubicar parche[i] arriba de parche[j].
    """
    parches = np.asarray(parches, dtype=np.float64)
    N = len(parches)
    D_horiz = np.zeros((N, N), dtype=np.float64)
    D_vert = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                D_horiz[i, j] = np.inf
                D_vert[i, j] = np.inf
            else:
                D_horiz[i, j] = costo_borde_horizontal(parches[i], parches[j], alfa)
                D_vert[i, j] = costo_borde_vertical(parches[i], parches[j], alfa)
                
    return D_horiz, D_vert