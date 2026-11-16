import numpy as np

def reconstruir_imagen(parches, P_hat):
    """
    Construye la imagen RGB a partir de los parches y la matriz P_hat.

    Parámetros:
        parches: arreglo (N, H, W, 3) con los parches desordenados.
        P_hat: matriz (R, C) con el ID del parche asignado a cada celda.

    Retorna:
        imagen: imagen RGB reconstruida de dimensiones (R*H, C*W, 3).
    """
    R, C = P_hat.shape
    N, H, W, canales = parches.shape

    if np.any(P_hat < 0) or np.any(P_hat >= N):
        raise ValueError("P_hat contiene índices de parches inválidos")

    if len(np.unique(P_hat)) != R * C:
        raise ValueError("P_hat contiene parches repetidos")

    imagen = np.zeros((R * H, C * W, canales), dtype=parches.dtype)

    for r in range(R):
        for c in range(C):
            id_parche = P_hat[r, c]

            imagen[
                r*H:(r+1)*H,
                c*W:(c+1)*W,
                :
            ] = parches[id_parche]
    # pasar a uint8 si es necesario
    if np.issubdtype(parches.dtype, np.floating):
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)   
    return imagen