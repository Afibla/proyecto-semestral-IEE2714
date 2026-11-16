import numpy as np

def ensamblar_mosaico_greedy(D_horiz, D_vert, R, C, anclaje_src, anclaje_tgt):
    """
    Reconstruye la grilla de R x C parches.
    
    anclaje_src: (r_s, c_s) posición del parche en la grilla desordenada (0-indexed).
    anclaje_tgt: (r_t, c_t) posición asignada en la grilla reconstruida (0-indexed).
    
    Retorna:
      P_hat: Matriz (R, C) con el ID del parche desordenado asignado a cada celda.
    """
    N = R * C

    if D_horiz.shape != (N, N) or D_vert.shape != (N, N):
        raise ValueError("Las matrices deben tener dimensiones N x N")

    r_s, c_s = anclaje_src
    r_t, c_t = anclaje_tgt

    if not (0 <= r_s < R and 0 <= c_s < C):
        raise ValueError("Anclaje de origen inválido")

    if not (0 <= r_t < R and 0 <= c_t < C):
        raise ValueError("Anclaje de destino inválido")
    
    # ID del parche de origen en la grilla desordenada
    id_ancla = r_s * C + c_s
    
    grid_P = -np.ones((R, C), dtype=int)
    grid_P[r_t, c_t] = id_ancla
    usados = {id_ancla}
    
    while len(usados) < N:
        mejor_num_vecinos = -1
        mejor_costo = np.inf
        mejor_pos = None
        mejor_patch = None
        
        # Buscar celdas vacías adyacentes a piezas ya colocadas
        for r in range(R):
            for c in range(C):
                if grid_P[r, c] != -1:
                    continue  # Celda ocupada
                
                # Identificar vecinos ya colocados
                vecinos = []
                if r > 0 and grid_P[r-1, c] != -1:
                    vecinos.append((D_vert[grid_P[r-1, c]], "arriba"))
                if r < R-1 and grid_P[r+1, c] != -1:
                    vecinos.append((D_vert[:, grid_P[r+1, c]], "abajo"))
                if c > 0 and grid_P[r, c-1] != -1:
                    vecinos.append((D_horiz[grid_P[r, c-1]], "izquierda"))
                if c < C-1 and grid_P[r, c+1] != -1:
                    vecinos.append((D_horiz[:, grid_P[r, c+1]], "derecha"))
                
                if not vecinos:
                    continue

                num_vecinos = len(vecinos)

                # Priorizar posiciones con más restricciones
                if num_vecinos < mejor_num_vecinos:
                    continue

                if num_vecinos > mejor_num_vecinos:
                    mejor_num_vecinos = num_vecinos
                    mejor_costo = np.inf
                    mejor_pos = None
                    mejor_patch = None
                
                # Probar todos los parches libres candidatos
                for k in range(N):
                    if k in usados:
                        continue

                    costos = []
                    for costos_direccion, _ in vecinos:
                        costos.append(costos_direccion[k])

                    costo = np.mean(costos)

                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_pos = (r, c)
                        mejor_patch = k

        if mejor_pos is None:
            raise RuntimeError("No se pudo completar el ensamblaje")

                        
        grid_P[mejor_pos] = mejor_patch
        usados.add(mejor_patch)
    return grid_P