import numpy as np
from scipy.ndimage import sobel

def costo_borde_horizontal(parche_left, parche_right, alfa=0.5):
    """
    Calcula la discrepancia de colocar 'parche_left' a la IZQUIERDA de 'parche_right'.
    
    parche_left: (H, W, 3)
    parche_right: (H, W, 3)
    alfa: Peso para balancear la diferencia de color y la continuidad de gradiente.
    """
    # Validar alpha en el rango [0, 1]
    if not (0 <= alfa <= 1):
        raise ValueError("El parámetro alfa debe estar en el rango [0, 1]")
    
    # Convertir a punto flotante para evitar saturación y errores de tipo
    L = np.asarray(parche_left, dtype=np.float64)
    R = np.asarray(parche_right, dtype=np.float64)
    
    # Columna derecha de L (borde Este) y columna izquierda de R (borde Oeste)
    borde_L = L[:, -1, :]
    borde_R = R[:, 0, :]
    
    # 1. Discrepancia directa de intensidad/color (SAD)
    costo_color = np.mean(np.abs(borde_L - borde_R))
    
    # 2. Comparación de Gradientes Modificados (MGC)
    # Predicción del borde derecho de L hacia R: L[:,-1] + (L[:,-1] - L[:,-2])
    dL = L[:, -1, :] - L[:, -2, :]
    pred_R = borde_L + dL
    costo_mgc_L = np.mean(np.abs(pred_R - borde_R))
    
    # Predicción del borde izquierdo de R hacia L: R[:,0] - (R[:,1] - R[:,0])
    dR = R[:, 1, :] - R[:, 0, :]
    pred_L = borde_R - dR
    costo_mgc_R = np.mean(np.abs(borde_L - pred_L))
    
    costo_gradiente = 0.5 * (costo_mgc_L + costo_mgc_R)
    
    return (1.0 - alfa) * costo_color + alfa * costo_gradiente


def costo_borde_vertical(parche_top, parche_bottom, alfa=0.5):
    """
    Calcula la discrepancia de colocar 'parche_top' ARRIBA de 'parche_bottom'.
    
    parche_top: (H, W, 3)
    parche_bottom: (H, W, 3)
    """
    if not (0 <= alfa <= 1):
        raise ValueError("El parámetro alfa debe estar en el rango [0, 1]")
    
    T = np.asarray(parche_top, dtype=np.float64)
    B = np.asarray(parche_bottom, dtype=np.float64)

    borde_T = T[-1, :, :]
    borde_B = B[0, :, :]
    
    costo_color = np.mean(np.abs(borde_T - borde_B))
    
    dT = T[-1, :, :] - T[-2, :, :]
    pred_B = borde_T + dT
    costo_mgc_T = np.mean(np.abs(pred_B - borde_B))
    
    dB = B[1, :, :] - B[0, :, :]
    pred_T = borde_B - dB
    costo_mgc_B = np.mean(np.abs(borde_T - pred_T))
    
    costo_gradiente = 0.5 * (costo_mgc_T + costo_mgc_B)
    
    return (1.0 - alfa) * costo_color + alfa * costo_gradiente

def costo_borde_horizontal_avanzado(parche_left, parche_right, W_b=1, alfa=0.4):
    L = parche_left.astype(np.float64)
    R = parche_right.astype(np.float64)
    
    # 1. Extraer franjas de ancho W_b en lugar de 1 sola columna
    franja_L = L[:, -W_b:, :]
    franja_R = R[:, :W_b, :]
    
    # Diferencia de intensidad en la franja
    costo_color = np.mean(np.abs(franja_L - franja_R))
    
    # 2. Gradientes 2D usando Sobel en la franja limítrofe
    gx_L = sobel(franja_L, axis=1)
    gy_L = sobel(franja_L, axis=0)
    mag_L = np.sqrt(gx_L**2 + gy_L**2)
    
    gx_R = sobel(franja_R, axis=1)
    gy_R = sobel(franja_R, axis=0)
    mag_R = np.sqrt(gx_R**2 + gy_R**2)
    
    costo_grad = np.mean(np.abs(mag_L - mag_R))
    
    return (1.0 - alfa) * costo_color + alfa * costo_grad

def costo_borde_vertical_avanzado(parche_top, parche_bottom, W_b=4, alfa=0.4):
    """
    Calcula la discrepancia al colocar 'parche_top' ARRIBA de 'parche_bottom'.
    Evalúa una franja de alto W_b píxeles en el borde de contacto (Sur/Norte).
    """
    T = parche_top.astype(np.float64)
    B = parche_bottom.astype(np.float64)
    
    # 1. Extraer franjas de alto W_b (filas del borde Sur de T y borde Norte de B)
    franja_T = T[-W_b:, :, :]  # Últimas W_b filas del parche superior
    franja_B = B[:W_b, :, :]   # Primeras W_b filas del parche inferior
    
    # 2. Diferencia directa de intensidad/color en la franja
    costo_color = np.mean(np.abs(franja_T - franja_B))
    
    # 3. Gradientes 2D usando Sobel en la franja limítrofe
    gx_T = sobel(franja_T, axis=1)
    gy_T = sobel(franja_T, axis=0)
    mag_T = np.sqrt(gx_T**2 + gy_T**2)
    
    gx_B = sobel(franja_B, axis=1)
    gy_B = sobel(franja_B, axis=0)
    mag_B = np.sqrt(gx_B**2 + gy_B**2)
    
    costo_grad = np.mean(np.abs(mag_T - mag_B))
    
    return (1.0 - alfa) * costo_color + alfa * costo_grad
