# randomizador.py
# Genera un caso de prueba a partir de una imagen:
# recorta, divide en parches, agrega franjas de ceros y baraja.

import numpy as np
from PIL import Image


def calcularGeometria(alto0, ancho0, R, C, d):
    """Calcula el tamaño de celda (sh, sw) y de parche útil (ph, pw)."""
    sh = alto0 // R
    sw = ancho0 // C
    ph = sh - d
    pw = sw - d

    # Rechazar configuraciones inválidas (pide el enunciado)
    if d not in (0, 1, 3, 5):
        raise ValueError("d debe ser 0, 1, 3 o 5")
    if ph < 64 or ph > 512 or pw < 64 or pw > 512:
        raise ValueError("Parche fuera del rango 64-512")
    return sh, sw, ph, pw


def cropCentrado(imagen, alto, ancho):
    """Recorta una ventana de alto x ancho centrada en la imagen."""
    alto0, ancho0 = imagen.shape[:2]
    arriba = (alto0 - alto) // 2
    izquierda = (ancho0 - ancho) // 2
    return imagen[arriba:arriba + alto, izquierda:izquierda + ancho]


def dividirEnParches(imagenCrop, R, C, sh, sw, ph, pw, d):
    """
    Corta el parche útil de cada celda.
    El parche empieza q píxeles después del borde de la celda.
    El parche k está en la fila r = k // C, columna c = k % C.
    """
    q = d // 2
    parches = []
    for r in range(R):
        for c in range(C):
            fila = r * sh + q
            col = c * sw + q
            parches.append(imagenCrop[fila:fila + ph, col:col + pw])
    return np.array(parches)


def armarMosaico(parches, R, C, sh, sw, d):
    """Pone los parches sobre un lienzo de ceros (las franjas quedan en 0)."""
    q = d // 2
    ph = parches.shape[1]
    pw = parches.shape[2]
    mosaico = np.zeros((R * sh, C * sw, 3), dtype=np.uint8)
    for r in range(R):
        for c in range(C):
            fila = r * sh + q
            col = c * sw + q
            mosaico[fila:fila + ph, col:col + pw] = parches[r * C + c]
    return mosaico


def construirGroundTruth(perm, R, C):
    """
    perm[i] = parche original que quedó en la posición i del mosaico.
    P[r, c] = posición del mosaico donde está el parche de (r, c).
    P es la inversa de perm.
    """
    inversa = np.zeros(len(perm), dtype=int)
    for i in range(len(perm)):
        inversa[perm[i]] = i
    return inversa.reshape(R, C)


def generarCaso(rutaImagen, R, C, d, semilla, rutaMosaico, rutaGT):
    """
    Función principal: guarda el mosaico (png) y el ground truth (npz).
    Devuelve (mosaico, P, parchesOriginales).
    """
    # 1. Leer imagen
    imagen = np.array(Image.open(rutaImagen).convert("RGB"), dtype=np.uint8)
    alto0, ancho0 = imagen.shape[:2]

    # 2. Geometría y crop
    sh, sw, ph, pw = calcularGeometria(alto0, ancho0, R, C, d)
    imagenCrop = cropCentrado(imagen, R * sh, C * sw)

    # 3. Dividir en parches
    parchesOriginales = dividirEnParches(imagenCrop, R, C, sh, sw, ph, pw, d)

    # 4. Barajar con semilla fija
    generador = np.random.default_rng(semilla)
    perm = generador.permutation(R * C)
    parchesBarajados = parchesOriginales[perm]

    # 5. Armar mosaico con franjas de ceros
    mosaico = armarMosaico(parchesBarajados, R, C, sh, sw, d)

    # 6. Guardar resultados (el ground truth va en un archivo aparte)
    matrizP = construirGroundTruth(perm, R, C)
    Image.fromarray(mosaico).save(rutaMosaico)
    np.savez(rutaGT, P=matrizP, perm=perm, R=R, C=C, d=d, semilla=semilla)

    return mosaico, matrizP, parchesOriginales