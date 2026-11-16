# pruebaBloque1.py
# Genera un caso, estima d, extrae parches y verifica contra el original.

import numpy as np
from randomizador import generarCaso
from estimacion_interleave import estimarInterleave
from extraccion_parches import extraerParches

rutaImagen = "imagenprueba1.png"   # cualquier imagen RGB suficientemente grande
R, C = 3, 7
semilla = 10

for d in (0, 1, 3, 5):
    mosaico, matrizP, originales = generarCaso(
        rutaImagen, R, C, d, semilla,
        "mosaico_d" + str(d) + ".png",
        "groundTruth_d" + str(d) + ".npz",
    )

    dEstimado, diagnostico = estimarInterleave(mosaico, R, C)
    parches = extraerParches(mosaico, R, C, dEstimado)

    # Verificación: el parche en la posición k del mosaico debe ser
    # el original número perm[k]; con P: original j = parche P[j] del mosaico
    perm = np.load("groundTruth_d" + str(d) + ".npz")["perm"]
    extraccionOk = np.array_equal(parches, originales[perm])

    print("d real:", d, "| d estimado:", dEstimado, "| extracción OK:", extraccionOk)