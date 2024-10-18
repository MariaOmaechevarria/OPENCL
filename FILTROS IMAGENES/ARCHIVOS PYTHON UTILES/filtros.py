import numpy as np
#SUAVIZAR LA IMAGEN
filtro_mean=np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

#SUAVIZAR IMAGEN,ELIMINA LOS VALORES DE ALTAS FRECUENCIAS
filtro_gaussiani=np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)

#ENFOCA
filtro_enfoque=np.array([
    [0, -1, 0],
    [-1, 5, 1],
    [0, -1, 0]
], dtype=np.float32)

#DESENFOCA
filtro_desenfoque=np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

#FILTRO SOBEL: DETECTA BORDES

filtro_sobel_X=np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)

filtro_sobel_Y=np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
