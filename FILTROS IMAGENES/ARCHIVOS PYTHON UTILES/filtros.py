import numpy as np


#FILTROS 3X3:

filtro_mean=np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

filtro_meanX=np.array([[1/3,1/3,1/3]], dtype=np.float32)
filtro_meanY=np.array([[1/3],[1/3],[1/3]], dtype=np.float32)

filtro_gaussianH = np.array([[1/4, 2/4, 1/4]], dtype=np.float32)
filtro_gaussianV = np.array([[1/4], [2/4], [1/4]], dtype=np.float32)

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

filtrosobelX=np.array([[1,0,-1]], dtype=np.float32)
filtrosobelY=np.array([[1],[2],[1]], dtype=np.float32)

#FILTRO 5X5

filtro_mean5x5=np.array([
    [1/25, 1/25, 1/25,1/25,1/25],
    [1/25, 1/25, 1/25,1/25,1/25],
    [1/25, 1/25, 1/25,1/25,1/25],
    [1/25, 1/25, 1/25,1/25,1/25],
    [1/25, 1/25, 1/25,1/25,1/25]
], dtype=np.float32)

filtro_mean5X5X=np.array([[1/5, 1/5, 1/5,1/5,1/5]], dtype=np.float32)
filtro_mean5X5Y=np.array([[1/5],[1/5],[1/5],[1/5],[1/5]], dtype=np.float32)

#FILTRO 7X7

filtro_mean7x7 = np.array([
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49]
], dtype=np.float32)
filtro_mean7x7Y = np.array([[1/7], [1/7], [1/7], [1/7], [1/7], [1/7], [1/7]], dtype=np.float32)
filtro_mean7x7X = np.array([[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]], dtype=np.float32)

#FILTRO 9X9

filtro_mean9x9 = np.array([
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81],
    [1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81, 1/81]
], dtype=np.float32)
filtro_mean9x9X = np.array([[1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]], dtype=np.float32)
filtro_mean9x9Y = np.array([[1/9], [1/9], [1/9], [1/9], [1/9], [1/9], [1/9], [1/9], [1/9]], dtype=np.float32)


#FILTRO 16X16
filtro_mean16x16 = np.full((16, 16), 1/256, dtype=np.float32)  # 256 es 16*16

filtro_mean16x16X = np.full((1, 16), 1/16, dtype=np.float32)  # Filtro X
filtro_mean16x16Y = np.full((16, 1), 1/16, dtype=np.float32)  # Filtro Y



# FILTROS 11X11 A 33X33

filtro_mean11x11 = np.full((11, 11), 1/(11*11), dtype=np.float32)
filtro_mean13x13 = np.full((13, 13), 1/(13*13), dtype=np.float32)
filtro_mean15x15 = np.full((15, 15), 1/(15*15), dtype=np.float32)
filtro_mean17x17 = np.full((17, 17), 1/(17*17), dtype=np.float32)
filtro_mean19x19 = np.full((19, 19), 1/(19*19), dtype=np.float32)
filtro_mean21x21 = np.full((21, 21), 1/(21*21), dtype=np.float32)
filtro_mean23x23 = np.full((23, 23), 1/(23*23), dtype=np.float32)
filtro_mean25x25 = np.full((25, 25), 1/(25*25), dtype=np.float32)
filtro_mean27x27 = np.full((27, 27), 1/(27*27), dtype=np.float32)
filtro_mean29x29 = np.full((29, 29), 1/(29*29), dtype=np.float32)
filtro_mean31x31 = np.full((31, 31), 1/(31*31), dtype=np.float32)
filtro_mean33x33 = np.full((33, 33), 1/(33*33), dtype=np.float32)


filtro_mean11X = np.full((1, 11), 1/11, dtype=np.float32)
filtro_mean11Y = np.full((11, 1), 1/11, dtype=np.float32)

filtro_mean13X = np.full((1, 13), 1/13, dtype=np.float32)
filtro_mean13Y = np.full((13, 1), 1/13, dtype=np.float32)

filtro_mean15X = np.full((1, 15), 1/15, dtype=np.float32)
filtro_mean15Y = np.full((15, 1), 1/15, dtype=np.float32)

filtro_mean17X = np.full((1, 17), 1/17, dtype=np.float32)
filtro_mean17Y = np.full((17, 1), 1/17, dtype=np.float32)

filtro_mean19X = np.full((1, 19), 1/19, dtype=np.float32)
filtro_mean19Y = np.full((19, 1), 1/19, dtype=np.float32)

filtro_mean21X = np.full((1, 21), 1/21, dtype=np.float32)
filtro_mean21Y = np.full((21, 1), 1/21, dtype=np.float32)

filtro_mean23X = np.full((1, 23), 1/23, dtype=np.float32)
filtro_mean23Y = np.full((23, 1), 1/23, dtype=np.float32)

filtro_mean25X = np.full((1, 25), 1/25, dtype=np.float32)
filtro_mean25Y = np.full((25, 1), 1/25, dtype=np.float32)

filtro_mean27X = np.full((1, 27), 1/27, dtype=np.float32)
filtro_mean27Y = np.full((27, 1), 1/27, dtype=np.float32)

filtro_mean29X = np.full((1, 29), 1/29, dtype=np.float32)
filtro_mean29Y = np.full((29, 1), 1/29, dtype=np.float32)

filtro_mean31X = np.full((1, 31), 1/31, dtype=np.float32)
filtro_mean31Y = np.full((31, 1), 1/31, dtype=np.float32)

filtro_mean33X = np.full((1, 33), 1/33, dtype=np.float32)
filtro_mean33Y = np.full((33, 1), 1/33, dtype=np.float32)