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

import numpy as np

# FILTROS DE RADIO 1 A 32 (TAMAÑOS DE FILTRO DE 3x3 A 64x64)

# Filtros cuadrados (de tamaño 3x3 hasta 64x64)
filtro_mean3x3 = np.full((3, 3), 1/(3*3), dtype=np.float32)
filtro_mean5x5 = np.full((5, 5), 1/(5*5), dtype=np.float32)
filtro_mean7x7 = np.full((7, 7), 1/(7*7), dtype=np.float32)
filtro_mean9x9 = np.full((9, 9), 1/(9*9), dtype=np.float32)
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
filtro_mean35x35 = np.full((35, 35), 1/(35*35), dtype=np.float32)
filtro_mean37x37 = np.full((37, 37), 1/(37*37), dtype=np.float32)
filtro_mean39x39 = np.full((39, 39), 1/(39*39), dtype=np.float32)
filtro_mean41x41 = np.full((41, 41), 1/(41*41), dtype=np.float32)
filtro_mean43x43 = np.full((43, 43), 1/(43*43), dtype=np.float32)
filtro_mean45x45 = np.full((45, 45), 1/(45*45), dtype=np.float32)
filtro_mean47x47 = np.full((47, 47), 1/(47*47), dtype=np.float32)
filtro_mean49x49 = np.full((49, 49), 1/(49*49), dtype=np.float32)
filtro_mean51x51 = np.full((51, 51), 1/(51*51), dtype=np.float32)
filtro_mean53x53 = np.full((53, 53), 1/(53*53), dtype=np.float32)
filtro_mean55x55 = np.full((55, 55), 1/(55*55), dtype=np.float32)
filtro_mean57x57 = np.full((57, 57), 1/(57*57), dtype=np.float32)
filtro_mean59x59 = np.full((59, 59), 1/(59*59), dtype=np.float32)
filtro_mean61x61 = np.full((61, 61), 1/(61*61), dtype=np.float32)
filtro_mean63x63 = np.full((63, 63), 1/(63*63), dtype=np.float32)
filtro_mean64x64 = np.full((64, 64), 1/(64*64), dtype=np.float32)

# Filtros 1D en X y Y (de radio 1 a 32)
filtro_mean1X = np.full((1, 1), 1/1, dtype=np.float32)
filtro_mean2X = np.full((1, 2), 1/2, dtype=np.float32)
filtro_mean3X = np.full((1, 3), 1/3, dtype=np.float32)
filtro_mean4X = np.full((1, 4), 1/4, dtype=np.float32)
filtro_mean5X = np.full((1, 5), 1/5, dtype=np.float32)
filtro_mean6X = np.full((1, 6), 1/6, dtype=np.float32)
filtro_mean7X = np.full((1, 7), 1/7, dtype=np.float32)
filtro_mean8X = np.full((1, 8), 1/8, dtype=np.float32)
filtro_mean9X = np.full((1, 9), 1/9, dtype=np.float32)
filtro_mean10X = np.full((1, 10), 1/10, dtype=np.float32)
filtro_mean11X = np.full((1, 11), 1/11, dtype=np.float32)
filtro_mean12X = np.full((1, 12), 1/12, dtype=np.float32)
filtro_mean13X = np.full((1, 13), 1/13, dtype=np.float32)
filtro_mean14X = np.full((1, 14), 1/14, dtype=np.float32)
filtro_mean15X = np.full((1, 15), 1/15, dtype=np.float32)
filtro_mean16X = np.full((1, 16), 1/16, dtype=np.float32)
filtro_mean17X = np.full((1, 17), 1/17, dtype=np.float32)
filtro_mean18X = np.full((1, 18), 1/18, dtype=np.float32)
filtro_mean19X = np.full((1, 19), 1/19, dtype=np.float32)
filtro_mean20X = np.full((1, 20), 1/20, dtype=np.float32)
filtro_mean21X = np.full((1, 21), 1/21, dtype=np.float32)
filtro_mean22X = np.full((1, 22), 1/22, dtype=np.float32)
filtro_mean23X = np.full((1, 23), 1/23, dtype=np.float32)
filtro_mean24X = np.full((1, 24), 1/24, dtype=np.float32)
filtro_mean25X = np.full((1, 25), 1/25, dtype=np.float32)
filtro_mean26X = np.full((1, 26), 1/26, dtype=np.float32)
filtro_mean27X = np.full((1, 27), 1/27, dtype=np.float32)
filtro_mean28X = np.full((1, 28), 1/28, dtype=np.float32)
filtro_mean29X = np.full((1, 29), 1/29, dtype=np.float32)
filtro_mean30X = np.full((1, 30), 1/30, dtype=np.float32)
filtro_mean31X = np.full((1, 31), 1/31, dtype=np.float32)
filtro_mean32X = np.full((1, 32), 1/32, dtype=np.float32)

# Filtros 1D en Y (de radio 1 a 32)
filtro_mean1Y = np.full((1, 1), 1/1, dtype=np.float32)
filtro_mean2Y = np.full((2, 1), 1/2, dtype=np.float32)
filtro_mean3Y = np.full((3, 1), 1/3, dtype=np.float32)
filtro_mean4Y = np.full((4, 1), 1/4, dtype=np.float32)
filtro_mean5Y = np.full((5, 1), 1/5, dtype=np.float32)
filtro_mean6Y = np.full((6, 1), 1/6, dtype=np.float32)
filtro_mean7Y = np.full((7, 1), 1/7, dtype=np.float32)
filtro_mean8Y = np.full((8, 1), 1/8, dtype=np.float32)
filtro_mean9Y = np.full((9, 1), 1/9, dtype=np.float32)
filtro_mean10Y = np.full((10, 1), 1/10, dtype=np.float32)
filtro_mean11Y = np.full((11, 1), 1/11, dtype=np.float32)
filtro_mean12Y = np.full((12, 1), 1/12, dtype=np.float32)
filtro_mean13Y = np.full((13, 1), 1/13, dtype=np.float32)
filtro_mean14Y = np.full((14, 1), 1/14, dtype=np.float32)
filtro_mean15Y = np.full((15, 1), 1/15, dtype=np.float32)
filtro_mean16Y = np.full((16, 1), 1/16, dtype=np.float32)
filtro_mean17Y = np.full((17, 1), 1/17, dtype=np.float32)
filtro_mean18Y = np.full((18, 1), 1/18, dtype=np.float32)
filtro_mean19Y = np.full((19, 1), 1/19, dtype=np.float32)
filtro_mean20Y = np.full((20, 1), 1/20, dtype=np.float32)
filtro_mean21Y = np.full((21, 1), 1/21, dtype=np.float32)
filtro_mean22Y = np.full((22, 1), 1/22, dtype=np.float32)
filtro_mean23Y = np.full((23, 1), 1/23, dtype=np.float32)
filtro_mean24Y = np.full((24, 1), 1/24, dtype=np.float32)
filtro_mean25Y = np.full((25, 1), 1/25, dtype=np.float32)
filtro_mean26Y = np.full((26, 1), 1/26, dtype=np.float32)
filtro_mean27Y = np.full((27, 1), 1/27, dtype=np.float32)
filtro_mean28Y = np.full((28, 1), 1/28, dtype=np.float32)
filtro_mean29Y = np.full((29, 1), 1/29, dtype=np.float32)
filtro_mean30Y = np.full((30, 1), 1/30, dtype=np.float32)
filtro_mean31Y = np.full((31, 1), 1/31, dtype=np.float32)
filtro_mean32Y = np.full((32, 1), 1/32, dtype=np.float32)
