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

import numpy as np

# Función para crear filtros cuadrados en base al radio
def crear_filtro_cuadrado(radio):
    tamaño = 2 * radio + 1
    return np.full((tamaño, tamaño), 1 / (tamaño * tamaño), dtype=np.float32)

# Generar filtros de radios 1 a 32
filtro_mean1 = crear_filtro_cuadrado(1)
filtro_mean2 = crear_filtro_cuadrado(2)
filtro_mean3 = crear_filtro_cuadrado(3)
filtro_mean4 = crear_filtro_cuadrado(4)
filtro_mean5 = crear_filtro_cuadrado(5)
filtro_mean6 = crear_filtro_cuadrado(6)
filtro_mean7 = crear_filtro_cuadrado(7)
filtro_mean8 = crear_filtro_cuadrado(8)
filtro_mean9 = crear_filtro_cuadrado(9)
filtro_mean10 = crear_filtro_cuadrado(10)
filtro_mean11 = crear_filtro_cuadrado(11)
filtro_mean12 = crear_filtro_cuadrado(12)
filtro_mean13 = crear_filtro_cuadrado(13)
filtro_mean14 = crear_filtro_cuadrado(14)
filtro_mean15 = crear_filtro_cuadrado(15)
filtro_mean16 = crear_filtro_cuadrado(16)
filtro_mean17 = crear_filtro_cuadrado(17)
filtro_mean18 = crear_filtro_cuadrado(18)
filtro_mean19 = crear_filtro_cuadrado(19)
filtro_mean20 = crear_filtro_cuadrado(20)
filtro_mean21 = crear_filtro_cuadrado(21)
filtro_mean22 = crear_filtro_cuadrado(22)
filtro_mean23 = crear_filtro_cuadrado(23)
filtro_mean24 = crear_filtro_cuadrado(24)
filtro_mean25 = crear_filtro_cuadrado(25)
filtro_mean26 = crear_filtro_cuadrado(26)
filtro_mean27 = crear_filtro_cuadrado(27)
filtro_mean28 = crear_filtro_cuadrado(28)
filtro_mean29 = crear_filtro_cuadrado(29)
filtro_mean30 = crear_filtro_cuadrado(30)
filtro_mean31 = crear_filtro_cuadrado(31)
filtro_mean32 = crear_filtro_cuadrado(32)


# Filtros 1D en X y Y (de radio 1 a 32)
import numpy as np

# Filtros 1D en X (de radio 1 a 32)
filtro_mean1X = np.full((1, 3), 1/3, dtype=np.float32)
filtro_mean2X = np.full((1, 5), 1/5, dtype=np.float32)
filtro_mean3X = np.full((1, 7), 1/7, dtype=np.float32)
filtro_mean4X = np.full((1, 9), 1/9, dtype=np.float32)
filtro_mean5X = np.full((1, 11), 1/11, dtype=np.float32)
filtro_mean6X = np.full((1, 13), 1/13, dtype=np.float32)
filtro_mean7X = np.full((1, 15), 1/15, dtype=np.float32)
filtro_mean8X = np.full((1, 17), 1/17, dtype=np.float32)
filtro_mean9X = np.full((1, 19), 1/19, dtype=np.float32)
filtro_mean10X = np.full((1, 21), 1/21, dtype=np.float32)
filtro_mean11X = np.full((1, 23), 1/23, dtype=np.float32)
filtro_mean12X = np.full((1, 25), 1/25, dtype=np.float32)
filtro_mean13X = np.full((1, 27), 1/27, dtype=np.float32)
filtro_mean14X = np.full((1, 29), 1/29, dtype=np.float32)
filtro_mean15X = np.full((1, 31), 1/31, dtype=np.float32)
filtro_mean16X = np.full((1, 33), 1/33, dtype=np.float32)
filtro_mean17X = np.full((1, 35), 1/35, dtype=np.float32)
filtro_mean18X = np.full((1, 37), 1/37, dtype=np.float32)
filtro_mean19X = np.full((1, 39), 1/39, dtype=np.float32)
filtro_mean20X = np.full((1, 41), 1/41, dtype=np.float32)
filtro_mean21X = np.full((1, 43), 1/43, dtype=np.float32)
filtro_mean22X = np.full((1, 45), 1/45, dtype=np.float32)
filtro_mean23X = np.full((1, 47), 1/47, dtype=np.float32)
filtro_mean24X = np.full((1, 49), 1/49, dtype=np.float32)
filtro_mean25X = np.full((1, 51), 1/51, dtype=np.float32)
filtro_mean26X = np.full((1, 53), 1/53, dtype=np.float32)
filtro_mean27X = np.full((1, 55), 1/55, dtype=np.float32)
filtro_mean28X = np.full((1, 57), 1/57, dtype=np.float32)
filtro_mean29X = np.full((1, 59), 1/59, dtype=np.float32)
filtro_mean30X = np.full((1, 61), 1/61, dtype=np.float32)
filtro_mean31X = np.full((1, 63), 1/63, dtype=np.float32)
filtro_mean32X = np.full((1, 65), 1/65, dtype=np.float32)

# Filtros 1D en Y (de radio 1 a 32)
filtro_mean1Y = np.full((3, 1), 1/3, dtype=np.float32)
filtro_mean2Y = np.full((5, 1), 1/5, dtype=np.float32)
filtro_mean3Y = np.full((7, 1), 1/7, dtype=np.float32)
filtro_mean4Y = np.full((9, 1), 1/9, dtype=np.float32)
filtro_mean5Y = np.full((11, 1), 1/11, dtype=np.float32)
filtro_mean6Y = np.full((13, 1), 1/13, dtype=np.float32)
filtro_mean7Y = np.full((15, 1), 1/15, dtype=np.float32)
filtro_mean8Y = np.full((17, 1), 1/17, dtype=np.float32)
filtro_mean9Y = np.full((19, 1), 1/19, dtype=np.float32)
filtro_mean10Y = np.full((21, 1), 1/21, dtype=np.float32)
filtro_mean11Y = np.full((23, 1), 1/23, dtype=np.float32)
filtro_mean12Y = np.full((25, 1), 1/25, dtype=np.float32)
filtro_mean13Y = np.full((27, 1), 1/27, dtype=np.float32)
filtro_mean14Y = np.full((29, 1), 1/29, dtype=np.float32)
filtro_mean15Y = np.full((31, 1), 1/31, dtype=np.float32)
filtro_mean16Y = np.full((33, 1), 1/33, dtype=np.float32)
filtro_mean17Y = np.full((35, 1), 1/35, dtype=np.float32)
filtro_mean18Y = np.full((37, 1), 1/37, dtype=np.float32)
filtro_mean19Y = np.full((39, 1), 1/39, dtype=np.float32)
filtro_mean20Y = np.full((41, 1), 1/41, dtype=np.float32)
filtro_mean21Y = np.full((43, 1), 1/43, dtype=np.float32)
filtro_mean22Y = np.full((45, 1), 1/45, dtype=np.float32)
filtro_mean23Y = np.full((47, 1), 1/47, dtype=np.float32)
filtro_mean24Y = np.full((49, 1), 1/49, dtype=np.float32)
filtro_mean25Y = np.full((51, 1), 1/51, dtype=np.float32)
filtro_mean26Y = np.full((53, 1), 1/53, dtype=np.float32)
filtro_mean27Y = np.full((55, 1), 1/55, dtype=np.float32)
filtro_mean28Y = np.full((57, 1), 1/57, dtype=np.float32)
filtro_mean29Y = np.full((59, 1), 1/59, dtype=np.float32)
filtro_mean30Y = np.full((61, 1), 1/61, dtype=np.float32)
filtro_mean31Y = np.full((63, 1), 1/63, dtype=np.float32)
filtro_mean32Y = np.full((65, 1), 1/65, dtype=np.float32)
