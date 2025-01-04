
'''
ARCHIVO DONDE SE EJECUTAN LOS EXPERIMENTOS DE FILTROS DE IMÁGENES
'''

#Librerias:
import pyopencl as cl
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from collections import defaultdict

#Archivos aplicar filtros
import funciones_ejecutar_kernel_filtros as ff
import kernels_filtros_imagenes as kernel
import filtros as f
import funciones_experimento_filtros as  ex

#Ruta del archivo (MODIFICAR SI ES NECESARIO)
path="C:/Users/Eevee"

# Datos GPU (ADAPTAR SEGÚN EL DISPOSITIVO)
compute_units = 82  
processing_elements = 128

# Datos DEVICE
device_type = cl.device_type.GPU 

#Lista de las imagenes:
lista_paths = [
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen64x64.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen128x128.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen640x480.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen800x600.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen720x1280.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen1920x1080.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen2160x3840.jpg"),
       os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen8000x6000.jpg")]

image_names = [
    "imagen64x64.jpg",
    "imagen128x128.jpg",
    "imagen640x480.jpg",
    "imagen800x600.jpg",
    "imagen720x1280.jpg",
    "imagen1920x1080.jpg",
    "imagen2160x3840.jpg",
    "imagen8000x6000.jpg"
]

'''
EXPERIMENTO 1- Mejor local size: Para distintos filtros, obtener los tiempos de ejecución y determinar el mejor
 local size para cada imagen. Para cada filtro, dos tablas con los resultados y los mejores resultados, 
 además de 3 gráficos.
'''

def obtener_local_size(device_type, lista_paths, compute_units: int, processing_elements: int):
    """
    Obtiene el tamaño local adecuado para los kernels en función del dispositivo y las configuraciones proporcionadas.

    Esta función configura y ejecuta experimentos sobre imágenes aplicando diferentes filtros de procesamiento
    (Media, Gaussiano y Sobel). Para cada filtro, se definen los kernels correspondientes y las funciones que
    aplican dichos filtros a las imágenes. Los resultados se guardan en un directorio especificado.

    Parámetros:
    device_type (str): Tipo de dispositivo a utilizar para el procesamiento (por ejemplo, 'CPU' o 'GPU').
    lista_paths (list of str): Lista de rutas de las imágenes sobre las que se realizarán los experimentos.
    compute_units (int): Número de unidades de cómputo disponibles en el dispositivo (por ejemplo, núcleos de la GPU).
    processing_elements (int): Número de elementos de procesamiento por unidad de cómputo.
    
    Salida:
    None: La función ejecuta los experimentos y guarda los resultados en el directorio indicado.
    
    Detalles adicionales:
    - Los filtros se aplican a las imágenes utilizando kernels diseñados para cada tipo de filtro.
    - Los resultados de los experimentos se almacenan en un directorio específico para análisis posteriores.
    """

    # Definir los filtros y sus configuraciones
    filtros = [f.filtro_mean, f.filtro_gaussiani,(f.filtro_sobel_X, f.filtro_sobel_Y)]

    aplicar_filtro_funcs = [
        ff.aplicar_filtro_color,  # Para filtro Mean
        ff.aplicar_filtro_color,  # Para filtro gaussiano
        ff.aplicar_filtro_sobel  # Para filtro Sobel
        ]
    kernel_codes = [
        kernel.kernel_filter_color,  # Kernel para filtro Mean
        kernel.kernel_filter_color,  # Kernel para filtro gaussiano
        kernel.kernel_filter_color_sobel  # Kernel para filtro Sobel
        ]

    filtros_nombres=['mean','gaussian','sobel']

    kernel_names = [
        "kernel_filter_color",
        "kernel_filter_color",
        "kernel_filter_color_sobel"
        ]

    # Directorio base para guardar los gráficos
    base_save_dir = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/")

        # Ejecutar los experimentos
    ex.ejecutar_experimentos(
            lista_paths=lista_paths,
            filtros=filtros,
            filtros_nombres=filtros_nombres,
            aplicar_filtro_funcs=aplicar_filtro_funcs,
            kernel_codes=kernel_codes,
            kernel_names=kernel_names,
            device_type=device_type,
            compute_units=compute_units,
            processing_elements=processing_elements,
            base_save_dir=base_save_dir
        )


'''
EXPERIMENTO 2: Comparar kernels
'''

def comparacion_kernels(device_type,lista_paths):
    # Parámetros de los kernels y funciones
    kernels = [
        kernel.kernel_filter_color,
        kernel.kernel_filter_color_local_ineficiente,
        kernel.kernel_filter_color_local_hebra_maestra,
        kernel.kernel_filter_color_local_organizado,
        kernel.kernel_filter_color_local_organizado_junto
    ]
    kernels_names = [
        "kernel_filter_color",
        "kernel_filter_color_local_ineficiente",
        "kernel_filter_color_local_hebra_maestra",
        "kernel_filter_color_local_organizado",
        "kernel_filter_color_local_organizado_junto"
    ]
    funciones = [
        ff.aplicar_filtro_color,
        ff.aplicar_filtro_local,
        ff.aplicar_filtro_local,
        ff.aplicar_filtro_local,
        ff.aplicar_filtro_local
    ]
    filtro = f.filtro_mean
    local_size = (8,8)  # Cambia esto al tamaño local que desees

    # Directorio base para guardar los gráficos
    base_save_dir = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/COMPARACION_KERNELS/MEM_LOCAL/")

    image_names = [path.split('/')[-1] for path in lista_paths]

    # Ejecutar el experimento
    resultados_finales = ex.experimento_kernels(
        lista_paths,
        filtro,
        kernels,
        kernels_names,
        funciones,
        device_type,
        local_size,
        base_save_dir
    )

    ex.guardar_dataframes_excel(resultados_finales, resultados_finales, base_save_dir, "filtro_color", "experimento_kernels")


'''
EXPERIMENTO 3: EXPERIMENTO: Se comparan kernels que aplican filtros de manera normal o filtros de manera dividida, con y sin memoria local. Se ha fijado un local size, una imagen y se están cambiando el tamaño de los filtros.


OBJETIVO: Determinar que es mejor aplicar filtros de manera dividida, sobre todo para filtros grandes.


OUTPUT:Tabla con los valores, gráfico comparando.


DONDE: EXPERIMENTOS/RESULTADOS/COMPARACION_KERNELS/COMPARACION FILTROS
'''



#KERNELS A USAR
kernels_codes = [kernel.kernel_filter_color_local_organizado, kernel.kernel_filter_color_rectangular,kernel.kernel_filter_color_local_rectangular]

#NOMBRES DE LOS KERNELS

kernels_names = ["kernel_filter_color_local_organizado", "kernel_filter_color_rectangular","kernel_filter_color_local_rectangular"]

#FUNCIONES A APLICAR

funciones = [ff.aplicar_filtro_local, ff.aplicar_filtro_color_dividido,ff.aplicar_filtro_local_dividido]

#IMAGEN PARA APLICAR FILTROS

image_path = os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen800x600.jpg")

#LOCAL SIZE FIJADO

local_size = (8, 8)  # Tamaño local deseado

# Filtros de ejemplo
# Filtros en filtros1 (3x3 a 64x64)
filtros1 = [
    f.filtro_mean1, f.filtro_mean2, f.filtro_mean3, f.filtro_mean4, 
    f.filtro_mean5, f.filtro_mean6, f.filtro_mean7, 
    f.filtro_mean8, f.filtro_mean9, f.filtro_mean10, 
    f.filtro_mean11, f.filtro_mean12, f.filtro_mean13, 
    f.filtro_mean14, f.filtro_mean15, f.filtro_mean16,
    f.filtro_mean17, f.filtro_mean18, f.filtro_mean19, 
    f.filtro_mean20, f.filtro_mean21, f.filtro_mean22, 
    f.filtro_mean23, f.filtro_mean24, f.filtro_mean25,
    f.filtro_mean26, f.filtro_mean27, f.filtro_mean28, 
    f.filtro_mean29, f.filtro_mean30, f.filtro_mean31, f.filtro_mean32
]

# Filtros en filtros2 (versiones divididas)
filtros2 = [
    (f.filtro_mean1X, f.filtro_mean1Y),
    (f.filtro_mean2X, f.filtro_mean2Y),
    (f.filtro_mean3X, f.filtro_mean3Y), 
    (f.filtro_mean4X, f.filtro_mean4Y),
    (f.filtro_mean5X5X, f.filtro_mean5X5Y), 
    (f.filtro_mean6X, f.filtro_mean6Y),
    (f.filtro_mean7x7X, f.filtro_mean7x7Y), 
    (f.filtro_mean8X, f.filtro_mean8Y),
    (f.filtro_mean9x9X, f.filtro_mean9x9Y), 
    (f.filtro_mean10X, f.filtro_mean10Y),
    (f.filtro_mean11X, f.filtro_mean11Y),
    (f.filtro_mean12X, f.filtro_mean12Y),
    (f.filtro_mean13X, f.filtro_mean13Y),
    (f.filtro_mean14X, f.filtro_mean14Y),
    (f.filtro_mean15X, f.filtro_mean15Y),
    (f.filtro_mean16X, f.filtro_mean16Y),
    (f.filtro_mean17X, f.filtro_mean17Y),
    (f.filtro_mean18X, f.filtro_mean18Y),
    (f.filtro_mean19X, f.filtro_mean19Y),
    (f.filtro_mean20X, f.filtro_mean20Y),
    (f.filtro_mean21X, f.filtro_mean21Y),
    (f.filtro_mean22X, f.filtro_mean22Y),
    (f.filtro_mean23X, f.filtro_mean23Y),
    (f.filtro_mean24X, f.filtro_mean24Y),
    (f.filtro_mean25X, f.filtro_mean25Y),
    (f.filtro_mean26X, f.filtro_mean26Y),
    (f.filtro_mean27X, f.filtro_mean27Y),
    (f.filtro_mean28X, f.filtro_mean28Y),
    (f.filtro_mean29X, f.filtro_mean29Y),
    (f.filtro_mean30X, f.filtro_mean30Y),
    (f.filtro_mean31X, f.filtro_mean31Y),
    (f.filtro_mean32X, f.filtro_mean32Y)
    ]  # Para filtros divididos de 3x3 hasta 64x64

#LLAMAR A LA FUNCION 

save_path = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/COMPARACION_KERNELS/COMPARACION FILTROS/")
df_resultados = ex.comparar_filtros(kernels_codes, kernels_names, funciones, image_path, local_size, cl.device_type.GPU, filtros1, filtros2,save_path)




'''
EXPERIMENTO 4:EXPERIMENTO:  Para distintos filtros se calculan los tiempos de ejecución al aplicar el filtro. Los tiempos se calculan haciendo un promedio. Se ejecuta el kernel 1000 veces y se calcula dividiendo el tiempo total entre 1000.

OBJETIVO: Estudiar los mejores local sizes para cada imagen.

OUTPUT: Para cada filtro estudiado se obtienen dos tablas, resultados generales y los mejores, aparte de varios gráficos

DONDE: EXPERIMENTOS/RESULTADOS/1000VECES
'''

# Datos DEVICE
device_type = cl.device_type.GPU 

# Definir los filtros y sus configuraciones
filtros = [
    # Filtro Mean
    f.filtro_mean,  
    # Filtro Gaussian
    f.filtro_gaussiani
]

aplicar_filtro_funcs = [
    ff.aplicar_filtro_color_100,  # Para filtro Mean
    ff.aplicar_filtro_color_100  # Para filtro Gaussian
    ]

kernel_codes = [
    kernel.kernel_filter_color,  # Kernel para filtro Mean
    kernel.kernel_filter_color
    ]

kernel_names = [
    "kernel_filter_color",
    "kernel_filter_color",
    ]


base_save_dir = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/1000VECES/")
os.makedirs(base_save_dir, exist_ok=True)
nombres=['mean','gaussian']

    # Ejecutar los experimentos
ex.ejecutar_experimentos(
        lista_paths=lista_paths,
        filtros=filtros,
        filtros_nombres=nombres,
        aplicar_filtro_funcs=aplicar_filtro_funcs,
        kernel_codes=kernel_codes,
        kernel_names=kernel_names,
        device_type=device_type,
        compute_units=compute_units,
        processing_elements=processing_elements,
        base_save_dir=base_save_dir
    )