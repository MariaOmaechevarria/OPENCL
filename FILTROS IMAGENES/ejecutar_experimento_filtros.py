
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

'''
EXPERIMENTO 1- Mejor local size: Para distintos filtros, obtener los tiempos de ejecución y determinar el mejor
 local size para cada imagen. Para cada filtro, dos tablas con los resultados y los mejores resultados, 
 además de 3 gráficos.
'''

def obtener_local_size(path:str,device_type:str, lista_paths:list, compute_units: int, processing_elements: int):
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
EXPERIMENTO 2: Comparar kernels: Compara los kernels de memoria local para determinar cual es el más óptimo. Guarda en la carpeta de resultados
un gráfico y una tabla.
'''

def comparacion_kernels(path:str, device_type:str, lista_paths:list):
    """
    Realiza la comparación de varios kernels en la aplicación de filtros a imágenes utilizando OpenCL.
    
    Esta función ejecuta un experimento con diferentes versiones de kernels para aplicar filtros a imágenes de 
    colores. Los kernels son evaluados para determinar cuál ofrece el mejor rendimiento en términos de eficiencia 
    en el uso de memoria y el procesamiento de las imágenes.

    Parámetros:
    - path (str): Ruta base donde se guardarán los resultados y gráficos generados.
    - device_type (str): Tipo de dispositivo a utilizar (por ejemplo, 'GPU' o 'CPU') para ejecutar los kernels.
    - lista_paths (list): Lista de rutas de las imágenes sobre las que se aplicarán los filtros.

    El experimento compara los siguientes kernels:
    - kernel_filter_color
    - kernel_filter_color_local_ineficiente
    - kernel_filter_color_local_hebra_maestra
    - kernel_filter_color_local_organizado
    
    Cada kernel se aplica con un filtro promedio (filtro_mean), utilizando un tamaño de bloque local de (8, 8).
    
    Los resultados obtenidos del experimento se guardan como archivos Excel para su posterior análisis.
    """

    # Definición de los kernels a comparar
    kernels = [
        kernel.kernel_filter_color,                    # Kernel básico para filtro de color
        kernel.kernel_filter_color_local_ineficiente,  # Kernel con uso ineficiente de memoria local
        kernel.kernel_filter_color_local_hebra_maestra, # Kernel con hebras maestras para acceso a memoria
        kernel.kernel_filter_color_local_organizado   # Kernel con acceso organizado a memoria local
    ]
    
    # Nombres de los kernels para identificación en los resultados
    kernels_names = [
        "kernel_filter_color",
        "kernel_filter_color_local_ineficiente",
        "kernel_filter_color_local_hebra_maestra",
        "kernel_filter_color_local_organizado"
    ]
    
    # Funciones que aplican los filtros a las imágenes en el CPU o GPU
    funciones = [
        ff.aplicar_filtro_color,  # Función para aplicar el filtro básico de color
        ff.aplicar_filtro_local,  # Función para aplicar filtro local (ineficiente)
        ff.aplicar_filtro_local,  # Función para aplicar filtro local (hebra maestra)
        ff.aplicar_filtro_local   # Función para aplicar filtro local (organizado)
    ]
    
    # Tipo de filtro que se aplica (filtro promedio en este caso)
    filtro = [f.filtro_mean, f.filtro_mean, f.filtro_mean, f.filtro_mean]
    
    # Tamaño local utilizado para los work groups en OpenCL
    local_size = (8, 8)  
    
    # Directorio base donde se guardarán los resultados y gráficos
    base_save_dir = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/COMPARACION_KERNELS/MEM_LOCAL/")

    # Nombres de las imágenes, extraídos de las rutas de las imágenes
    image_names = [path.split('/')[-1] for path in lista_paths]

    # Ejecutar el experimento con los kernels definidos
    resultados_finales = ex.experimento_kernels(
        lista_paths,        # Rutas de las imágenes a procesar
        filtro,             # Filtros a aplicar
        kernels,            # Kernels a comparar
        kernels_names,      # Nombres de los kernels
        funciones,          # Funciones que aplican los filtros
        device_type,        # Tipo de dispositivo (CPU o GPU)
        local_size,         # Tamaño local para los work groups
        base_save_dir       # Directorio para guardar los resultados
    )

    # Guardar los resultados del experimento en un archivo Excel
    ex.guardar_dataframes_excel(resultados_finales, resultados_finales, base_save_dir, "filtro_color", "experimento_kernels")


'''
EXPERIMENTO 3: Comparar filtros divididos vs no: Compara la aplicación de un filtro de manera dividida o no,
y con memoria local o no. Para determinar cual es el más óptimo. Guarda en la carpeta de resultados
un gráfico y una tabla.
'''

def filtros_divididos_o_no(path, device_type, image_path):
    """
    Realiza un experimento para comparar el rendimiento de diferentes kernels que aplican filtros de manera 
    tradicional y de manera dividida, con y sin el uso de memoria local. El experimento evalúa el impacto del 
    tamaño del filtro y el uso de memoria local en la eficiencia del procesamiento de imágenes.

    Objetivo: 
    Determinar cuál es la mejor estrategia para aplicar filtros, especialmente para filtros grandes, comparando 
    el enfoque de aplicación dividida versus la aplicación normal de filtros.

    Parámetros:
    - path (str): Ruta base donde se guardarán los resultados generados (gráficos y archivos).
    - device_type (str): Tipo de dispositivo para ejecutar los kernels (por ejemplo, 'CPU' o 'GPU').
    - image_path (str): Ruta de la imagen sobre la que se aplicarán los filtros.

    El experimento compara tres versiones de kernels:
    - kernel_filter_color_local_organizado: Kernel con acceso organizado a memoria local.
    - kernel_filter_color_rectangular: Kernel que aplica un filtro rectangular en la imagen.
    - kernel_filter_color_local_rectangular: Kernel que aplica un filtro rectangular con uso de memoria local.
    
    Además, se utilizan dos tipos de filtros:
    1. Filtros estándar (filtros1) con tamaños de 3x3 a 64x64.
    2. Filtros divididos (filtros2), donde cada filtro es dividido en dos componentes (X e Y).

    Los resultados se guardan en una tabla y se generan gráficos comparando el rendimiento de los kernels.

    Salida:
    - Tabla con los resultados comparativos de los kernels.
    - Gráfico visualizando el rendimiento de cada kernel.
    """

    # Definir los kernels a usar en el experimento
    kernels_codes = [
        kernel.kernel_filter_color_local_organizado,  # Kernel con acceso organizado a memoria local
        kernel.kernel_filter_color_rectangular,       # Kernel que aplica un filtro rectangular
        kernel.kernel_filter_color_local_rectangular  # Kernel que aplica un filtro rectangular con memoria local
    ]

    # Nombres identificativos de los kernels
    kernels_names = [
        "kernel_filter_color_local_organizado",
        "kernel_filter_color_rectangular",
        "kernel_filter_color_local_rectangular"
    ]

    # Funciones que aplican los filtros correspondientes a cada kernel
    funciones = [
        ff.aplicar_filtro_local,                     # Función que aplica filtro con memoria local
        ff.aplicar_filtro_color_dividido,            # Función que aplica filtro dividido (X, Y)
        ff.aplicar_filtro_local_dividido             # Función que aplica filtro dividido con memoria local
    ]

    # Tamaño local del work group en OpenCL (8x8)
    local_size = (8, 8)  # Tamaño local fijado para el experimento

    # Filtros estándar (de 3x3 a 64x64)
    filtros1 = [
        f.filtro_mean1, f.filtro_mean2, f.filtro_mean3, f.filtro_mean4, f.filtro_mean5, f.filtro_mean6, f.filtro_mean7, 
        f.filtro_mean8, f.filtro_mean9, f.filtro_mean10, f.filtro_mean11, f.filtro_mean12, f.filtro_mean13, f.filtro_mean14,
        f.filtro_mean15, f.filtro_mean16,f.filtro_mean17, f.filtro_mean18, f.filtro_mean19, f.filtro_mean20, f.filtro_mean21, 
        f.filtro_mean22, f.filtro_mean23, f.filtro_mean24, f.filtro_mean25,f.filtro_mean26, f.filtro_mean27, f.filtro_mean28, 
        f.filtro_mean29, f.filtro_mean30, f.filtro_mean31, f.filtro_mean32
    ]

    # Filtros divididos (combinaciones de filtros X e Y de distintos tamaños)
    filtros2 = [
        (f.filtro_mean1X, f.filtro_mean1Y), (f.filtro_mean2X, f.filtro_mean2Y),(f.filtro_mean3X, f.filtro_mean3Y),(f.filtro_mean4X, f.filtro_mean4Y),
        (f.filtro_mean5X5X, f.filtro_mean5X5Y),(f.filtro_mean6X, f.filtro_mean6Y),(f.filtro_mean7x7X, f.filtro_mean7x7Y),(f.filtro_mean8X, f.filtro_mean8Y),
        (f.filtro_mean9x9X, f.filtro_mean9x9Y),(f.filtro_mean10X, f.filtro_mean10Y),(f.filtro_mean11X, f.filtro_mean11Y),(f.filtro_mean12X, f.filtro_mean12Y),
        (f.filtro_mean13X, f.filtro_mean13Y),(f.filtro_mean14X, f.filtro_mean14Y),(f.filtro_mean15X, f.filtro_mean15Y),(f.filtro_mean16X, f.filtro_mean16Y),
        (f.filtro_mean17X, f.filtro_mean17Y),(f.filtro_mean18X, f.filtro_mean18Y),(f.filtro_mean19X, f.filtro_mean19Y),(f.filtro_mean20X, f.filtro_mean20Y),
        (f.filtro_mean21X, f.filtro_mean21Y),(f.filtro_mean22X, f.filtro_mean22Y),(f.filtro_mean23X, f.filtro_mean23Y),(f.filtro_mean24X, f.filtro_mean24Y),
        (f.filtro_mean25X, f.filtro_mean25Y),(f.filtro_mean26X, f.filtro_mean26Y),(f.filtro_mean27X, f.filtro_mean27Y),(f.filtro_mean28X, f.filtro_mean28Y),
        (f.filtro_mean29X, f.filtro_mean29Y),(f.filtro_mean30X, f.filtro_mean30Y),(f.filtro_mean31X, f.filtro_mean31Y),(f.filtro_mean32X, f.filtro_mean32Y)
    ]

    # Llamar a la función que realiza la comparación de los filtros
    save_path = os.path.join(path, "OPENCL/FILTROS IMAGENES/EXPERIMENTOS/RESULTADOS/COMPARACION_KERNELS/COMPARACION FILTROS/")
    df_resultados = ex.comparar_filtros(
        kernels_codes,   # Lista de kernels a usar
        kernels_names,   # Nombres de los kernels
        funciones,       # Funciones que aplican los filtros
        image_path,      # Ruta de la imagen a procesar
        local_size,      # Tamaño local de los work groups
        device_type,     # Tipo de dispositivo (CPU o GPU)
        filtros1,        # Filtros estándar (sin dividir)
        filtros2,        # Filtros divididos (X, Y)
        save_path        # Ruta donde guardar los resultados
    )


'''
EXPERIMENTO 4- EXPERIMENTO 1000:  Para distintos filtros se calculan los tiempos de ejecución al aplicar el filtro.
Los tiempos se calculan haciendo un promedio. 
Se ejecuta el kernel 1000 veces y se calcula dividiendo el tiempo total entre 1000.
Para cada filtro estudiado se obtienen dos tablas, resultados generales y los mejores, aparte de varios gráficos.Se guardan
en la carpeta de resultados en 1000 veces.
'''
def mejor_local_size_1000(path:str, device_type:str, lista_paths:list, compute_units:int, processing_elements:int):
    """
    Ejecuta experimentos para evaluar el rendimiento de los filtros `Mean` y `Gaussian` 
    aplicados a un conjunto de imágenes utilizando OpenCL. Los filtros se aplican con el 
    kernel `kernel_filter_color` y se guardan los resultados de los experimentos en un 
    directorio específico.

    Parámetros:
  
    path : Ruta base donde se guardarán los resultados del experimento.
    device_type : cl.device_type. Tipo de dispositivo OpenCL a utilizar. 
    lista_paths : list of str. Lista de rutas a las imágenes sobre las cuales se aplicarán los filtros. 
    compute_units : int. Número de unidades de cómputo disponibles en el dispositivo OpenCL. 
    processing_elements : int. Número de elementos de procesamiento disponibles por unidad de cómputo en el 
        dispositivo. 

    Salida: Nada

    Descripción:
  
    La función `mejor_local_size_1000` realiza experimentos para comparar el rendimiento 
    de dos filtros aplicados a un conjunto de imágenes utilizando OpenCL. Los filtros 
    utilizados son el filtro `Mean` (filtro promedio) y el filtro `Gaussian` (filtro Gaussiano), 
    y ambos se aplican utilizando el kernel `kernel_filter_color`. 

    Los resultados de los experimentos se almacenan en un directorio específico dentro 
    de la ruta proporcionada en el parámetro `path`. Los experimentos se ejecutan 
    utilizando los parámetros `compute_units` y `processing_elements` para configurar 
    el dispositivo OpenCL, y los resultados se guardan en un formato adecuado para 
    su posterior análisis.

    """

    
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
    


#FUNCIÓN QUE EJECUTA TODOS LOS EXPERIMENTOS ANTERIORES

if __name__ == "__main__":
    #Ruta del archivo (MODIFICAR SI ES NECESARIO)
    path="C:/Users/Eevee"

    image_path = os.path.join(path, "OPENCL/FILTROS IMAGENES/IMAGENES/imagen800x600.jpg")

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

    device_type= cl.device_type.GPU 
    obtener_local_size(path,device_type, lista_paths, compute_units, processing_elements)
    comparacion_kernels(path,device_type,lista_paths)
    filtros_divididos_o_no(path,device_type,image_path)
    mejor_local_size_1000(path,device_type,lista_paths,compute_units,processing_elements)

