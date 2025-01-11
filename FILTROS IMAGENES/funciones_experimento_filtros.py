'''
FUNCIONES PARA REALIZAR EXPERIMENTOS CON LOS FILTROS DE IMÁGENES EN LA GPU
'''

#Librerias a importar
import numpy as np
import pandas as pd
import os
import re
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import math

'''
FUNCIONES PARA DETERMINA EL POSIBLE MEJOR LOCAL SIZE
'''

def factorizar(n: int) -> list[tuple[int, int]]:
    """
    Encuentra todos los factores de un número entero y devuelve pares de factores.

    Inputs:
    - n (int): Número entero a factorizar.

    Outputs:
    - list[tuple[int, int]]: Lista de pares de factores (x, y) tal que x * y = n.
    """
    factores = []
    # Iteramos desde 1 hasta la raíz cuadrada del número
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:  # Si `i` es un factor
            factores.append((i, n // i))  # Agregar el par de factores
    return factores


def optimal_local_size(
    global_size: tuple[int, int], 
    max_compute_units: int, 
    processing_elements: int
) -> list[tuple[int, int]]:
    """
    Determina los tamaños de workgroup (local sizes) compatibles con un tamaño global dado.

    Inputs:
    - global_size (tuple[int, int]): Dimensiones del espacio global de hilos (X, Y).
    - max_compute_units (int): Número máximo de unidades de cómputo en el dispositivo.
    - processing_elements (int): Número total de elementos de procesamiento disponibles.

    Outputs:
    - list[tuple[int, int]]: Lista de pares (local_x, local_y) compatibles.
    """
    tam_x = global_size[0]  # Tamaño global en el eje X
    tam_y = global_size[1]  # Tamaño global en el eje Y
    
    # Factorizamos los elementos de procesamiento
    factores = factorizar(processing_elements)  
    # Ejemplo: para 128 devuelve [(1,128), (2,64), (4,32), (8,16)]

    opciones = []  # Lista para almacenar las combinaciones compatibles
    
    # Recorremos los pares de factores
    for factor in factores:
        local_x, local_y = factor

        # Verificamos si son compatibles con el tamaño global
        if tam_x % local_x == 0 and tam_y % local_y == 0:
            opciones.append((local_x, local_y))  # Agregar como opción válida

        # También consideramos la permutación de los factores
        local_x_perm, local_y_perm = factor[::-1]  # Invertir los factores
        if local_x_perm != local_x and tam_x % local_x_perm == 0 and tam_y % local_y_perm == 0:
            opciones.append((local_x_perm, local_y_perm))  # Agregar la permutación válida
    
    return opciones


'''
FUNCIONES PARA GUARDAR DATA FRAMES EN FORMATO EXCEL
'''

# GUARDA DOS DATA FRAMES EN UN ARCHIVO

def guardar_dataframes_excel(resultados, best_results_df, base_save_dir, filtro_nombre, funcion_nombre):
    """
    Guarda dos DataFrames en un archivo Excel con hojas separadas, formateando celdas numéricas y creando directorios si no existen.
    
    Parámetros:
        - resultados: DataFrame con los resultados combinados.
        - best_results_df: DataFrame con los mejores resultados.
        - base_save_dir: Ruta base donde se guardará el archivo Excel.
        - filtro_nombre: Nombre del filtro para definir subcarpeta dentro de base_save_dir.
        - funcion_nombre: Nombre de la función (opcional para estructuras adicionales de directorios).
    """

    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    #funcion_dir = os.path.join(filtro_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Usar xlsxwriter como motor para formatear las celdas durante la escritura
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        # Guardar los DataFrames en hojas separadas
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)
        best_results_df.to_excel(writer, sheet_name='Mejores Resultados', index=True)

        # Acceder al workbook y crear un formato para números con 6 decimales
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados Combinados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

        # Formatear 'Mejores Resultados'
        worksheet = writer.sheets['Mejores Resultados']
        for idx, col in enumerate(best_results_df.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


#GUARDA UN DATA FRAME EN FORMATO EXCEL
    
def guardar_dataframe_excel(resultados,base_save_dir):
    """
    Guarda un DataFrame en un archivo Excel con formato numérico en celdas y crea directorios si no existen.
    
    Parámetros:
        - resultados: DataFrame con los datos a guardar.
        - base_save_dir: Ruta base donde se guardará el archivo Excel.
    """
 
    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Usar xlsxwriter como motor para formatear las celdas durante la escritura
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        # Guardar los DataFrames en hojas separadas
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)
       
        # Acceder al workbook y crear un formato para números con 6 decimales
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados Combinados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


'''
FUNCIONES COMUNES A LOS EXPERIMENTOS
'''

#FUNCION PARA OBTENER EL TAMAÑO DE UNA IMAGEN
def obtener_tamano_imagen(path:str)-> int:
    '''
     Calcula el tamaño total en píxeles de una imagen dada su ruta.
     Parametros:
        path : ruta de la imagen
    Output:
        numero pixesl:int

    '''
    with Image.open(path) as img:
        return img.size[0] * img.size[1]  

# Función para extraer las dimensiones de las imágenes
def extraer_dimensiones(nombre_archivo:str):
    """
    Extrae las dimensiones de un archivo a partir de su nombre utilizando un patrón como '128x128' o '640x480'.
    Args:
        nombre_archivo (str): El nombre del archivo que contiene las dimensiones en formato 'AnchoxAlto', 
                              por ejemplo, 'imagen_128x128.jpg'.
    Returns:
        tuple: Una tupla con las dimensiones extraídas en formato (ancho, alto). 
               Si no se encuentran las dimensiones, devuelve (0, 0).
    """
    # Buscar un patrón como '128x128', '640x480', etc.
    dimensiones = re.findall(r'(\d+)x(\d+)', nombre_archivo)
    if dimensiones:
        # Convertir a entero y devolver como tupla (ancho, alto)
        return tuple(map(int, dimensiones[0]))
    else:
        # Si no se encuentran dimensiones, devolver un valor que permita ordenar correctamente
        return (0, 0)
    
#FUNCION QUE PARA UN LOCAL SIZE CALCULA LOS TIEMPOS DE EJECUCION AL APLICAR UN FILTRO  PARA UN CIERTO KERNEL Y 
# UNA CIERTA FUNCION.
#DEVUELVE UN DATA FRAME CON LOS VALORES CORRESPONDIENTES

def filtros_local_size_fijado(lista_paths:list, filtro:list, aplicar_filtro_func:function, kernel_code:str, kernel_name:str, device_type:str, local_size:tuple)-> pd.DataFrame:
    """
    Aplica un filtro en imágenes utilizando un kernel de OpenCL, con un tamaño de grupo de trabajo (`local_size`) especificado,
    y registra los tiempos de ejecución.

    Args:
        lista_paths (list): Lista de rutas a las imágenes que serán procesadas.
        filtro (tuple or list): Parámetros del filtro. Puede ser un único valor o un par de valores (filtroX, filtroY).
        aplicar_filtro_func (function): Función que aplica el filtro en una imagen y devuelve la imagen resultante y el tiempo de ejecución.
        kernel_code (str): Código fuente del kernel de OpenCL.
        kernel_name (str): Nombre del kernel dentro del código fuente.
        device_type (str): Tipo de dispositivo (CPU, GPU, etc.) en el que se ejecutará el kernel.
        local_size (tuple): Tamaño del grupo de trabajo para la ejecución del kernel.

    Returns:
        pandas.DataFrame: Un DataFrame con los tiempos de ejecución para cada imagen procesada.
                          Las filas corresponden a los nombres de las imágenes y las columnas a los tiempos de ejecución.
    """
    
    # Crear un diccionario para almacenar los tiempos de ejecución
    results = {os.path.basename(path): [] for path in lista_paths}
    
    #Si se trata de un filtro doble , como el caso sobel
    if len(filtro) == 2:
        filtroX, filtroY = filtro[0], filtro[1]

        for path in lista_paths:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    (filtroX, filtroY),
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                results[os.path.basename(path)].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)}: {e}")
                results[os.path.basename(path)].append(None)  # Manejo de error
    #Filtro normal
    else:
        for path in lista_paths:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    filtro,
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                
                results[os.path.basename(path)].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)}: {e}")
                results[os.path.basename(path)].append(None)  # Manejo de error

    # Crear DataFrame de resultados
    results_general = pd.DataFrame.from_dict(results, orient='index', columns=['Execution Time'])
    return results_general


#DADO UN DATA FRAME DETERMINA LOS MEJORES LOCAL SIZES PARA CADA IMAGEN.
def mejores_valores(results_combined:pd.Dataframe)-> pd.Dataframe:
    """
    Encuentra los mejores valores (mínimos) en cada fila de un DataFrame y los tamaños locales asociados.

    Args:
        results_combined (pandas.DataFrame): DataFrame donde las filas representan imágenes y las columnas
                                             representan tiempos de ejecución para diferentes tamaños locales.

    Returns:
        pandas.DataFrame: Un DataFrame con los mejores valores por imagen, incluyendo:
                          - Nombre de la imagen.
                          - El valor mínimo.
                          - Los tamaños locales asociados con el valor mínimo.
    """
    best_results = []

    # Iterar sobre las filas del DataFrame
    for index, row in results_combined.iterrows():
        # Encontrar el valor mínimo, ignorando NaN
        min_value = row.min()
        # Encontrar todas las columnas (local sizes) que tienen el valor mínimo
        min_local_sizes = row[row == min_value].index.tolist()

        # Agregar un único resultado por imagen, concatenando los tamaños locales en una cadena
        best_results.append({
            'Image Name': index,
            'Best Value': min_value,
            'Local Size': min_local_sizes  # Mantener los tamaños locales como lista
        })

    # Crear un DataFrame de los mejores resultados
    best_results_df = pd.DataFrame(best_results)
    
    return best_results_df
    

'''
FUNCIONES PARA REALIZAR EL EXPERIMENTO_MEJOR_LOCAL_SIZE Y EXPERIMENTO_1000VECES : PARA DISTINTOS LOCAL SIZES OBTENER LOS TIEMPOS DE EJECUCION Y 
DETERMINAR LOS MEJORES. SE HACE PARA NUMEROSOS FILTROS Y KERNELS
'''


#FUNCION PARA APLICAR A UNA LISTA DE IMAGENES UN FILTRO DADO CON LA FUNCION CORRESPONDIENTE SEGÚN EL KERNEL .

def filtros_generales(lista_paths:list, filtro:list, aplicar_filtro_func:function, kernel_code:str, kernel_name:str, device_type:str)->pd.DataFrame:
    """
    Aplica un filtro a una lista de imágenes para múltiples configuraciones de tamaño local.

    Args:
        lista_paths (list[str]): Lista de rutas de las imágenes.
        filtro: Parámetros del filtro a aplicar.
        aplicar_filtro_func (function): Función que aplica el filtro y devuelve la imagen resultante y el tiempo de ejecución.
        kernel_code (str): Código del kernel OpenCL.
        kernel_name (str): Nombre del kernel.
        device_type (str): Tipo de dispositivo (e.g., 'CPU', 'GPU').
    
    Returns:
        pandas.DataFrame: DataFrame con los tiempos de ejecución para cada imagen y tamaño local.
    """
    local_sizes = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    results = {size: [] for size in local_sizes}

    # Procesar cada imagen
    for path in lista_paths:
        for local_size in local_sizes:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    filtro,
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                results[local_size].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                results[local_size].append(None)  # O puedes agregar un valor especial para indicar error

    # Crear DataFrame
    results_general = pd.DataFrame(results, index=[os.path.basename(path) for path in lista_paths])

    # Agregar la columna con los nombres de las imágenes
    results_general.index.name = 'Image Name'

    return results_general

#Aplica un filtro a una lista de imágenes utilizando tamaños locales óptimos.

def filtros_optimos(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_unit, processing_elements):
    """
    Aplica un filtro a una lista de imágenes utilizando tamaños locales óptimos.

    Args:
        lista_paths (list[str]): Lista de rutas de las imágenes.
        filtro: Parámetros del filtro a aplicar.
        aplicar_filtro_func (function): Función que aplica el filtro y devuelve la imagen resultante y el tiempo de ejecución.
        kernel_code (str): Código del kernel OpenCL.
        kernel_name (str): Nombre del kernel.
        device_type (str): Tipo de dispositivo (e.g., 'CPU', 'GPU').
        compute_unit (int): Número de unidades de cómputo del dispositivo.
        processing_elements (int): Número de elementos de procesamiento por unidad de cómputo.

    Returns:
        pandas.DataFrame: DataFrame con los tiempos de ejecución para cada imagen y tamaño local óptimo.
    """
    

    # Inicializar el diccionario para almacenar los resultados
    results = defaultdict(lambda: defaultdict(list))  # Almacena resultados para cada tamaño local

    # Procesar cada imagen
    for path in lista_paths:
        try:
            # Abrir la imagen
            imagen = Image.open(path)
            imagen_np = np.array(imagen).astype(np.uint8)
            tam_x, tam_y = imagen_np.shape[:2]
            global_size = (tam_x, tam_y)

            # Obtener los tamaños locales óptimos
            local_sizes_optimos = optimal_local_size(global_size, compute_unit, processing_elements)

            for local_size in local_sizes_optimos:
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        path,
                        filtro,
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[local_size][os.path.basename(path)] = exec_time
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                    results[local_size][os.path.basename(path)] = None  # O puedes agregar un valor especial para indicar error
        except Exception as e:
            print(f"Error al abrir la imagen {os.path.basename(path)}: {e}")
            # Agregar None para todos los tamaños locales óptimos en caso de error al abrir la imagen
            for local_size in optimal_local_size((1,1), compute_unit, processing_elements):
                results[local_size][os.path.basename(path)] = None

    # Crear DataFrame con los resultados
    results_optimal = pd.DataFrame({size: [results[size].get(os.path.basename(path), None) for path in lista_paths] 
                                         for size in results.keys()}, 
                                     index=[os.path.basename(path) for path in lista_paths])

    # Agregar la columna con los nombres de las imágenes
    results_optimal.index.name = 'Image Name'

    return results_optimal

#FUNCION QUE GRAFICA UN DATA FRAME DONDE EL EJE X SON LAS IAMGENES, EL EJE Y LOS TIEMPOS DE EJECUCION Y CADA LOCAL SIZE ES UNA LINEA DEL GRAFICO
def graficar_tiempos_ejecucion(data:pd.DataFrame, columns_to_plot=None, save_path=None):
    """
    Genera un gráfico de tiempos de ejecución por tamaño de trabajo (local size).

    Args:
        data (pandas.DataFrame): DataFrame con tiempos de ejecución. 
                                 Las columnas representan los tamaños locales.
                                 Las filas representan imágenes.
        columns_to_plot (list[str], optional): Lista de columnas específicas (tamaños locales) a graficar.
                                               Si es None, se grafican todas.
        save_path (str, optional): Ruta donde se guardará el gráfico.
                                   Si es None, se muestra el gráfico en pantalla.

    Returns:
        None
    """
    #Crear figura
    plt.figure(figsize=(12, 8))
    
     # Filtrar columnas si se especificaron
    if columns_to_plot:
        data = data[columns_to_plot]

    # Iterar sobre cada columna del DataFrame (cada local size)
    for local_size in data.columns:
        # Obtener los valores de tiempo correspondientes a cada imagen
        row_values = data[local_size].dropna().values  # Eliminamos NaN
        image_names = data.index[data[local_size].notna()]  # Nombres de las imágenes sin NaN
        
        # Graficar solo si hay datos
        if len(row_values) > 0:
            plt.plot(image_names, row_values, marker='o', label=f'Local Size: {local_size}')

    # Configuraciones de la gráfica
    plt.title('Tiempos de Ejecución por Tamaño de Trabajo')
    plt.xlabel('Nombre de la Imagen')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    #plt.yscale('log')  # Usar escala logarítmica si es necesario
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor legibilidad
    plt.legend(title='Tamaños de Trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Ajustar el diseño

    # Guardar o mostrar la gráfica
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()


#Realiza un experimento de filtros en un conjunto de imágenes, aplicando filtros con diferentes tamaños de trabajo (local sizes).
# Genera gráficos con los resultados y devuelve los mejores resultados.

def experimento_filtros(lista_paths:list[str], filtro:list[float], aplicar_filtro_func:function, kernel_code:str, kernel_names:str, device_type:str, compute_units:int, processing_elements:int, filtro_nombre:str, funcion_nombre:str, base_save_dir='graficos'):
    """
    Realiza un experimento de filtros en un conjunto de imágenes, aplicando filtros con diferentes tamaños de trabajo (local sizes).
    Genera gráficos con los resultados y devuelve los mejores resultados.

    Args:
        lista_paths (list[str]): Lista de rutas a las imágenes a procesar.
        filtro (any): El filtro que se aplicará a las imágenes.
        aplicar_filtro_func (function): Función que aplica el filtro y devuelve la imagen procesada y el tiempo de ejecución.
        kernel_code (str): Código del kernel OpenCL que se utilizará.
        kernel_name (str): Nombre del kernel.
        device_type (str): Tipo de dispositivo (CPU o GPU).
        compute_units (int): Número de unidades de cómputo.
        processing_elements (int): Número de elementos de procesamiento.
        filtro_nombre (str): Nombre del filtro que se usará para los gráficos y directorios.
        funcion_nombre (str): Nombre de la función para el análisis y gráficos.
        base_save_dir (str, optional): Directorio base donde se guardarán los gráficos. Por defecto es 'graficos'.

    Returns:
        tuple: DataFrames con los resultados combinados y los mejores resultados por imagen.
    """
    
    # PARTE 1: APLICAR LOCAL SIZES GENERICAS
    results_general = filtros_generales(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type)

    # PARTE 2: APLICAR LOCAL SIZES OPTIMAS
    results_optimal = filtros_optimos(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_units, processing_elements)

    # PARTE 3: FUSIONAR LOS DOS DATA FRAMES
    results_combined = pd.merge(results_general, results_optimal, on='Image Name', how='outer')

    # Extraer el tamaño de la imagen y calcular el ancho y alto
    results_combined['Width'] = results_combined.index.to_series().str.extract(r'(\d+)x(\d+)').astype(int).apply(lambda x: x[0] * x[1], axis=1)

    # Ordenar por tamaño de imagen
    results_combined = results_combined.sort_values(by='Width')

    # Eliminar la columna temporal 'Width'
    results_combined = results_combined.drop(columns=['Width'])

    # PARTE 4: DEVOLVER LOS MEJORES VALORES PARA CADA FILA
    best_results_df = mejores_valores(results_combined)

    # Crear directorio para guardar los gráficos
    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    os.makedirs(funcion_dir, exist_ok=True)

    # PARTE 5: HACER Y GUARDAR UN GRAFICO COMBINADO
    combined_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(results_combined, save_path=combined_save_path)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_generales.png')
    graficar_tiempos_ejecucion(results_general, save_path=general_save_path)

    # PARTE 7: GRAFICAR SOLO LOS MEJORES RESULTADOS (excluyendo ciertos tamaños locales)
    excluded_columns = [(1, 1), (2, 2), (4, 4)]
    columns = [col for col in results_combined.columns if col not in excluded_columns]
    optimal_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_optimos.png')
    graficar_tiempos_ejecucion(results_combined, columns_to_plot=columns, save_path=optimal_save_path)

    # PARTE 8: Devolver los DataFrames
    return results_combined, best_results_df

#Ejecuta una serie de experimentos aplicando diferentes filtros a un conjunto de imágenes.

def ejecutar_experimentos(lista_paths:list[str], filtros:list[list],filtros_nombres:list[str] ,aplicar_filtro_funcs:list[function], kernel_codes:list[str], kernel_names:list[str], device_type:str, compute_units:int, processing_elements:int, base_save_dir='graficos'):
    """
    Ejecuta una serie de experimentos aplicando diferentes filtros a un conjunto de imágenes.
    Los resultados son guardados en gráficos y archivos Excel.

    Args:
        lista_paths (list[str]): Lista de rutas a las imágenes.
        filtros (list[any]): Lista de filtros a aplicar.
        filtros_nombres (list[str]): Lista con los nombres de los filtros.
        aplicar_filtro_funcs (list[function]): Lista de funciones que aplican los filtros a las imágenes.
        kernel_codes (list[str]): Lista de códigos de los kernels OpenCL.
        kernel_names (list[str]): Lista de nombres de los kernels OpenCL.
        device_type (str): Tipo de dispositivo (CPU o GPU).
        compute_units (int): Número de unidades de cómputo.
        processing_elements (int): Número de elementos de procesamiento.
        base_save_dir (str, optional): Directorio base para guardar los gráficos. Por defecto es 'graficos'.

    Returns:
        None
    """

    # Verificar que todas las listas tengan la misma longitud
    assert len(filtros) == len(aplicar_filtro_funcs) == len(kernel_codes) == len(kernel_names), "Las listas de filtros, funciones, kernels y nombres deben tener la misma longitud."

    for i, filtro in enumerate(filtros):
        aplicar_filtro_func = aplicar_filtro_funcs[i]
        kernel_code = kernel_codes[i]
        kernel_name = kernel_names[i]
        filtro_nombre = filtros_nombres[i] # Puedes personalizar el nombre según el filtro

        funcion_nombre = 'funcion_aplicada'  # Puedes personalizar el nombre según la función

        print(f"Ejecutando experimento para {filtro_nombre} con {funcion_nombre}")

        # Ejecutar el experimento de filtros
        resultados, best_results_df = experimento_filtros(
            lista_paths=lista_paths,
            filtro=filtro,
            aplicar_filtro_func=aplicar_filtro_func,
            kernel_code=kernel_code,
            kernel_name=kernel_name,
            device_type=device_type,
            compute_units=compute_units,
            processing_elements=processing_elements,
            filtro_nombre=filtro_nombre,
            funcion_nombre=funcion_nombre,
            base_save_dir=base_save_dir
        )


        # Guardar los DataFrames en Excel
        guardar_dataframes_excel(resultados, best_results_df, base_save_dir, filtro_nombre, funcion_nombre)



'''
FUNCIONES PARA COMPARAR KERNELS Y DETERMINAR EL MEJOR
'''


# Función para graficar los tiempos de ejecución de diferentes kernels para diferentes imágenes.
def graficar_tiempos_ejecucion_kernels(df,save_path=None):
    """
    Función para graficar los tiempos de ejecución de diferentes kernels para diferentes imágenes.
    
    Parámetros:
    df (DataFrame): Un DataFrame con los nombres de las imágenes y los tiempos de ejecución para cada kernel.
    """
    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar cada kernel
    plt.plot(df.index, df["kernel_filter_color"], marker='o', label='Filtro Color')
    plt.plot(df.index, df["kernel_filter_color_local_ineficiente"], marker='o', label='Filtro Color Memoria Local Ineficiente')
    plt.plot(df.index, df["kernel_filter_color_local_hebra_maestra"], marker='o', label='Filtro Color Memoria Local Hebra Maestra')
    plt.plot(df.index, df["kernel_filter_color_local_organizado"], marker='o', label='Filtro Color Memoria Local Organizado')
    # Personalizar el gráfico
    plt.title("Tiempos de Ejecución por Kernel")
    plt.xlabel("Nombre de la Imagen")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor visualización
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Ajustar el gráfico para evitar solapamientos
        # Guardar o mostrar la gráfica
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()

    
#Función para ejecutar un experimento de diferentes kernels, aplicando un filtro específico a cada imagen y guardando los resultados.

def experimento_kernels(lista_paths:list[str], lista_filtro:list[list], lista_kernels:list[str], lista_nombres_kernels:list[str], lista_funciones:list[function], device_type:str, local_size:tuple, base_save_dir:str)->pd.DataFrame:
    """
    Función para ejecutar un experimento de diferentes kernels, aplicando un filtro específico a cada imagen y guardando los resultados.

    Parámetros:
    - lista_paths (list): Rutas de las imágenes a procesar.
    - lista_filtro (list): Filtros a aplicar a las imágenes.
    - lista_kernels (list): Códigos de los kernels a ejecutar.
    - lista_nombres_kernels (list): Nombres de los kernels para etiquetar los resultados.
    - lista_funciones (list): Funciones para aplicar los filtros a las imágenes.
    - device_type (str): Tipo de dispositivo (CPU, GPU, etc.).
    - local_size (int): Tamaño de trabajo local a aplicar en los kernels.
    - base_save_dir (str): Directorio base para guardar los resultados y gráficos.

    Retorna:
    - resultados_finales (DataFrame): DataFrame con los tiempos de ejecución de cada kernel.
    """
    

    # Inicializar el DataFrame de resultados
    resultados_finales = pd.DataFrame()
    
    #Recorrer la lista de kernels

    for i in range(len(lista_kernels)):
        #Obtener los valores
        kernel_code=lista_kernels[i]
        kernel_name=lista_nombres_kernels[i]
        aplicar_filtro_func=lista_funciones[i]
        filtro=lista_filtro[i]

        #Ejecutar el kernel con la funcion que corresponde
        resultados_kernel = filtros_local_size_fijado(
            lista_paths,
            filtro,
            aplicar_filtro_func,
            kernel_code,
            kernel_name,
            device_type,
            local_size
        )
        
        # Asignar resultados al DataFrame final
        resultados_finales[kernel_name] = resultados_kernel['Execution Time']

    # Asignar nombres de las imágenes como filas
    resultados_finales.index = resultados_kernel.index
    resultados_finales.index.name = 'Image Name'
    
        # Guardar los DataFrames en Excel
    guardar_dataframes_excel(resultados_finales,resultados_finales, base_save_dir, 'kernels','kernels')

    # Crear directorio para guardar gráficos si no existe
    os.makedirs(base_save_dir, exist_ok=True)

   
    save_path = os.path.join(base_save_dir, f"KERNELS_tiempos_ejecucion.png")
    image_names = [path.split('/')[-1] for path in lista_paths]
    graficar_tiempos_ejecucion_kernels(resultados_finales, save_path=save_path)

    return resultados_finales

#Función para comparar el rendimiento de diferentes filtros aplicados con distintos kernels.

def comparar_filtros(kernels_codes, kernels_names, funciones, image_path, local_size, device_type, filtros1, filtros2,save_path):
    """
    Función para comparar el rendimiento de diferentes filtros aplicados con distintos kernels.

    Parámetros:
    - kernels_codes (list): Códigos de los kernels.
    - kernels_names (list): Nombres de los kernels.
    - funciones (list): Funciones para aplicar los filtros a las imágenes.
    - image_path (str): Ruta de la imagen a procesar.
    - local_size (int): Tamaño de trabajo local (workgroup size) para los kernels.
    - device_type (str): Tipo de dispositivo (CPU, GPU, etc.).
    - filtros1 (list): Filtros para el caso de "kernel_filter_color_local_organizado".
    - filtros2 (list): Pares de filtros (filtroX, filtroY) para otros casos.
    - save_path (str): Ruta para guardar el gráfico generado y el DataFrame.

    Retorna:
    - df (DataFrame): DataFrame con los tiempos de ejecución de cada kernel.
    """
    
    # Crear un diccionario para almacenar los resultados
    results = {name: [] for name in kernels_names}

    for i in range(len(kernels_codes)):
        kernel_code = kernels_codes[i]
        kernel_name = kernels_names[i]
        aplicar_filtro_func = funciones[i]

        for j in range(len(filtros1)):
            if kernel_name == "kernel_filter_color_local_organizado":
                filtro = filtros1[j]
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        image_path,
                        filtro,
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[kernel_name].append(exec_time)
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(image_path)}: {e}")
                    results[kernel_name].append(None)  # Añadir None en caso de error
            else:
                filtroX, filtroY = filtros2[j]
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        image_path,
                        (filtroX, filtroY),
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[kernel_name].append(exec_time)
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(image_path)}: {e}")
                    results[kernel_name].append(None)  # Añadir None en caso de error

    # Convertir el diccionario en un DataFrame
    df = pd.DataFrame.from_dict(results, orient='index').T
    guardar_dataframe_excel(df,save_path)

    # Renombrar el índice
    df.index = [str(i) for i in range(1, 33)]

    # Crear gráfico de líneas
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        if column=='kernel_filter_color_local_organizado':
           label='Kernel Memoria Local'
           
        elif column=="kernel_filter_color_local_rectangular":
           label='Kernel Memoria Local Separado'

        else:
            label='Kernel Color Separado'
            
        plt.plot(df.index, df[column], marker='o', label=label)

    # Configurar el gráfico
    plt.title('Comparación de Tiempos de Ejecución por Kernel')
    plt.xlabel('Radio del Filtro')
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.legend(title='Kernels')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

# Mostrar gráfico
    plt.savefig(save_path+'comparacion_filtros.png')
    plt.show()

    return df




'''
FUNCIONES PARA REPETIR LOS MISMOS EXPERIMENTOS---> NO SE USA
'''

#FUNCION PARA REPETIR EL MISMO EXPERIMENTO PARA EL MISMO LOCAL SIZE MUCHAS VECES
def repetir_experimento(local_size, kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type):
    valores = []
    
    for i in range(100):
        try:
            # Procesar las imágenes con el tamaño local dado
            results = filtros_local_size_fijado(lista_paths, filter, aplicar_filtro_func, kernel_code, kernel_name, device_type, local_size).T
            valores.append(results)
        except Exception as e:
            # Mostrar un mensaje de error específico si ocurre
            print(f"Error al procesar con local_size {local_size}: {str(e)}")
    
    # Convertir la lista de DataFrames a un solo DataFrame concatenado
    if valores:
        df_final = pd.concat(valores, ignore_index=True)
        # Calcular la media por cada columna
        medias = df_final.mean()
    else:
        medias = pd.Series()  # Si no se procesó ningún resultado, devolver un Series vacío

    return medias

#FUNCION PARA REPETIR EXPERIMENTO PARA UNA LISTA DADA DE LOCAL SIZES. DEVUELVE DOS DATA FRAMES CON LOS VALORES Y VARIOS GRAFICOS.
def repetir_experimento_local_sizes(kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type,base_save_dir):
    local_sizes = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (1, 128), (128, 1), (2, 64), (64, 2), 
                   (32, 4), (4, 32), (8, 16), (16, 8)]

    # Inicializar DataFrame vacío con los nombres de las imágenes como índice
    df_final = pd.DataFrame(index=[os.path.basename(path) for path in lista_paths])

    for local_size in local_sizes:
        # Ejecutar el experimento con el local_size actual
        df_temp = repetir_experimento(local_size, kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type)

        # Verificar que df_temp no esté vacío
        if not df_temp.empty:
            # Convertir el local_size a un string para usarlo como nombre de columna
            local_size_str = f"{local_size[0]}x{local_size[1]}"
            # Asegurar que la columna del DataFrame tenga el nombre correcto
            if isinstance(df_temp, pd.Series):
                df_temp = df_temp.to_frame(name=local_size_str)
            else:
                df_temp.columns = [local_size_str]
            # Alinear por índice (nombres de las imágenes) y concatenar con df_final
            df_final = df_final.join(df_temp, how='outer')

    # Aplicar la función a los índices del DataFrame (los nombres de los archivos)
    df_final['Dimensiones'] = df_final.index.map(extraer_dimensiones)

    # Ordenar el DataFrame por las dimensiones (primero por ancho y luego por alto)
    df_final = df_final.sort_values(by=['Dimensiones'], ascending=True)

   # Eliminar la columna 'Dimensiones' si no quieres que aparezca en el DataFrame final
    df_final = df_final.drop(columns=['Dimensiones'])
    #Guardar data frame
    best_results_df=mejores_valores(df_final)
    
    #Guardar Data frames
    guardar_dataframes_excel(df_final, best_results_df, base_save_dir, 'filter', 'fun')

    #Hacer Graficos
    # PARTE 5: HACER Y GUARDAR UN GRAFICO COMBINADO
    combined_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(df_final, save_path=combined_save_path)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_generales.png')
    #graficar_tiempos_ejecucion(results_general, save_path=general_save_path)


    return df_final


