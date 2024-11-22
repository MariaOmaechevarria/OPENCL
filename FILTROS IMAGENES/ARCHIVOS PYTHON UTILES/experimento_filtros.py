import pyopencl as cl 
import numpy as np
import pandas as pd
import os
import re
from PIL import Image, ImageFilter
from collections import defaultdict
import matplotlib.pyplot as plt

import funciones_filtros as ff
import kernels_filtros_imagenes as kernel
import determinar_mejor_local_size as mejor
import filtros as f


# FUNCIONES PARA GUARDAR DATA FRAMES EN FORMATO EXCEL

def guardar_dataframes_excel(resultados: pd.DataFrame, best_results_df: pd.DataFrame, base_save_dir: str, filtro_nombre: str, funcion_nombre: str) -> None:
    """
    Guarda dos DataFrames en un archivo Excel con diferentes hojas, formateando las celdas para números con 6 decimales.

    :param resultados: DataFrame con resultados combinados.
    :param best_results_df: DataFrame con los mejores resultados.
    :param base_save_dir: Ruta base donde se guardará el archivo.
    :param filtro_nombre: Nombre del filtro.
    :param funcion_nombre: Nombre de la función.
    """
    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')

    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)
        best_results_df.to_excel(writer, sheet_name='Mejores Resultados', index=True)

        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})

        worksheet = writer.sheets['Resultados Combinados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

        worksheet = writer.sheets['Mejores Resultados']
        for idx, col in enumerate(best_results_df.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


def guardar_dataframe_excel(resultados: pd.DataFrame, base_save_dir: str) -> None:
    """
    Guarda un DataFrame en un archivo Excel, formateando las celdas para números con 6 decimales.

    :param resultados: DataFrame con los resultados.
    :param base_save_dir: Ruta base donde se guardará el archivo.
    """
    funcion_dir = os.path.join(base_save_dir)
    os.makedirs(funcion_dir, exist_ok=True)
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')

    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)

        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})

        worksheet = writer.sheets['Resultados Combinados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


# FUNCIONES COMUNES A LOS EXPERIMENTOS

def obtener_tamano_imagen(path: str) -> int:
    """
    Obtiene el tamaño de una imagen como el producto de su ancho y alto.

    :param path: Ruta de la imagen.
    :return: Tamaño de la imagen (ancho * alto).
    """
    with Image.open(path) as img:
        return img.size[0] * img.size[1]  # Ancho * Alto


def extraer_dimensiones(nombre_archivo: str) -> tuple[int, int]:
    """
    Extrae las dimensiones de un nombre de archivo en formato 'Ancho x Alto'.

    :param nombre_archivo: Nombre del archivo.
    :return: Tupla (ancho, alto). Devuelve (0, 0) si no se encuentran dimensiones.
    """
    dimensiones = re.findall(r'(\d+)x(\d+)', nombre_archivo)
    if dimensiones:
        return tuple(map(int, dimensiones[0]))
    else:
        return (0, 0)


def filtros_local_size_fijado(
        lista_paths: list[str],
        filtro: list | tuple,
        aplicar_filtro_func: callable,
        kernel_code: str,
        kernel_name: str,
        device_type: cl.device_type,
        local_size: tuple[int, int]) -> pd.DataFrame:
    """
    Calcula los tiempos de ejecución al aplicar un filtro dado para un kernel y local size.

    :param lista_paths: Lista de rutas a imágenes.
    :param filtro: Filtro aplicado, como una lista o tupla.
    :param aplicar_filtro_func: Función que aplica el filtro.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local de trabajo.
    :return: DataFrame con los tiempos de ejecución por imagen.
    """
    results = {os.path.basename(path): [] for path in lista_paths}

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
            results[os.path.basename(path)].append(None)

    return pd.DataFrame.from_dict(results, orient='index', columns=['Execution Time'])




# DADO UN DATA FRAME DETERMINA LOS MEJORES LOCAL SIZES PARA CADA IMAGEN. DEVUELVE UN DATA FRAME
def mejores_valores(results_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Determina los mejores tamaños locales para cada imagen en un DataFrame.

    :param results_combined: DataFrame con los tiempos de ejecución para distintos tamaños locales.
    :return: DataFrame con las mejores combinaciones de tamaño local para cada imagen.
    """
    best_results = []

    for index, row in results_combined.iterrows():
        min_value = row.min()
        min_local_sizes = row[row == min_value].index.tolist()
        best_results.append({
            'Image Name': index,
            'Best Value': min_value,
            'Local Size': min_local_sizes
        })

    return pd.DataFrame(best_results)


def filtros_generales(
        lista_paths: list[str],
        filtro: list | tuple,
        aplicar_filtro_func: callable,
        kernel_code: str,
        kernel_name: str,
        device_type: cl.device_type) -> pd.DataFrame:
    """
    Aplica un filtro dado a una lista de imágenes con diferentes tamaños locales y calcula los tiempos de ejecución.

    :param lista_paths: Lista de rutas de las imágenes.
    :param filtro: Filtro aplicado (puede ser lista o tupla).
    :param aplicar_filtro_func: Función para aplicar el filtro.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :return: DataFrame con los tiempos de ejecución para cada tamaño local.
    """
    local_sizes = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    results = {size: [] for size in local_sizes}

    for path in lista_paths:
        for local_size in local_sizes:
            try:
                _, exec_time = aplicar_filtro_func(
                    path, filtro, kernel_code, kernel_name, device_type, local_size)
                results[local_size].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                results[local_size].append(None)

    return pd.DataFrame(results, index=[os.path.basename(path) for path in lista_paths], columns=local_sizes)


def filtros_optimos(
        lista_paths: list[str],
        filtro: list | tuple,
        aplicar_filtro_func: callable,
        kernel_code: str,
        kernel_name: str,
        device_type: cl.device_type,
        compute_unit: int,
        processing_elements: int) -> pd.DataFrame:
    """
    Calcula los tiempos de ejecución utilizando los tamaños locales óptimos para cada imagen.

    :param lista_paths: Lista de rutas de las imágenes.
    :param filtro: Filtro aplicado (puede ser lista o tupla).
    :param aplicar_filtro_func: Función para aplicar el filtro.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param compute_unit: Unidades de cómputo del dispositivo.
    :param processing_elements: Elementos de procesamiento por unidad de cómputo.
    :return: DataFrame con los tiempos de ejecución para los tamaños locales óptimos.
    """
    results = defaultdict(lambda: defaultdict(list))

    for path in lista_paths:
        try:
            imagen = Image.open(path)
            imagen_np = np.array(imagen).astype(np.uint8)
            tam_x, tam_y = imagen_np.shape[:2]
            global_size = (tam_x, tam_y)

            local_sizes_optimos = mejor.optimal_local_size(global_size, compute_unit, processing_elements)

            for local_size in local_sizes_optimos:
                try:
                    _, exec_time = aplicar_filtro_func(
                        path, filtro, kernel_code, kernel_name, device_type, local_size)
                    results[local_size][os.path.basename(path)] = exec_time
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                    results[local_size][os.path.basename(path)] = None
        except Exception as e:
            print(f"Error al abrir la imagen {os.path.basename(path)}: {e}")
            for local_size in mejor.optimal_local_size((1, 1), compute_unit, processing_elements):
                results[local_size][os.path.basename(path)] = None

    return pd.DataFrame({size: [results[size].get(os.path.basename(path), None) for path in lista_paths] 
                         for size in results.keys()}, 
                         index=[os.path.basename(path) for path in lista_paths])


def graficar_tiempos_ejecucion(data: pd.DataFrame, columns_to_plot: list = None, save_path: str = None) -> None:
    """
    Genera un gráfico de los tiempos de ejecución para distintos tamaños locales.

    :param data: DataFrame con los tiempos de ejecución.
    :param columns_to_plot: Columnas específicas a graficar (opcional).
    :param save_path: Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(12, 8))

    if columns_to_plot:
        data = data[columns_to_plot]

    for local_size in data.columns:
        row_values = data[local_size].dropna().values
        image_names = data.index[data[local_size].notna()]
        if len(row_values) > 0:
            plt.plot(image_names, row_values, marker='o', label=f'Local Size: {local_size}')

    plt.title('Tiempos de Ejecución por Tamaño de Trabajo')
    plt.xlabel('Nombre de la Imagen')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.xticks(rotation=45)
    plt.legend(title='Tamaños de Trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()


def experimento_filtros(
        lista_paths: list[str],
        filtro: list | tuple,
        aplicar_filtro_func: callable,
        kernel_code: str,
        kernel_name: str,
        device_type: cl.device_type,
        compute_units: int,
        processing_elements: int,
        filtro_nombre: str,
        funcion_nombre: str,
        base_save_dir: str = 'graficos') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza un experimento para aplicar filtros a imágenes y determina los mejores tamaños locales.

    :param lista_paths: Lista de rutas de las imágenes.
    :param filtro: Filtro aplicado (puede ser lista o tupla).
    :param aplicar_filtro_func: Función para aplicar el filtro.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param compute_units: Unidades de cómputo del dispositivo.
    :param processing_elements: Elementos de procesamiento por unidad de cómputo.
    :param filtro_nombre: Nombre del filtro.
    :param funcion_nombre: Nombre de la función aplicada.
    :param base_save_dir: Ruta base para guardar los resultados.
    :return: Dos DataFrames: resultados combinados y los mejores resultados.
    """
    results_general = filtros_generales(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type)
    results_optimal = filtros_optimos(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_units, processing_elements)
    results_combined = pd.merge(results_general, results_optimal, on='Image Name', how='outer')

    best_results_df = mejores_valores(results_combined)

    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    os.makedirs(funcion_dir, exist_ok=True)

    combined_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(results_combined, save_path=combined_save_path)

    return results_combined, best_results_df


#EJECUTA LA FUNCION EXPERIMENTO_FILTROS PARA UNA LISTA DE FILTROS,KERNELS Y FUNCIONES A APLICAR. DEVUELVE PARA CADA KERNEL DOS TABLAS Y TRES GRAFICOS

def ejecutar_experimentos(lista_paths, filtros,filtros_nombres ,aplicar_filtro_funcs, kernel_codes, kernel_names, device_type, compute_units, processing_elements, base_save_dir='graficos'):
  
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
FUNCIONES COMPARACION KERNELS USADAS EN Pruebas_kernels_filtros_local_size y en Experimento_distintos_filtros
DETERMINAR MEJOR KERNEL 
'''



# DADO UN DATA FRAME CON DISTINTOS KERNELS REALIZA UN GRAFICO DONDE EL EJE X SON LAS DIMENSIONES DE LAS IMAGENES, EL EJE Y LOS TIEMPOS DE EJECUCION Y CADA
# LINEA DEL GFRAFICO ES UN KERNEL DISTINTO

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
    plt.plot(df.index, df["kernel_filter_color_local"], marker='o', label='Filtro Color Memoria Local Ineficiente')
    plt.plot(df.index, df["kernel_filter_color_local2"], marker='o', label='Filtro Color Memoria Local Hebra Maestra')
    plt.plot(df.index, df["kernel_filter_color_local3"], marker='o', label='Filtro Color Memoria Local Organizado')
    plt.plot(df.index, df["kernel_filter_color_local4"], marker='o', label='Filtro Color Memoria Local Pixels a la vez ')
    plt.plot(df.index, df["kernel_filter_color_local_rectangular"], marker='o', label='Filtro Color Memoria Local Dividido')
    plt.plot(df.index, df["kernel_filter_color_rectangular"], marker='o', label='Filtro Color Dividido')
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


#FUNCION QUE DADA UNA LISTA DE KERNELS,LISTA DE FILTROS Y LISTA DE FUNCIONES ,CALCULA PARA CADA UNO LOS TIEMPOS DE EJECUCION CON UN LCOAL SIZE FIADO
#DEVUELVE AL FINAL UN DATA FRAME CON LOS VALORES PARA TODOS LOS KERNELS Y UN GRAFICO

def experimento_kernels(lista_paths, lista_filtro, lista_kernels, lista_nombres_kernels, lista_funciones, device_type, local_size, base_save_dir):
    # Inicializar el DataFrame de resultados
    resultados_finales = pd.DataFrame()

    for i in range(len(lista_kernels)):
        kernel_code=lista_kernels[i]
        kernel_name=lista_nombres_kernels[i]
        aplicar_filtro_func=lista_funciones[i]
        filtro=lista_filtro[i]

    
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

#FUNCION PARA COMPARAR KERNELS , FIJADO UN LOCAL SIZE, CALCULA LOS TIEMPOS DE EJECUCION PARA UNA LISTA DE IMAGENES Y UNA LISTA DE FILTROS
# DEVUELVE UN DATA FRAME CON LOS VALORES Y UN GRAFICO

def comparar_filtros(kernels_codes, kernels_names, funciones, image_path, local_size, device_type, filtros1, filtros2,save_path):
    # Crear un diccionario para almacenar los resultados
    results = {name: [] for name in kernels_names}

    for i in range(len(kernels_codes)):
        kernel_code = kernels_codes[i]
        kernel_name = kernels_names[i]
        aplicar_filtro_func = funciones[i]

        for j in range(len(filtros1)):
            if kernel_name == "kernel_filter_color_local4":
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
    df.index = ['filtro 3x3', 'filtro 5x5', 'filtro 7x7', 'filtro 9x9', 
    'filtro 11x11', 'filtro 13x13', 'filtro 15x15', 
    'filtro 16x16', 'filtro 17x17', 'filtro 19x19', 
    'filtro 21x21', 'filtro 23x23', 'filtro 25x25', 
    'filtro 27x27', 'filtro 29x29', 'filtro 31x31', 
    'filtro 33x33']




    # Crear gráfico de líneas
    plt.figure(figsize=(10, 6))
    for column in df.columns:
     plt.plot(df.index, df[column], marker='o', label=column)

    # Configurar el gráfico
    plt.title('Comparación de Tiempos de Ejecución por Kernel')
    plt.xlabel('Filtros')
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

