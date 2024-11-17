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
