import funciones_matrices as fm
import kernels_matrices as km
import numpy as np
import pandas as pd
import pyopencl as cl

import os
import matplotlib.pyplot as plt
import determinar_mejor_local_size as mejor
from collections import defaultdict

'''
DETERMINA LOS LOCAL SIZES ÓPTIMOS Y CALCULA LOS TIEMPOS DE EJECUCIÓN
'''
def local_sizes_optimos(funcion_aplicar, kernel_code: str, kernel_name: str, device_type: cl.device_type,
                        compute_unit: int, processing_elements: int) -> pd.DataFrame:
    """
    Determina los tamaños locales óptimos para un kernel y calcula los tiempos de ejecución
    aplicando la función `funcion_aplicar` para cada combinación de tamaño y dimensión.

    :param funcion_aplicar: Función que aplica el kernel y devuelve el tiempo de ejecución.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param compute_unit: Número de unidades de cómputo del dispositivo.
    :param processing_elements: Número de elementos de procesamiento por unidad de cómputo.
    :return: DataFrame con los tiempos de ejecución para cada tamaño local y dimensión.
    """
    local_sizes_optimos = mejor.optimal_local_size((128, 128), compute_unit, processing_elements)
    columns = [2 ** i for i in range(1, 14)]  # Dimensiones de 2 a 8192
    index = [f"({i}/{j})" for i, j in local_sizes_optimos]

    results_df = pd.DataFrame(index=index, columns=columns)

    for local_size in local_sizes_optimos:
        dim = 2
        while dim <= 8192:
            A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
            B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
            try:
                exec_time, C = funcion_aplicar(dim, local_size, device_type, kernel_code, kernel_name, A, B)
                results_df.loc[f"({local_size[0]}/{local_size[1]})", dim] = exec_time
            except Exception as e:
                print(f"Error al procesar con tamaño local {local_size}: {e}")
                results_df.loc[f"({local_size[0]}/{local_size[1]})", dim] = None
            dim *= 2
            del A, B

    return results_df


'''
APLICA UN KERNEL PARA DIFERENTES LOCAL SIZES CUADRADOS
'''
def aplicar_kernel_local_sizes(kernel_code: str, kernel_name: str, device_type: cl.device_type,
                               funcion_aplicar) -> pd.DataFrame:
    """
    Aplica un kernel para tamaños locales cuadrados (1x1, 2x2, ..., 32x32) y calcula los tiempos de ejecución.

    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param funcion_aplicar: Función que aplica el kernel y devuelve el tiempo de ejecución.
    :return: DataFrame con los tiempos de ejecución para cada tamaño local y dimensión.
    """
    index = [(f"({2 ** i}/{2 ** i})" if i != 0 else "(1/1)") for i in range(0, 5)]
    columns = [2 ** i for i in range(1, 14)]  # Dimensiones de 2 a 8192
    results_df = pd.DataFrame(index=index, columns=columns)

    i = 1
    while i <= 32:
        local_size = (i, i)
        dim = 2 if i == 1 else i
        while dim <= 8192:
            A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
            B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
            exec_time, C = funcion_aplicar(dim, local_size, device_type, kernel_code, kernel_name, A, B)
            results_df.loc[f"({i}/{i})", dim] = exec_time if exec_time is not None else None
            dim *= 2
            del A, B
        i *= 2

    return results_df


'''
GRAFICA LOS TIEMPOS DE EJECUCIÓN DESDE UN DATAFRAME
'''
def graficar_tiempos_ejecucion(data: pd.DataFrame, columns_to_plot: list[str] = None, save_path: str = None):
    """
    Genera un gráfico de los tiempos de ejecución desde un DataFrame.

    :param data: DataFrame con los tiempos de ejecución. Filas representan dimensiones, columnas tamaños locales.
    :param columns_to_plot: Lista de columnas específicas a graficar (opcional).
    :param save_path: Ruta para guardar el gráfico (opcional). Si no se proporciona, se mostrará.
    """
    data = data.apply(pd.to_numeric, errors='coerce')  # Convertir a numérico y manejar errores
    data = data.dropna(axis=1, how='all')  # Eliminar columnas completamente vacías

    plt.figure(figsize=(12, 8))

    if columns_to_plot is not None:
        data = data[columns_to_plot]
    else:
        data = data.dropna(axis=0, how='all')  # Eliminar filas completamente vacías

    for local_size in data.columns:
        row_values = data[local_size].dropna().values  # Filtrar NaN
        dim_matrix = data.index[data[local_size].notna()]  # Dimensiones sin NaN
        if len(row_values) > 0:
            plt.plot(dim_matrix, row_values, marker='o', label=f'Tamaño Local: {local_size}')

    plt.title('Tiempos de Ejecución por Tamaño de Trabajo')
    plt.xlabel('Dimensiones de las Matrices')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    ticks = [2 ** i for i in range(1, 14)]  # Ticks para dimensiones 2, 4, ..., 8192
    plt.xticks(ticks, labels=[str(t) for t in ticks], rotation=45)
    plt.xscale('log')  # Escala logarítmica en el eje X
    plt.gca().set_xticks(ticks)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.legend(title='Tamaños de Trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()




# FUNCIÓN QUE DADO UN DATA FRAME DEVUELVE LOS MEJORES LOCAL SIZES PARA CADA DIMENSIÓN
def mejores_valores(results_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Encuentra los mejores local sizes para cada dimensión en un DataFrame.

    :param results_combined: DataFrame donde las filas son dimensiones y las columnas son local sizes.
    :return: DataFrame con las mejores combinaciones de local sizes y sus valores mínimos.
    """
    results_combined = results_combined.apply(pd.to_numeric, errors='coerce')  # Convertir a numérico
    
    best_results = []
    for index, row in results_combined.iterrows():
        min_value = row.min()  # Valor mínimo ignorando NaN
        min_local_sizes = row[row == min_value].index.tolist()  # Local sizes con valor mínimo
        
        best_results.append({
            'Dimension Matrix': index,
            'Best Value': min_value,
            'Local Size': min_local_sizes
        })
    
    return pd.DataFrame(best_results)


# FUNCIÓN PARA APLICAR UN EXPERIMENTO A UN KERNEL Y UNA FUNCIÓN
def experimento_matrices(funcion_aplicar, kernel_code: str, kernel_name: str, device_type: cl.device_type, 
                         compute_units: int, processing_elements: int, funcion_nombre: str, 
                         base_save_dir: str = 'graficos') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza un experimento sobre un kernel, probando diferentes local sizes y generando gráficos.

    :param funcion_aplicar: Función que aplica el kernel y devuelve el tiempo de ejecución.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param compute_units: Número de unidades de cómputo del dispositivo.
    :param processing_elements: Número de elementos de procesamiento por unidad de cómputo.
    :param funcion_nombre: Nombre de la función asociada al kernel.
    :param base_save_dir: Directorio base para guardar gráficos y tablas.
    :return: Tuple con dos DataFrames: resultados combinados y mejores resultados.
    """
    results_general = aplicar_kernel_local_sizes(kernel_code, kernel_name, device_type, funcion_aplicar)
    results_optimal = local_sizes_optimos(funcion_aplicar, kernel_code, kernel_name, device_type, compute_units, processing_elements)
    
    df_combined = pd.concat([results_general, results_optimal], axis=0)
    best_results_df = mejores_valores(df_combined.T)
    
    guardar_dataframes_excel(df_combined, best_results_df, base_save_dir, funcion_nombre)
    
    funcion_dir = os.path.join(base_save_dir, kernel_name)
    os.makedirs(funcion_dir, exist_ok=True)
    
    graficar_tiempos_ejecucion(df_combined.T, save_path=os.path.join(funcion_dir, 'tiempos_ejecucion_combined.png'))
    graficar_tiempos_ejecucion(results_general.T, save_path=os.path.join(funcion_dir, 'tiempos_ejecucion_generales.png'))
    
    excluded_columns = ['(1/1)', '(2/2)', '(4/4)']
    columns = [col for col in df_combined.T.columns if col not in excluded_columns]
    graficar_tiempos_ejecucion(df_combined.T, columns_to_plot=columns, 
                               save_path=os.path.join(funcion_dir, 'tiempos_ejecucion_optimos.png'))
    
    return df_combined, best_results_df


# FUNCIÓN PARA APLICAR UN EXPERIMENTO A MULTIPLES KERNELS Y FUNCIONES
def ejecutar_experimentos(aplicar_funcs: list, kernel_codes: list[str], kernel_names: list[str], 
                          device_type: cl.device_type, compute_units: int, processing_elements: int, 
                          base_save_dir: str):
    """
    Ejecuta experimentos para múltiples kernels y funciones, generando resultados y gráficos.

    :param aplicar_funcs: Lista de funciones que aplican cada kernel.
    :param kernel_codes: Lista de códigos fuente de los kernels.
    :param kernel_names: Lista de nombres de los kernels.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param compute_units: Número de unidades de cómputo del dispositivo.
    :param processing_elements: Número de elementos de procesamiento por unidad de cómputo.
    :param base_save_dir: Directorio base para guardar gráficos y tablas.
    """
    assert len(aplicar_funcs) == len(kernel_codes) == len(kernel_names), "Las listas deben tener la misma longitud."
    
    for funcion_aplicar, kernel_code, kernel_name in zip(aplicar_funcs, kernel_codes, kernel_names):
        print(f"Ejecutando experimento con {kernel_name}")
        experimento_matrices(funcion_aplicar, kernel_code, kernel_name, device_type, compute_units, 
                             processing_elements, kernel_name, base_save_dir)


# FUNCIÓN PARA COMPARAR DIFERENTES KERNELS
def experimento_kernels(lista_kernels: list[str], lista_nombres_kernels: list[str], lista_funciones: list, 
                        device_type: cl.device_type, local_size: tuple[int, int], 
                        base_save_dir: str) -> pd.DataFrame:
    """
    Compara diferentes kernels para un tamaño local fijo y genera un gráfico.

    :param lista_kernels: Lista de códigos fuente de los kernels.
    :param lista_nombres_kernels: Lista de nombres de los kernels.
    :param lista_funciones: Lista de funciones que aplican cada kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param local_size: Tamaño local fijo.
    :param base_save_dir: Directorio base para guardar gráficos y tablas.
    :return: DataFrame con los tiempos de ejecución para cada kernel y dimensión.
    """
    resultados_finales = pd.DataFrame()
    
    for kernel_code, kernel_name, aplicar_func in zip(lista_kernels, lista_nombres_kernels, lista_funciones):
        resultados_kernel = aplicar_kernel_local_fijado(kernel_code, kernel_name, device_type, aplicar_func, local_size)
        resultados_finales[kernel_name] = resultados_kernel.iloc[0]
    
    resultados_finales.index = resultados_kernel.columns
    resultados_finales.index.name = 'Dim Matrix'
    
    guardar_dataframes_excel(resultados_finales, resultados_finales, base_save_dir, f'{kernel_name}_resultados.xlsx')
    graficar_tiempos_ejecucion_kernels(resultados_finales, save_path=os.path.join(base_save_dir, f"KERNELS_tiempos_ejecucion.png"))
    
    return resultados_finales


# FUNCIÓN PARA GRAFICAR LOS TIEMPOS DE EJECUCIÓN DE DIFERENTES KERNELS
def graficar_tiempos_ejecucion_kernels(df: pd.DataFrame, save_path: str = None):
    """
    Genera un gráfico comparativo de tiempos de ejecución para diferentes kernels.

    :param df: DataFrame con los tiempos de ejecución. Filas son dimensiones, columnas son kernels.
    :param save_path: Ruta para guardar el gráfico (opcional). Si no se proporciona, se mostrará.
    """
    plt.figure(figsize=(10, 6))
    
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', label=column)
    
    plt.title("Tiempos de Ejecución por Kernel")
    plt.xlabel("Dimensiones de la Matriz")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.xscale('log')
    dimensiones = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    plt.xticks(dimensiones, labels=[str(d) for d in dimensiones], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()
    plt.close()


# FUNCIÓN PARA APLICAR UN KERNEL CON UN TAMAÑO LOCAL FIJADO
def aplicar_kernel_local_fijado(kernel_code: str, kernel_name: str, device_type: cl.device_type, 
                                funcion_aplicar, local_size: tuple[int, int]) -> pd.DataFrame:
    """
    Aplica un kernel con un tamaño local fijo para diferentes dimensiones de matrices.

    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param funcion_aplicar: Función que aplica el kernel y devuelve el tiempo de ejecución.
    :param local_size: Tamaño local fijo.
    :return: DataFrame con los tiempos de ejecución para cada dimensión.
    """
    columns = [2 ** i for i in range(3, 14)]  # Dimensiones de 8 a 8192
    results_df = pd.DataFrame(columns=columns)
    
    dim = 8
    while dim <= 8192:
        A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
        B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
        
        exec_time, C = funcion_aplicar(dim, local_size, device_type, kernel_code, kernel_name, A, B)
        results_df[dim] = [exec_time] if exec_time is not None else ["NP"]
        
        dim *= 2
        del A, B  # Liberar memoria
    
    results_df.index = [local_size]
    return results_df





'''
FUNCIÓN UTIL PARA GUARDAR DATA FRAMES
'''

def guardar_dataframes_excel(resultados: pd.DataFrame, best_results_df: pd.DataFrame, base_save_dir: str, funcion_nombre: str):
    """
    Guarda dos DataFrames en un archivo Excel con hojas separadas y formato específico.

    :param resultados: DataFrame con los resultados combinados.
    :param best_results_df: DataFrame con los mejores resultados.
    :param base_save_dir: Directorio base para guardar el archivo.
    :param funcion_nombre: Nombre asociado a la función para organizar los archivos.
    """
    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
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
        float_format = workbook.add_format({'num_format': '0.000000'})  # Formato para 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados Combinados']
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para columnas de datos
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna

        # Formatear 'Mejores Resultados'
        worksheet = writer.sheets['Mejores Resultados']
        for idx, col in enumerate(best_results_df.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")
