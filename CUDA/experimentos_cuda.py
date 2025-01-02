'''
ARCHIVO CON FUNCIONES PARA COMPARAR OPENCL Y CUDA
'''



import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

import pyopencl as cl

import mult_matrices_basico_cuda as cuda
import mult_matrices_basica_opencl as opencl



# FUNCIONES PARA GUARDAR DATAFRAMES EN EXCEL

def guardar_dataframe_excel(resultados: pd.DataFrame, base_save_dir: str, funcion_nombre: str) -> None:
    """
    Guarda un DataFrame en un archivo Excel formateado.

    Inputs:
    - resultados (pd.DataFrame): DataFrame con los resultados a guardar.
    - base_save_dir (str): Ruta base para guardar el archivo Excel.
    - funcion_nombre (str): Nombre de la función que genera los resultados.

    Outputs:
    - None: Guarda el DataFrame en Excel en la ubicación especificada.
    """
    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Guardar con formato numérico de 6 decimales
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados', index=True)
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})
        worksheet = writer.sheets['Resultados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

    print(f"DataFrame guardado en {excel_save_path}")


def guardar_dataframes_excel(
    resultados: pd.DataFrame,
    best_results_df: pd.DataFrame,
    base_save_dir: str,
    funcion_nombre: str
) -> None:
    """
    Guarda dos DataFrames en un archivo Excel con hojas separadas.

    Inputs:
    - resultados (pd.DataFrame): DataFrame con resultados generales.
    - best_results_df (pd.DataFrame): DataFrame con los mejores resultados.
    - base_save_dir (str): Ruta base para guardar el archivo Excel.
    - funcion_nombre (str): Nombre de la función que genera los resultados.

    Outputs:
    - None: Guarda los DataFrames en Excel en la ubicación especificada.
    """
    # Crear estructura de directorios
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Guardar en Excel con formato numérico de 6 decimales
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)
        best_results_df.to_excel(writer, sheet_name='Mejores Resultados', index=True)

        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})

        # Formatear hojas
        worksheet = writer.sheets['Resultados Combinados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

        worksheet = writer.sheets['Mejores Resultados']
        for idx, col in enumerate(best_results_df.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)

    print(f"DataFrames guardados en {excel_save_path}")


# FUNCIÓN PARA COMPARAR OPENCL Y CUDA EN MULTIPLICACIÓN DE MATRICES

def comparar(path: str) -> None:
    """
    Compara los tiempos de ejecución de CUDA y OpenCL para la multiplicación de matrices de distintos tamaños.

    Inputs:
    - path (str): Ruta para guardar los resultados y gráficos.

    Outputs:
    - None: Genera gráficos y guarda resultados en Excel.
    """
    device_type = cl.device_type.GPU  # Usar GPU como dispositivo OpenCL

    # Configuración de local size y block/grid size
    local_size = (8, 8)
    block = (8, 8, 1)

    # Dimensiones de las matrices
    dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    results = []

    for dim in dims:
        # Generar matrices aleatorias
        A = np.random.random(size=(dim, dim)).astype(np.float32)
        B = np.random.random(size=(dim, dim)).astype(np.float32)

        grid = (dim // 8, dim // 8)

        # Medir tiempos de OpenCL y CUDA
        exec_time_cl, C_cl = opencl.mult_mat_basica(dim, local_size, device_type, opencl.MatrixMul_kernel1, "MatrixMul_kernel1", A, B)
        exec_time_cuda, C_cuda = cuda.ejecutar_kernel(dim, A, B, block, grid)

        # Guardar resultados
        results.append({
            "Dimensión": dim,
            "Tiempo OpenCL (s)": exec_time_cl,
            "Tiempo CUDA (s)": exec_time_cuda
        })

    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)

    # Guardar en Excel
    guardar_dataframe_excel(df_results, path, 'comparacion_cuda_opencl')

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Dimensión"], df_results["Tiempo OpenCL (s)"], label="OpenCL", marker='o')
    plt.plot(df_results["Dimensión"], df_results["Tiempo CUDA (s)"], label="CUDA", marker='s')

    # Configurar el gráfico
    plt.xlabel("Dimensión de la Matriz")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución entre OpenCL y CUDA")
    plt.xticks(dims, labels=[str(d) for d in dims], rotation=45)
    plt.xscale('log')
    plt.legend()
    plt.grid(True)

    # Guardar gráfico
    save_path = os.path.join(path, "grafico_cuda_opencl.png")
    plt.savefig(save_path)
    plt.show()


# FUNCIÓN PARA APLICAR EXPERIMENTOS EN LOCAL SIZES COMPLETOS

def aplicar_kernel_local_sizes_completo() -> pd.DataFrame:
    """
    Aplica kernels CUDA para distintas configuraciones de bloque y dimensiones de matrices.

    Outputs:
    - pd.DataFrame: Resultados con tiempos de ejecución por configuración de bloque y dimensiones.
    """
    combinaciones_fijas = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    combinaciones_128 = [(x, 128 // x) for x in range(1, 129) if 128 % x == 0]
    todas_combinaciones = combinaciones_fijas + combinaciones_128

    index = [f"Block ({block[0]}/{block[1]})" for block in todas_combinaciones]
    columns = [2 ** i for i in range(1, 14)]

    results_df = pd.DataFrame(index=index, columns=columns)

    for block in todas_combinaciones:
        block_x, block_y = block
        block_size = block_x * block_y

        for dim in columns:
            if block_size > dim * dim:
                results_df.loc[f"Block ({block_x}/{block_y})", dim] = "NaN"
                continue

            A = np.random.random(size=(dim, dim)).astype(np.float32)
            B = np.random.random(size=(dim, dim)).astype(np.float32)

            grid_x = math.ceil(dim / block_x)
            grid_y = math.ceil(dim / block_y)

            block_value = (block_x, block_y, 1)
            grid_value = (grid_x, grid_y)

            try:
                exec_time, _ = cuda.ejecutar_kernel(dim, A, B, block_value, grid_value)
                results_df.loc[f"Block ({block_x}/{block_y})", dim] = exec_time if exec_time is not None else "NP"
            except Exception as e:
                results_df.loc[f"Block ({block_x}/{block_y})", dim] = f"Error: {str(e)}"

    return results_df
