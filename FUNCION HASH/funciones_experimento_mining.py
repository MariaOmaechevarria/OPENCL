'''
ARCHIVO CON FUNCIONES PARA REALIZAR EXPERIMENTOS DE MINERÍA DE BLOQUES DEL BLOCKCHAIN
'''

import pyopencl as cl
import numpy as np
import os
import pandas as pd


from Mineria_GPU_def import  mining_GPU
import kernel_mining as kernel
import matplotlib.pyplot as plt


# FUNCION PARA ALMACENAR DATA FRAMES EN FORMATO EXCEL, GUARDA EL DATA FRAME RESULTADOS Y EL DATA FRAME MEJORES RESULTADOS

def guardar_dataframes_excel(resultados: pd.DataFrame, base_save_dir: str, funcion_nombre: str) -> None:
    """
    Guarda los resultados en un archivo Excel en la ruta especificada.

    Inputs:
    - resultados (pd.DataFrame): DataFrame con los resultados del experimento.
    - base_save_dir (str): Directorio base donde se almacenarán los resultados.
    - funcion_nombre (str): Nombre de la función para organizar los resultados.

    Outputs:
    - None: Guarda un archivo Excel formateado en el directorio especificado.
    """
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados', index=True)
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000000'})  # Formato numérico para alta precisión
        worksheet = writer.sheets['Resultados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)
    
    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")

#FUNCION PARA EXPERIMENTAR CON DISTINTOS GLOBAL SIZES PARA MINAR UN BLOQUE CON UN TARGET DADO

def experimento_global_sizes(
    path: str, 
    target: np.ndarray, 
    target_name: str
) -> None:
    """
    Experimenta con diferentes tamaños de global y local sizes en OpenCL.

    Inputs:
    - path (str): Ruta base donde se guardarán los resultados.
    - target (np.ndarray): Array objetivo (target) que define la dificultad.
    - target_name (str): Nombre identificador del objetivo.

    Outputs:
    - None: Genera gráficos y guarda resultados en Excel.
    """
    kernel_name = "kernel_mining"
    device_type = cl.device_type.GPU

    # Configuración inicial
    block = bytearray(80)
    global_sizes = [(2**7,), (2**8,), (2**9,), (2**10,), (2**12,), (2**15,), (2**16,), (2**20,)]
    local_sizes = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,)]

    # Diccionario para almacenar resultados
    results_dict = {gs[0]: [] for gs in global_sizes}

    # Realizar experimentos
    for global_size in global_sizes:
        for local_size in local_sizes:
            exec_time, result_nonce, hash_value = mining_GPU(kernel.kernel_mining, kernel_name, block, target, global_size, local_size, device_type)
            results_dict[global_size[0]].append(exec_time)

    # Convertir resultados a DataFrame
    df = pd.DataFrame(results_dict, index=[ls[0] for ls in local_sizes])
    df.index.name = 'Local Size'
    df.columns.name = 'Global Size'

    # Guardar resultados en Excel
    output_dir2 = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir2, exist_ok=True)

    output_dir = os.path.join(output_dir2, target_name)
    os.makedirs(output_dir, exist_ok=True)

    guardar_dataframes_excel(df, output_dir, 'mining_global_sizes')

    # Graficar resultados
    plt.figure(figsize=(12, 8))
    for global_size in global_sizes:
        plt.plot([ls[0] for ls in local_sizes], df[global_size[0]], marker='o', label=f'Global Size {global_size[0]}')
    plt.xlabel('Local Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Local Size for Different Global Sizes')
    plt.legend(title='Global Sizes')
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(ticks=[1, 2, 4, 8, 16, 32, 64, 128], labels=[1, 2, 4, 8, 16, 32, 64, 128])
    plt.tight_layout()

    plt_path = os.path.join(output_dir, "execution_time_line_plot.png")
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")
    plt.show()

#FUNCION QUE CONVUERTE UN ARRAY EN NOTACION CIENTIFICA , USADO PARA LOS TARGETS

def array_hex_to_scientific_notation(hex_array: np.ndarray) -> str:
    """
    Convierte un array de números hexadecimales en un número entero 
    y lo devuelve en notación científica.
    
    Args:
        hex_array (np.ndarray): Array de números hexadecimales (dtype=np.uint32).
    
    Returns:
        str: Número en notación científica.
    """
    if hex_array.dtype != np.uint32:
        raise ValueError("El array debe ser de tipo np.uint32")
    
    # Convertir los valores del array a un único entero
    result = 0
    for i, value in enumerate(reversed(hex_array)):
        result += int(value) << (32 * i)
    
    # Convertir a notación científica
    scientific_notation = f"{result:.5e}"
    return scientific_notation


#FUNCION QUE COMPARA DISTINTOS TARGETS EN LA MINERÍA DE UN BLOQUE DEL BLOCKCHAIN CON UN GLOBAL SIZE FIJADO


def comparacion_targets(path: str) -> None:
    """
    Compara diferentes objetivos (targets) y mide los tiempos de ejecución.

    Inputs:
    - path (str): Ruta base donde se guardarán los resultados.

    Outputs:
    - None: Genera gráficos y guarda resultados en Excel.
    """
    # Lista de objetivos con diferentes niveles de dificultad
    targets = [
        np.array([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32),  # Mínima dificultad
        np.array([0x7FFFFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x00FFFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x000FFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x0000FFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x00000FFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x000000FF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x0000000F] + [0x00FFFFFF] * 7, dtype=np.uint32)]

    block = bytearray(80)
    global_size = (2**20,)
    kernel_name = "kernel_mining"
    device_type = cl.device_type.GPU
    local_sizes = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,)]

    # Diccionario para almacenar resultados
    results_dict = {tuple(target): [] for target in targets}

    for target in targets:
        for local_size in local_sizes:
            exec_time, result_nonce, hash_value = mining_GPU(kernel.kernel_mining, kernel_name, block, target, global_size, local_size, device_type)
            results_dict[tuple(target)].append(exec_time)

    # Convertir el diccionario a DataFrame
    df = pd.DataFrame(results_dict, index=[ls[0] for ls in local_sizes])
    df.index.name = 'Local Size'

    # Convertir las columnas (targets) a una representación más legible en notación científica
    target_labels = [ str(array_hex_to_scientific_notation(target)) for target in targets]
    df.columns = target_labels

    # Guardar resultados en Excel
    output_dir = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir, exist_ok=True)
    guardar_dataframes_excel(df, output_dir, 'mining_target')

    # Generar el gráfico
    plt.figure(figsize=(19, 8))
    for i, local_size in enumerate(local_sizes):
        plt.plot(df.columns, df.iloc[i], marker='o', label=f'Local Size {local_size[0]}')

    plt.xlabel('Target (Scientific Notation)')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Targets for Different Local Sizes')
    plt.legend(title='Local Sizes')
    plt.grid(True)

    # Ajustar escala del eje X
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Guardar el gráfico
    plt_path = os.path.join(output_dir, "execution_time_target_plot.png")
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")
    plt.show()
