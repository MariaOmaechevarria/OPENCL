
'''
ARCHIVO PARA COMPARAR LA EJECUCIÓN EN LAS TRES GPUs utilizadas y en la CPU
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos_cpu_gpu(cpu_path:str, gpu_paths:str)->pd.DataFrame:
    """
    Carga los datos de CPU y GPU desde archivos CSV o Excel.

    Args:
        cpu_path (str): Ruta al archivo CSV de la CPU.
        gpu_paths (dict): Diccionario con nombres de GPUs como claves y rutas de archivos como valores.

    Returns:
        dict: Diccionario con DataFrames para CPU y GPUs.
    """
    data = {}
    if cpu_path is not None:
       data['CPU'] = pd.read_csv(cpu_path)
    for gpu_name, path in gpu_paths.items():
        if path.endswith('.csv'):
            data[gpu_name] = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            data[gpu_name] = pd.read_excel(path)
    return data

def filtrar_por_local_size(df:pd.DataFrame, local_size:str, dim_0:int, dim_f:int)->tuple[pd.DataFrame,pd.DataFrame]:
    """
    Filtra los datos por el tamaño local y rango de dimensiones.

    Args:
        df (DataFrame): DataFrame con los resultados.
        local_size (str): Tamaño local a filtrar (e.g., "(4/4)").
        dim_0 (int): Dimensión mínima.
        dim_f (int): Dimensión máxima.

    Returns:
        tuple: Arrays de dimensiones y tiempos filtrados.
    """
    df_filtered = df[df['Unnamed: 0'] == local_size]
    df_dims = df_filtered.columns[1:].astype(int)
    df_times = df_filtered.iloc[0, 1:].values

    valid_indices = (df_dims >= dim_0) & (df_dims <= dim_f)
    return df_dims[valid_indices], df_times[valid_indices]

def crear_tabla_comparativa(dimensions:list[int], tiempos_dict:dict)->pd.DataFrame:
    """
    Crea un DataFrame comparativo con los tiempos de ejecución.

    Args:
        dimensions (array): Dimensiones de las matrices.
        tiempos_dict (dict): Diccionario con nombres de dispositivos y tiempos.

    Returns:
        DataFrame: Tabla comparativa de tiempos.
    """
    data = {'Dimensión de la Matriz': dimensions}
    for dispositivo, tiempos in tiempos_dict.items():
        data[dispositivo] = tiempos
    return pd.DataFrame(data)

def graficar_comparacion(dimensions:list[int], tiempos_dict:dict, title:str, save_path=None)->None:
    """
    Grafica los tiempos de ejecución para comparación.

    Args:
        dimensions (array): Dimensiones de las matrices.
        tiempos_dict (dict): Diccionario con nombres de dispositivos y tiempos.
        title (str): Título del gráfico.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    plt.figure(figsize=(10, 6))
    for dispositivo, tiempos in tiempos_dict.items():
        plt.plot(dimensions, tiempos, label=dispositivo, marker='o')

    plt.title(title)
    plt.xlabel('Dimensión de la Matriz')
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.xscale('log')
    plt.xticks(dimensions, labels=dimensions)

    for dim in dimensions:
        if dim in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            plt.axvline(x=dim, color='gray', linestyle='-', linewidth=0.5)

    plt.legend()
    plt.grid(True, axis='y', linestyle="--", linewidth=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()
    plt.close()

def guardar_excel(dataframe:pd.DataFrame, save_path:str, filename:str):
    """
    Guarda un DataFrame como archivo Excel.

    Args:
        dataframe (DataFrame): Datos a guardar.
        save_path (str): Ruta del directorio donde guardar.
        filename (str): Nombre del archivo Excel.
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{filename}.xlsx")
    dataframe.to_excel(file_path, index=False)
    print(f"Archivo Excel guardado en {file_path}")

# Ejemplo de uso
if __name__ == "__main__":
    cpu_path = "C:/Users/Eevee/OPENCL/RESULTADOS_GOOGLE_COLLAB/Mult_Mat_Basica_CPU.csv"
    gpu_paths = {
        'NVIDIA Tesla T4': "C:/Users/Eevee/OPENCL/RESULTADOS_GOOGLE_COLLAB/Mult_Mat_Basica_GPU.csv",
        'Intel Iris Xe': "C:/Users/Eevee/OPENCL/RESULTADOS_PORTATIL/OPENCL/MULTIPLICACION MATRICES/RESULTADOS/MatrixMul_kernel1/resultados.xlsx",
        'NVIDIA GeForce RTX 3090': "C:/Users/Eevee/OPENCL/MULTIPLICACION MATRICES/RESULTADOS/MatrixMul_kernel/resultados.xlsx"
    }
    save_path = 'C:/Users/Eevee/OPENCL/MULTIPLICACION MATRICES/RESULTADOS'
    
    #Comparar las 3 GPUS y CPU
    data = cargar_datos_cpu_gpu(cpu_path, gpu_paths)
    local_size = "(4/4)"
    dim_0, dim_f = 4, 2048

    tiempos = {}
    for dispositivo, df in data.items():
        dimensions, times = filtrar_por_local_size(df, local_size, dim_0, dim_f)
        tiempos[dispositivo] = times

    comparison_df = crear_tabla_comparativa(dimensions, tiempos)
    guardar_excel(comparison_df, save_path, 'Comparar_GPUS_CPU')

    graficar_comparacion(dimensions, tiempos, 'Tiempos de Ejecución para Local Size (4/4)',
                         os.path.join(save_path, 'grafico_comparacion_gpus_cpu.png'))
    
    #Comparar las tres GPUS

    data2 = cargar_datos_cpu_gpu(None, gpu_paths)
    local_size = "(4/4)"
    dim_0, dim_f = 4, 2048

    tiempos = {}
    for dispositivo, df in data2.items():
        dimensions, times = filtrar_por_local_size(df, local_size, dim_0, dim_f)
        tiempos[dispositivo] = times

    comparison_df = crear_tabla_comparativa(dimensions, tiempos)
    guardar_excel(comparison_df, save_path, 'Comparar_GPUS')

    graficar_comparacion(dimensions, tiempos, 'Tiempos de Ejecución para Local Size (4/4)',
                         os.path.join(save_path, 'grafico_comparacion_gpus.png'))

















