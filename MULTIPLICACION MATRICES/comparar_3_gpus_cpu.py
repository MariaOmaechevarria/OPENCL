
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























'''
#Librerias a importar
import pandas as pd
import matplotlib.pyplot as plt
import os
import funciones_experimentos_matrices as ex

#Path para almacenar resultados
save_path='C:/Users/Eevee/OPENCL/MULTIPLICACION MATRICES/RESULTADOS'

#CARGAR LOS DATOS DE LAS TRES GPUS Y LA CPU

# Cargar los datos
cpu_df = pd.read_csv("C:/Users/Eevee/OPENCL/RESULTADOS_GOOGLE_COLLAB/Mult_Mat_Basica_CPU.csv")
gpu_collab = pd.read_csv("C:/Users/Eevee/OPENCL/RESULTADOS_GOOGLE_COLLAB/Mult_Mat_Basica_GPU.csv")
gpu_portatil = pd.read_excel("C:/Users/Eevee/OPENCL/RESULTADOS_PORTATIL/OPENCL/MULTIPLICACION MATRICES/RESULTADOS/MatrixMul_kernel1/resultados.xlsx")
gpu_ucm = pd.read_excel("C:/Users/Eevee/OPENCL/MULTIPLICACION MATRICES/RESULTADOS/MatrixMul_kernel/resultados.xlsx")


# Filtrar por local_size (4/4)
local_size = "(4/4)"


def obtener_valor_dispositivo(df,local_size,dim_0,dim_f):
     
     df_filtered=df[df['Unnamed: 0']== local_size]
     df_dim,df_times=df_filtered.columns[1:].astype(int),df_filtered.iloc[0, 1:].values
     valid_indices=(df_dim>=dim_0)& (df_dim <= dim_f)
     df_dim=df_dim[valid_indices]
     df_times=df_times[valid_indices]

# Extraer dimensiones (columnas) y tiempos para graficar
cpu_dimensions,cpu_times = obtener_valor_dispositivo(cpu_df,local_size,4,2048)
gpu_collab_dimensions,gpu_collab_times = obtener_valor_dispositivo(gpu_collab,local_size,4,2048)
gpu_portatil_dimensions ,gpu_portatil_times= obtener_valor_dispositivo(gpu_portatil,local_size,4,2048)
gpu_ucm_dimensions,gpu_ucm_times = obtener_valor_dispositivo(gpu_ucm,local_size,4,2048)

# Asegurar que todos los arrays tengan la misma longitud
min_length = min(len(cpu_dimensions), len(cpu_times), len(gpu_collab_times), len(gpu_portatil_times), len(gpu_ucm_times))
cpu_dimensions = cpu_dimensions[:min_length]
cpu_times = cpu_times[:min_length]
gpu_collab_times = gpu_collab_times[:min_length]
gpu_portatil_times = gpu_portatil_times[:min_length]
gpu_ucm_times = gpu_ucm_times[:min_length]

# Crear una tabla comparativa
comparison_df = pd.DataFrame({
    'Dimensión de la Matriz': cpu_dimensions,
    'Intel Xeon Processor 2.20GHz  (s)': cpu_times,
    'NVIDIA Tesla T4 (s)': gpu_collab_times,
    'Intel Iris Xe (s)': gpu_portatil_times,
    'NVIDIA GeForce RTX 3090 (s)': gpu_ucm_times
})

#Guardar tabla como excel:
ex.guardar_dataframes_excel(comparison_df, comparison_df, save_path,'Comparar_GPUS')

# Graficar los datos
plt.figure(figsize=(10, 6))
plt.plot(cpu_dimensions, cpu_times, label='Intel Xeon Processor 2.20GHz', marker='o')
plt.plot(cpu_dimensions, gpu_collab_times, label='NVIDIA Tesla T4', marker='o')
plt.plot(cpu_dimensions, gpu_portatil_times, label='Intel Iris Xe', marker='o')
plt.plot(cpu_dimensions, gpu_ucm_times, label='NVIDIA GeForce RTX 3090', marker='o')

# Configurar título y etiquetas
plt.title('Tiempos de Ejecución para Local Size (4/4)')
plt.xlabel('Dimensión de la Matriz')
plt.ylabel('Tiempo de Ejecución (s)')
plt.xscale('log')  # Usar escala logarítmica en el eje x
plt.xticks(cpu_dimensions, labels=cpu_dimensions)  # Asegurar que las etiquetas se muestran correctamente

# Agregar líneas verticales solo en 8, 16, 32, ..., 2048
for dim in cpu_dimensions:
    if dim in [4,8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        plt.axvline(x=dim, color='gray', linestyle='-', linewidth=0.5)

plt.legend()
plt.grid(True, axis='y', linestyle="--", linewidth=0.5)  # Solo líneas horizontales discontinuas

if save_path:
        path = os.path.join(save_path,'grafico_comparacion_gpus')
        os.makedirs(path, exist_ok=True)
        plt.savefig(path)
        print(f"Gráfico guardado en {path}")
else:
        plt.show()
plt.close()


# Filtrar por local_size (4/4)
local_size = "(4/4)"

gpu_collab_filtered = gpu_collab[gpu_collab['Unnamed: 0'] == local_size]
gpu_portatil_filtered = gpu_portatil[gpu_portatil['Unnamed: 0'] == local_size]
gpu_ucm_filtered = gpu_ucm[gpu_ucm['Unnamed: 0'] == local_size]

# Extraer dimensiones (columnas) y tiempos para graficar
gpu_collab_dimensions = gpu_collab_filtered.columns[1:].astype(int)
gpu_collab_times = gpu_collab_filtered.iloc[0, 1:].values

gpu_portatil_dimensions = gpu_portatil_filtered.columns[1:].astype(int)
gpu_portatil_times = gpu_portatil_filtered.iloc[0, 1:].values

gpu_ucm_dimensions = gpu_ucm_filtered.columns[1:].astype(int)
gpu_ucm_times = gpu_ucm_filtered.iloc[0, 1:].values

# Filtrar dimensiones desde 4 hasta 8192
gpu_collab_valid_indices = (gpu_collab_dimensions >= 4) & (gpu_collab_dimensions <= 2048)
gpu_collab_dimensions = gpu_collab_dimensions[gpu_collab_valid_indices]
gpu_collab_times = gpu_collab_times[gpu_collab_valid_indices]

gpu_portatil_valid_indices = (gpu_portatil_dimensions >= 4) & (gpu_portatil_dimensions <= 2048)
gpu_portatil_dimensions = gpu_portatil_dimensions[gpu_portatil_valid_indices]
gpu_portatil_times = gpu_portatil_times[gpu_portatil_valid_indices]

gpu_ucm_valid_indices = (gpu_ucm_dimensions >= 4) & (gpu_ucm_dimensions <= 2048)
gpu_ucm_dimensions = gpu_ucm_dimensions[gpu_ucm_valid_indices]
gpu_ucm_times = gpu_ucm_times[gpu_ucm_valid_indices]

# Asegurar que todos los arrays tengan la misma longitud
min_length = min(len(gpu_collab_times), len(gpu_portatil_times), len(gpu_ucm_times))
gpu_collab_dimensions = gpu_collab_dimensions[:min_length]
gpu_collab_times = gpu_collab_times[:min_length]
gpu_portatil_times = gpu_portatil_times[:min_length]
gpu_ucm_times = gpu_ucm_times[:min_length]

# Crear una tabla comparativa
comparison_df = pd.DataFrame({
    'Dimensión de la Matriz': gpu_collab_dimensions,
    'NVIDIA Tesla T4 (s)': gpu_collab_times,
    'Intel Iris Xe (s)': gpu_portatil_times,
    'NVIDIA GeForce RTX 3090 (s)': gpu_ucm_times
})

# Guardar tabla como Excel

os.makedirs(save_path, exist_ok=True)
comparison_df.to_excel(os.path.join(save_path, 'Comparar_3_GPUS.xlsx'), index=False)

# Graficar los datos
plt.figure(figsize=(10, 6))
plt.plot(gpu_collab_dimensions, gpu_collab_times, label='NVIDIA Tesla T4', marker='o')
plt.plot(gpu_collab_dimensions, gpu_portatil_times, label='Intel Iris Xe', marker='o')
plt.plot(gpu_collab_dimensions, gpu_ucm_times, label='NVIDIA GeForce RTX 3090', marker='o')

# Configurar título y etiquetas
plt.title('Tiempos de Ejecución para Local Size (4/4)')
plt.xlabel('Dimensión de la Matriz')
plt.ylabel('Tiempo de Ejecución (s)')
plt.xscale('log')  # Usar escala logarítmica en el eje x
plt.xticks(gpu_collab_dimensions, labels=gpu_collab_dimensions)

# Agregar líneas verticales en 4, 8, 16, 32, ..., 8192
for dim in gpu_collab_dimensions:
    if dim in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4196, 8192]:
        plt.axvline(x=dim, color='gray', linestyle='-', linewidth=0.5)

plt.legend()
plt.grid(True, axis='y', linestyle="--", linewidth=0.5)  # Solo líneas horizontales discontinuas

# Guardar el gráfico
plot_save_path = os.path.join(save_path, 'grafico_comparacion_3_gpus.png')
plt.savefig(plot_save_path)
print(f"Gráfico guardado en {plot_save_path}")

plt.close()
'''