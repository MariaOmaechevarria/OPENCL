


#Librerias a importar
import pandas as pd
import matplotlib.pyplot as plt
import os


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

cpu_filtered = cpu_df[cpu_df['Unnamed: 0'] == local_size]
gpu_collab_filtered = gpu_collab[gpu_collab['Unnamed: 0'] == local_size]
gpu_portatil_filtered = gpu_portatil[gpu_portatil['Unnamed: 0'] == local_size]
gpu_ucm_filtered = gpu_ucm[gpu_ucm['Unnamed: 0'] == local_size]

# Extraer dimensiones (columnas) y tiempos para graficar
cpu_dimensions = cpu_filtered.columns[1:].astype(int)
cpu_times = cpu_filtered.iloc[0, 1:].values
gpu_collab_dimensions = gpu_collab_filtered.columns[1:].astype(int)
gpu_collab_times = gpu_collab_filtered.iloc[0, 1:].values
gpu_portatil_dimensions = gpu_portatil_filtered.columns[1:].astype(int)
gpu_portatil_times = gpu_portatil_filtered.iloc[0, 1:].values
gpu_ucm_dimensions = gpu_ucm_filtered.columns[1:].astype(int)
gpu_ucm_times = gpu_ucm_filtered.iloc[0, 1:].values

# Filtrar dimensiones desde 4 hasta 2048
cpu_valid_indices = (cpu_dimensions >= 4) & (cpu_dimensions <= 2048)
cpu_dimensions = cpu_dimensions[cpu_valid_indices]
cpu_times = cpu_times[cpu_valid_indices]

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
#ex.guardar_dataframes_excel(comparison_df, comparison_df, save_path,'Comparar_GPUS')

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
