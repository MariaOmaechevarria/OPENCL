import pyopencl as cl
import numpy as np
import os
import pandas as pd
import struct
from Mineria_GPU_def import kernel_mining,mining_GPU,validate_nonce
import os
import matplotlib.pyplot as plt
import struct




#FUNCION PARA ALMACENAR DATA FRAMES EN FORMATO EXCEL , GUARDA EL DATA FRAME RESULTADOS Y EL DATA FRAME MEJORES RESULTADOS

# Función para guardar DataFrames en Excel
def guardar_dataframes_excel(resultados, base_save_dir, funcion_nombre):
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados', index=True)
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000000'})
        worksheet = writer.sheets['Resultados']
        for idx, col in enumerate(resultados.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)
    
    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")



# Función para experimentar con distintos global sizes
def experimento_global_sizes(path,target,target_name):
    #target = np.array([0x0000FFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32)  # Dificultad fija
    kernel_name = "kernel_mining"
    device_type = cl.device_type.GPU

    # Configuración del bloque
    block = bytearray(80)
    global_sizes = [(2**7,), (2**8,), (2**9,), (2**10,), (2**12,), (2**15,),(2**16,) ,(2**20,)]
    local_sizes = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,)]

    results_dict = {gs[0]: [] for gs in global_sizes}

    for global_size in global_sizes:
        for local_size in local_sizes:
            exec_time, result_nonce, hash_value = mining_GPU(kernel_mining,kernel_name, block, target, global_size, local_size,device_type)
            results_dict[global_size[0]].append(exec_time)

    df = pd.DataFrame(results_dict, index=[ls[0] for ls in local_sizes])
    df.index.name = 'Local Size'
    df.columns.name = 'Global Size'

    output_dir2 = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir2, exist_ok=True)

    output_dir = os.path.join(output_dir2, target_name)
    os.makedirs(output_dir, exist_ok=True)

    guardar_dataframes_excel(df, output_dir, 'mining_global_sizes')

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

def comparacion_targets(path):
    targets = [
        np.array([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32),  # Mínima dificultad
        np.array([0x7FFFFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x00FFFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x000FFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x0000FFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x00000FFF] + [0xFFFFFFFF] * 7, dtype=np.uint32),
        np.array([0x000000FF] + [0xFFFFFFFF] * 7, dtype=np.uint32)
    ]

    block = bytearray(80)
    global_size = (2**20,)
    kernel_name = "kernel_mining"
    device_type = cl.device_type.GPU
    local_sizes = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,)]

    # Crear un diccionario para almacenar los resultados
    results_dict = {tuple(target): [] for target in targets}

    for target in targets:
        for local_size in local_sizes:
            exec_time, result_nonce, hash_value = mining_GPU(kernel_mining, kernel_name, block, target, global_size, local_size, device_type)
            results_dict[tuple(target)].append(exec_time)

    # Convertir el diccionario a DataFrame
    df = pd.DataFrame(results_dict, index=[ls[0] for ls in local_sizes])
    df.index.name = 'Local Size'

    # Convertir las columnas (targets) a una representación más legible
    target_labels = [f"0x{target[0]:08X}" for target in targets]
    df.columns = target_labels

    # Guardar resultados en Excel
    output_dir = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir, exist_ok=True)
    guardar_dataframes_excel(df, output_dir, 'mining_target')

    # Generar el gráfico
    plt.figure(figsize=(19, 8))
    for i, local_size in enumerate(local_sizes):
        plt.plot(df.columns, df.iloc[i], marker='o', label=f'Local Size {local_size[0]}')

    plt.xlabel('Target (Hexadecimal)')
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
