import pyopencl as cl
import numpy as np
import os
import pandas as pd
import struct
from Mining_GPU import kernel_mining,mining_GPU
import os
import matplotlib.pyplot as plt




#FUNCION PARA ALMACENAR DATA FRAMES EN FORMATO EXCEL , GUARDA EL DATA FRAME RESULTADOS Y EL DATA FRAME MEJORES RESULTADOS

def guardar_dataframes_excel(resultados,  base_save_dir,funcion_nombre):
   
    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Usar xlsxwriter como motor para formatear las celdas durante la escritura
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        # Guardar los DataFrames en hojas separadas
        resultados.to_excel(writer, sheet_name='Resultados', index=True)
       

        # Acceder al workbook y crear un formato para números con 6 decimales
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional


    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")

def experimento_global_sizes(path):
    # Target para probar

    target = np.uint64(0x000000000000000F)

    # Nonmbre del kernel
    kernel_name = "kernel_mining"

    #Device type
    device_type = cl.device_type.GPU

    # Simulación de campos del encabezado
    version = 2  # Versión del bloque (4 bytes)
    prev_block_hash = "0000000000000000000babae9ed8f7bb2b8d3f9f97bba97b8b8b8b8b8b8b8b8b"  # Hash del bloque anterior (simulado, 32 bytes en hex)
    merkle_root = "4d5f5c9ac7ed8f96b0e8a6b3b1c1a3e5f7d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"  # Merkle root simulado (32 bytes en hex)
    timestamp = 1633046400  # Timestamp Unix (ejemplo)
    bits = 0x17148edf  # Dificultad en bits (ejemplo de Bitcoin)
    nonce = 2083236893  # Nonce simulado

    # Construcción del encabezado en binario
    header = (
        str(version).encode('utf-8') +  # Convertir version a bytes
        bytes.fromhex(prev_block_hash) +  # Convertir prev_block_hash a bytes desde hexadecimal
        bytes.fromhex(merkle_root) +  # Convertir merkle_root a bytes desde hexadecimal
        timestamp.to_bytes(4, 'little') +  # Convertir timestamp a 4 bytes (little-endian)
        bits.to_bytes(4, 'little') +  # Convertir bits a 4 bytes (little-endian)
        nonce.to_bytes(4, 'little')  # Convertir nonce a 4 bytes (little-endian)
    )

    #Distintas global y local sizes
    global_sizes=[(2**7,),(2**8,),(2**9,),(2**10,),(2**12,),(2**15,),(2**20,)]
    local_sizes=[(1,),(2,),(4,),(8,),(16,),(32,),(64,),(128,)]


    # Crear un diccionario para almacenar los resultados de manera organizada
    results_dict = {gs: [] for gs in global_sizes}

    for global_size in global_sizes:
        for local_size in local_sizes:
            exec_time, result_nonce = mining_GPU(kernel_name, kernel_mining, device_type, header, target, global_size, local_size)
            results_dict[global_size].append(exec_time)

    # Convertir el diccionario a DataFrame
    df = pd.DataFrame(results_dict, index=local_sizes)
    df.index.name = 'Local Size'
    df.columns.name = 'Global Size'

    # Crear el directorio "RESULTADOS" dentro de la carpeta "HASH" si no existe
    output_dir = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir, exist_ok=True)

    # Guardar el DataFrame en un archivo CSV en la carpeta "RESULTADOS"
    guardar_dataframes_excel(df,  output_dir,'mining_global_sizes')
    

    # Generar el gráfico de líneas de tiempos de ejecución
    plt.figure(figsize=(12, 8))
    for global_size in global_sizes:
        plt.plot(local_sizes, df[global_size], marker='o', label=f'Global Size {global_size[0]}')

    # Personalizar el gráfico
    plt.xlabel('Local Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Local Size for Different Global Sizes')
    plt.legend(title='Global Sizes')
    plt.grid(True)

    # Configurar etiquetas específicas del eje X
    plt.xscale('log')
    plt.xticks(ticks=[1, 2, 4, 8, 16, 32, 64, 128], labels=[1, 2, 4, 8, 16, 32, 64, 128])

    # Guardar el gráfico en la carpeta "RESULTADOS"
    plt_path = os.path.join(output_dir, "execution_time_line_plot.png")
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")
    plt.show()

def comparacion_targets(path):
    targets = [
        np.uint64(0x00FFFFFFFFFFFFFFFF),
        np.uint64(0x000FFFFFFFFFFFFFFF),
        np.uint64(0x0000FFFFFFFFFFFFFF),
        np.uint64(0x00000FFFFFFFFFFFFF),
        np.uint64(0x000000FFFFFFFFFFFF),
        np.uint64(0x0000000FFFFFFFFFFF),
        np.uint64(0x00000000FFFFFFFFFF),
        np.uint64(0x000000000FFFFFFFFF),
        np.uint64(0x0000000000FFFFFFFF),
        np.uint64(0x00000000000FFFFFFF),
        np.uint64(0x0000000000000FFFFF),
        np.uint64(0x00000000000000FFFF),
        np.uint64(0x000000000000000FFF),
        np.uint64(0x0000000000000000FF),
        np.uint64(0x00000000000000000F)
    ]
    print(targets)
    # Simulación de campos del encabezado
    version = 2  # Versión del bloque (4 bytes)
    prev_block_hash = "0000000000000000000babae9ed8f7bb2b8d3f9f97bba97b8b8b8b8b8b8b8b8b"
    merkle_root = "4d5f5c9ac7ed8f96b0e8a6b3b1c1a3e5f7d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
    timestamp = 1633046400
    bits = 0x17148edf
    nonce = 2083236893

    # Construcción del encabezado en binario
    header = (
        str(version).encode('utf-8') +
        bytes.fromhex(prev_block_hash) +
        bytes.fromhex(merkle_root) +
        timestamp.to_bytes(4, 'little') +
        bits.to_bytes(4, 'little') +
        nonce.to_bytes(4, 'little')
    )
   
    # Fijar un global size
    global_size = (2**10,)

    # Nombre del kernel
    kernel_name = "kernel_mining"

    # Device type
    device_type = cl.device_type.GPU

    # Probar distintos local sizes
    local_sizes = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,)]

    # Crear un diccionario para almacenar los resultados de manera organizada
    results_dict = {target: [] for target in targets}

    for target in targets:
        for local_size in local_sizes:
            exec_time, result_nonce = mining_GPU(kernel_name, kernel_mining, device_type, header, target, global_size, local_size)
            results_dict[(target)].append(exec_time)

    # Convertir el diccionario a DataFrame
    df = pd.DataFrame(results_dict, index=[local_size[0] for local_size in local_sizes])
    df.index.name = 'Local Size'
    df.columns.name = 'Target'

    # Crear el directorio "RESULTADOS" dentro de la carpeta "HASH" si no existe
    output_dir = os.path.join(path, "FUNCION HASH/RESULTADOS")
    os.makedirs(output_dir, exist_ok=True)
    guardar_dataframes_excel(df,  output_dir,'mining_target')
  

    # Generar el gráfico de líneas para tiempos de ejecución
    plt.figure(figsize=(19, 8))

    # Iterar sobre todos los local_sizes para graficar una línea por cada uno
    for i, local_size in enumerate(local_sizes):
        plt.plot(df.columns, df.iloc[i], marker='o', label=f'Local Size {local_size[0]}')

    # Personalizar el gráfico
    plt.xlabel('Target ')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Targets for Different Local Sizes')
    
    # Añadir leyenda
    plt.legend(title='Local Sizes')
    plt.grid(True)

    values=[18446744073709551615, 1152921504606846975, 72057594037927935, 4503599627370495, 281474976710655, 17592186044415, 1099511627775, 68719476735, 4294967295, 268435455, 1048575, 65535, 4095, 255, 15]
    # Configurar etiquetas específicas del eje X
    plt.xscale('log')
    plt.xticks(ticks=values, labels=[hex(value) for value in values],rotation=45)

    # Mostrar el gráfico
    plt.tight_layout()
    # Guardar el gráfico en la carpeta "RESULTADOS"
    plt_path = os.path.join(output_dir, "execution_time_target_plot.png")
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")
    plt.show()

# Llamar a la función con la ruta correcta
path = "C:/Users/Eevee/OPENCL-3"
comparacion_targets(path)
experimento_global_sizes(path)
