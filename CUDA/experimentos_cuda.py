
import pycuda.driver as cuda
import pycuda.compiler as SourceModule
import numpy as np
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pyopencl as cl
import mult_matrices_basico_cuda as cuda
import matplotlib.ticker as ticker
import mult_matrices_basica_opencl as opencl
import math


def guardar_dataframe_excel(resultados,base_save_dir,funcion_nombre):
   
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
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


#FUNCION PARA ALMACENAR DATA FRAMES EN FORMATO EXCEL , GUARDA EL DATA FRAME RESULTADOS Y EL DATA FRAME MEJORES RESULTADOS

def guardar_dataframes_excel(resultados, best_results_df, base_save_dir,funcion_nombre):
   
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
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados Combinados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

        # Formatear 'Mejores Resultados'
        worksheet = writer.sheets['Mejores Resultados']
        for idx, col in enumerate(best_results_df.columns, start=1):
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")



#COMPARAR OPENCL Y CUDA MULTIPLICANDO MATRICES DE DISTINTOS TAMAÑOS

def comparar(path):

    device_type=device_type = cl.device_type.GPU 

    #Global size y local size
    local_size=(8,8)

    #Block y grid size
    block=(8, 8,1)

    dims=[8,16,32,64,128,256,512,1024,2048,4096,8192]

    results=[]

    for dim in dims:
        A = np.random.random(size=(dim, dim)).astype(np.float32)
        B = np.random.random(size=(dim, dim)).astype(np.float32)

        grid=(dim//8, dim//8)
        
        exec_time_cl,C_cl=opencl.mult_mat_basica(dim,local_size,device_type,opencl.MatrixMul_kernel1,"MatrixMul_kernel1",A,B)
        exec_time_cuda,C_cuda=cuda.ejecutar_kernel(dim,A,B,block,grid)

         # Añadir los resultados a la lista
        results.append({
            "Dimensión": dim,
            "Tiempo OpenCL (s)": exec_time_cl,
            "Tiempo CUDA (s)": exec_time_cuda
        })

    # Crear un DataFrame con los resultados
    df_results = pd.DataFrame(results)

    # Mostrar la tabla de resultados
    guardar_dataframe_excel(df_results,path,'comparacion_cuda_opencl')

    # Graficar los tiempos de ejecución
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Dimensión"], df_results["Tiempo OpenCL (s)"], label="OpenCL", marker='o')
    plt.plot(df_results["Dimensión"], df_results["Tiempo CUDA (s)"], label="CUDA", marker='s')

    # Personalizar el gráfico
    plt.xlabel("Dimensión de la Matriz")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución entre OpenCL y CUDA")

    # Establecer las etiquetas personalizadas para el eje X
    dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    plt.xticks(dims, labels=[str(d) for d in dims], rotation=45)

    # Forzar la escala logarítmica con etiquetas específicas
    plt.gca().set_xscale('log')
    plt.gca().set_xticks(dims)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.gca().get_xaxis().set_minor_formatter(plt.NullFormatter())

    # Agregar leyenda y rejilla
    plt.legend()
    plt.grid(True)

    # Guardar o mostrar el gráfico
    save_path = "grafico_cuda_opencl.png"
    plt.savefig(save_path)
    plt.show()





#Funcion para aplicar kernel_cuda para distintos block y grid sizes

def aplicar_kernel_local_sizes_completo():
    # Combinaciones de bloques fijas y dinámicas (128 hebras por bloque)
    combinaciones_fijas = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    combinaciones_128 = [(x, 128 // x) for x in range(1, 129) if 128 % x == 0]
    todas_combinaciones = combinaciones_fijas + combinaciones_128

    # Índices y columnas del DataFrame
    index = [f"Block ({block[0]}/{block[1]})" for block in todas_combinaciones]
    columns = [2 ** i for i in range(1, 14)]  # Dimensiones de las matrices

    # Crear el DataFrame
    results_df = pd.DataFrame(index=index, columns=columns)

    for block in todas_combinaciones:
        block_x, block_y = block
        block_size = block_x * block_y

        for dim in columns:
            # Verificar que la configuración del bloque sea válida
            if block_size > dim * dim:
                # Si el bloque excede el tamaño de la matriz, omitir esta configuración
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


def graficar_dataframe(df,save_path):
    """
    Genera un gráfico de líneas para cada fila del DataFrame dado.
    
    Args:
    - df (pd.DataFrame): DataFrame con índices representando configuraciones de bloques
                         y columnas representando las dimensiones de las matrices.
                         
    Devuelve:
    - Un gráfico con líneas representando los tiempos de ejecución por configuración de bloque.
    """
    plt.figure(figsize=(12, 8))
    
    # Asegurarse de que los valores sean numéricos, reemplazando errores con NaN
    df = df.apply(pd.to_numeric, errors='coerce')  

    for block in df.index:
        plt.plot(
            df.columns,
            df.loc[block],
            marker='o',
            label=block
        )

    # Configuraciones del gráfico
    plt.title("Tiempos de Ejecución por Configuraciones de Bloque")
    plt.xlabel("Dimensiones de Matrices")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.xscale('log')  # Escala logarítmica para el eje X
    plt.xticks(df.columns, labels=[str(c) for c in df.columns], rotation=45)
    plt.legend(title="Configuraciones de Bloque", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Mostrar el gráfico
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()




def mejores_valores(results_combined):
    # Convertir columnas a numéricas y reemplazar errores con NaN
    results_combined = results_combined.apply(pd.to_numeric, errors='coerce')

    best_results = []

    # Iterar sobre las filas del DataFrame
    for index, row in results_combined.iterrows():
        # Ignorar NaN al calcular el mínimo
        min_value = row.min()
        
        # Encontrar todas las columnas (local sizes) que tienen el valor mínimo
        min_local_sizes = row[row == min_value].index.tolist()

        
        best_results.append({
            'Dimension Matrix': index,
            'Best Value': min_value,
            'Local Size': min_local_sizes  
        })

    # Crear un DataFrame de los mejores resultados
    best_results_df = pd.DataFrame(best_results)
   
    best_results_df['Dimension Matrix'] = best_results_df['Dimension Matrix'].astype(str)

    
    return best_results_df


#FUNCION PARA APLICAR EL EXPERIMENTO PARA UN KERNEL Y UNA FUNCION DADA. LLAMA A LAS FUNCIONES ANTERIORES.
#DEVUELVE DOS DATA FRAMES (TODOS LOS RESULTADOS,MEJORES RESULTADOS) Y TRES GRAFICOS DE LINEAS

def experimento_matrices(base_save_dir,funcion_nombre='kernel_cuda'):
    
    # PARTE 1: APLICAR LOCAL SIZES GENERICAS
    results_general = aplicar_kernel_local_sizes_completo()
   

    # PARTE 4: DEVOLVER LOS MEJORES VALORES PARA CADA FILA
    best_results_df = mejores_valores(results_general.T)

    #Guardar Data Frames
    guardar_dataframes_excel(results_general, best_results_df, base_save_dir,funcion_nombre)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_generales.png')
    graficar_dataframe(results_general, save_path=general_save_path)


    # PARTE 8: Devolver los DataFrames
    return results_general , best_results_df
