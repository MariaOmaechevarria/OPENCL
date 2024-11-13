
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

    #Hacer un grafico
        # Graficar los tiempos de ejecución
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Dimensión"], df_results["Tiempo OpenCL (s)"], label="OpenCL", marker='o')
    plt.plot(df_results["Dimensión"], df_results["Tiempo CUDA (s)"], label="CUDA", marker='s')

    # Personalizar el gráfico
    plt.xlabel("Dimensión de la Matriz")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución entre OpenCL y CUDA")
    # Establecer la escala logarítmica para el eje X
   
    plt.xscale('log')
    

    # Asegurarnos de que las etiquetas del eje X correspondan a las dimensiones
    plt.xticks(dims)  # Mostrar solo los valores de dims en el eje X
    
    # Mejorar la visualización de las etiquetas, por si acaso
    plt.xticks(rotation=45)
    

    plt.legend()
    plt.grid(True) # Mostrar solo los valores de dims en el eje X
 

    #Guardar grafico

    save_path =os.path.join(path, 'opencl_cuda.png')
    plt.savefig(save_path)



#Funcion para aplicar kernel_cuda para distintos block y grid sizes

def aplicar_kernel_local_sizes():

  index = [(f"({2 ** i}/{2 ** i})" if i != 0 else "(1/1)") for i in range(0, 5)]
  columns = [2 ** i for i in range(1, 14)]  # 2^1 a 2^13 (de 2 a 8192)
  results_df = pd.DataFrame(index=index, columns=columns)
  i=1
  while i<=32:

    local_size=(i,i)
    if i==1:
        dim=2

    else:
      dim=i

    while dim<=8192:

       A = np.random.random(size=(dim, dim)).astype(np.float32)
       B = np.random.random(size=(dim, dim)).astype(np.float32)
       
       grid = (dim // i, dim // i)
       block = (i, i,1)

       exec_time,C=cuda.ejecutar_kernel(dim,A,B,block,grid)

       results_df.loc[f"({i}/{i})", dim] = exec_time if exec_time is not None else "NP"


       dim*=2

       del A,B


    i*=2
    
  

  return results_df


# DADO UN DATA FRAME REALIZA SU GRAFICO DONDE EN EL EJE X ESTAN LAS DIMENSIONES DE LAS MATRICES, EJE Y LOS TIEMPOS Y CADA LCOAL SIZE ES UNA LINEA
def graficar_tiempos_ejecucion(data, columns_to_plot=None, save_path=None):
    # Convertir a numérico y reemplazar errores con NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Eliminar columnas completamente vacías
    data = data.dropna(axis=1, how='all')

    plt.figure(figsize=(12, 8))

    # Si se especifican columnas a graficar, usarlas, de lo contrario, graficar todo
    if columns_to_plot is not None:
        data = data[columns_to_plot]
    else:
        data = data.dropna(axis=0, how='all')  # Eliminar filas completamente vacías

    # Iterar sobre cada columna del DataFrame (cada local size)
    for local_size in data.columns:
        # Obtener los valores de tiempo correspondientes a cada imagen
        row_values = data[local_size].dropna().values  # Eliminamos NaN
        dim_matrix = data.index[data[local_size].notna()]  
        
        # Graficar solo si hay datos
        if len(row_values) > 0:
            plt.plot(dim_matrix, row_values, marker='o', label=f'Tamaño Local: {local_size}')

    # Configuraciones de la gráfica
    plt.title('Tiempos de Ejecución por Tamaño de Trabajo')
    plt.xlabel('Dimensiones Matrices')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    ticks = [2 ** i for i in range(1, 14)]  # 2, 4, 8, ..., 8192
    plt.xticks(ticks, labels=[str(t) for t in ticks], rotation=45)

    # Usar escala logarítmica en el eje x
    plt.xscale('log')

    # Ajustar el formato del eje x para que muestre solo los ticks especificados
    plt.gca().set_xticks(ticks)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.legend(title='Tamaños de Trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Ajustar el diseño

    # Guardar o mostrar la gráfica
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
    results_general = aplicar_kernel_local_sizes()
   

    # PARTE 4: DEVOLVER LOS MEJORES VALORES PARA CADA FILA
    best_results_df = mejores_valores(results_general.T)

    #Guardar Data Frames
    guardar_dataframes_excel(results_general, best_results_df, base_save_dir,funcion_nombre)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_generales.png')
    graficar_tiempos_ejecucion(results_general.T, save_path=general_save_path)


    # PARTE 8: Devolver los DataFrames
    return results_general , best_results_df

