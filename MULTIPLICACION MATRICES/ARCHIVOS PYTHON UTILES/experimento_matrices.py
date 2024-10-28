import funciones_matrices as fm
import kernels_matrices as km
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import determinar_mejor_local_size as mejor
from collections import defaultdict

'''
FUNCIONES PARA APLICAR EXPERIMENTO_LOCAL_SIZES: PARA DISTINTOS KERNELS OBTENER EL MEJOR LOCAL SIZE. DEVOLVER TABLAS Y GRAFICOS
'''


# DADA UNA GPU, DETERMINA LOS LOCAL SIZES OPTIMOS Y CALCULA EL CORRESPONDIENTE DATA FRAME PARA ESOS LOCAL SIZES APLICANDO funcion_aplicar PARA EL CORRESPONDIENTE KERNEL

def local_sizes_optimos(funcion_aplicar, kernel_code, kernel_name, device_type, compute_unit, processing_elements):

    local_sizes_optimos = mejor.optimal_local_size((128,128), compute_unit, processing_elements)
    columns = [2 ** i for i in range(1, 14)]  # 2^1 a 2^13 (de 2 a 8192)
    index = [f"({i}/{j})"  for i,j in local_sizes_optimos]

    results_df = pd.DataFrame(index=index, columns=columns)
    for local_size in local_sizes_optimos:
      dim=2
      while dim<=8192:

       A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
       B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)

       try:

         exec_time,C=funcion_aplicar(dim,local_size,device_type,kernel_code,kernel_name,A,B)

         results_df.loc[f"({local_size[0]}/{local_size[1]})", dim] = exec_time 

       except Exception as e:

         print(f'Error al procesar con local size {local_size}: {e}')

         results_df.loc[f"({local_size[0]}/{local_size[1]})", dim] = ""

       dim*=2

       del A,B

    
    return results_df


# CALCULA LOS TIEMPOS APLICANDO funcion-aplicar PARA EL KERNEL CODE CON LOCAL SIZES CUADRADOS ( (1,1),(2,2),(4,4),(8,8),(16,16),(32,32))
#DEVUELVE UN DATA FRAME CON TODOS LOS TIEMPOS

def aplicar_kernel_local_sizes(kernel_code,kernel_name,device_type,funcion_aplicar):

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

       A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
       B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)

       exec_time,C=funcion_aplicar(dim,local_size,device_type,kernel_code,kernel_name,A,B)

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



#FUNCION QUE DADO UN DATA FRAME DEVUELVE LOS MEJORES LOCAL SIZES PARA CADA DIMENSION.
#DEVUELVE UN DATA FRAME CON ESOS VALORES

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

def experimento_matrices(funcion_aplicar, kernel_code, kernel_name, device_type, compute_units, processing_elements, funcion_nombre, base_save_dir='graficos'):
    
    # PARTE 1: APLICAR LOCAL SIZES GENERICAS
    results_general = aplicar_kernel_local_sizes(kernel_code,kernel_name,device_type,funcion_aplicar)

    # PARTE 2: APLICAR LOCAL SIZES OPTIMAS
    results_optimal = local_sizes_optimos(funcion_aplicar, kernel_code, kernel_name, device_type, compute_units, processing_elements)

    # PARTE 3: FUSIONAR LOS DOS DATA FRAMES
    # Fusión de los DataFrames
    df_combined = pd.concat([results_general, results_optimal], axis=0)
   

    # PARTE 4: DEVOLVER LOS MEJORES VALORES PARA CADA FILA
    best_results_df = mejores_valores(df_combined.T)

    #Guardar Data Frames
    guardar_dataframes_excel(df_combined, best_results_df, base_save_dir,funcion_nombre)

    # Crear directorio para guardar los gráficos
    funcion_dir = os.path.join(base_save_dir, kernel_name)

    os.makedirs(funcion_dir, exist_ok=True)

    # PARTE 5: HACER Y GUARDAR UN GRAFICO COMBINADO
    combined_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(df_combined.T, save_path=combined_save_path)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_generales.png')
    graficar_tiempos_ejecucion(results_general.T, save_path=general_save_path)

    # PARTE 7: GRAFICAR SOLO LOS MEJORES RESULTADOS (excluyendo ciertos tamaños locales)
    excluded_columns = ['(1/1)', '(2/2)', '(4/4)']
    columns = [col for col in (df_combined.T).columns if col not in excluded_columns]
    optimal_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_optimos.png')
    graficar_tiempos_ejecucion(df_combined.T , columns_to_plot=columns, save_path=optimal_save_path)

    # PARTE 8: Devolver los DataFrames
    return df_combined , best_results_df



#FUNCION PARA APLICAR experimento_matrices PARA DISTINTOS KERNELS Y FUNCIONES. DEVUELVE LO MISMO QUE LA FUNCION PERO PARA CADA KERNEL. 
# SE USA EN Experimento_local_sizes.ipynb

def ejecutar_experimentos(aplicar_funcs, kernel_codes, kernel_names, device_type, compute_units, processing_elements, base_save_dir):

    # Verificar que todas las listas tengan la misma longitud
    assert len(aplicar_funcs) == len(kernel_codes) == len(kernel_names), "Las listas de filtros, funciones, kernels y nombres deben tener la misma longitud."

    for i, filtro in enumerate(kernel_codes):
        funcion_aplicar = aplicar_funcs[i]
        kernel_code = kernel_codes[i]
        kernel_name = kernel_names[i]

        funcion_nombre = kernel_name  
        print(f"Ejecutando experimento  con {funcion_nombre}")

        # Ejecutar el experimento de filtros
        resultados, best_results_df = experimento_matrices(funcion_aplicar, kernel_code, kernel_name, device_type, compute_units, processing_elements, funcion_nombre, base_save_dir)



'''
FUNCIONES PARA REALIZAR EXPERIMENTO_COMPARACION_KERNELS: COMPARAR DISTINTSO KERNELS PARA VER CUAL ES MEJOR. DEVUELVE TABLS Y DATA FRAMES
'''
        

#FUNCION QUE PARA UNA LISTA DE KERNELS Y SUS FUNCIONES Y UN LCOAL SIZE FIJADO CALCULA SUS TIEMPOS PARA DISTINTAS DIMENSIONES DE MATRICES
#DEVUELVE UN DATA  FRAME Y UN GRAFICO
#SE USA EN experimento-comparacion_kernel_matrices.ipynb

def experimento_kernels(lista_kernels, lista_nombres_kernels, lista_funciones, device_type, local_size, base_save_dir):
    # Inicializar el DataFrame de resultados final
    resultados_finales = pd.DataFrame()

    # Procesar cada kernel
    for kernel_code, kernel_name, aplicar_filtro_func in zip(lista_kernels, lista_nombres_kernels, lista_funciones):
        # Obtener resultados para el kernel actual
        resultados_kernel = aplicar_kernel_local_fijado(kernel_code, kernel_name, device_type, aplicar_filtro_func, local_size)
        
        # Asignar resultados al DataFrame final
        resultados_finales[kernel_name] = resultados_kernel.iloc[0]  # Los tiempos de ejecución ya están en las columnas por dimensiones

    # Asignar las dimensiones de las matrices como filas
    resultados_finales.index = resultados_kernel.columns
    resultados_finales.index.name = 'Dim Matrix'

    # Guardar los DataFrames en Excel
    guardar_dataframes_excel(resultados_finales,resultados_finales, base_save_dir, f'{kernel_name}_resultados.xlsx')

    # Crear directorio para guardar gráficos si no existe
    os.makedirs(base_save_dir, exist_ok=True)

    # Graficar los resultados
    save_path = os.path.join(base_save_dir, f"KERNELS_tiempos_ejecucion.png")
    graficar_tiempos_ejecucion_kernels(resultados_finales, save_path=save_path)

    return resultados_finales


#FUNCION PARA GRAFICAR DISTINTOS KERNELS. EJE X DIMENSIONES MATRICES. EJE Y TIEMPOS. CADA KERNEL ES UNA LINEA

def graficar_tiempos_ejecucion_kernels(df, save_path=None):
    """
    Función para graficar los tiempos de ejecución de diferentes kernels para diferentes dimensiones.
    
    Parámetros:
    df (DataFrame): Un DataFrame con las dimensiones de las matrices y los tiempos de ejecución para cada kernel.
    save_path (str, optional): Ruta para guardar el gráfico. Si es None, se mostrará el gráfico.
    """
    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar cada kernel
    plt.plot(df.index, df["MatrixMul_kernel1"], marker='o', label='Mult Mat Básico')
    plt.plot(df.index, df["MatrixMul_kernel_localA_coallesced"], marker='o', label='Mult Mat Memoria Local')
    plt.plot(df.index, df["MatrixMul_Local_Memory"], marker='o', label='Mult Mat Memoria Local Tiles')

    # Personalizar el gráfico
    plt.title("Tiempos de Ejecución por Kernel")
    plt.xlabel("Dimensiones de la Matriz")
    plt.ylabel("Tiempo de Ejecución (segundos)")

    # Configurar las etiquetas del eje X
    plt.xscale('log')
    dimensiones = [ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    plt.xticks(dimensiones, labels=[str(d) for d in dimensiones], rotation=45)  # Establecer las dimensiones específicas

    plt.grid(True)  # Agregar una cuadrícula
    plt.legend()
    plt.tight_layout()  # Ajustar el gráfico para evitar solapamientos

    # Guardar o mostrar la gráfica
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()

    #FUNCION QUE DADO UN LOCAL SIZE FIJADO CALCULA PARA TODAS LAS DIMENSIONES LOS VALORES APLICANDO LA FUNCION funcion_aplicar PARA EL KERNEL_CODE CORRESPONDIENTE
#DEVUELVE UN DATA FRAME CON TODOS LOS VALORES

def aplicar_kernel_local_fijado(kernel_code, kernel_name, device_type, funcion_aplicar, local_size):
    # Columnas representan las dimensiones de la matriz
    columns = [2 ** i for i in range(3, 14)]  # Dimensiones desde 8 hasta 8192
    results_df = pd.DataFrame(columns=columns)

    dim = 8  
    while dim <= 8192:
        # Crear matrices A y B
        A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
        B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)

        # Aplicar el kernel y medir el tiempo de ejecución
        exec_time, C = funcion_aplicar(dim, local_size, device_type, kernel_code, kernel_name, A, B)

        # Almacenar el tiempo de ejecución en el DataFrame
        results_df[dim] = [exec_time] if exec_time is not None else ["NP"]

        dim *= 2  

        del A, B  # Eliminar matrices para liberar memoria

    results_df.index = [local_size]  # El índice es el tamaño local

    return results_df






'''
FUNCION UTIL PARA GUARDAR DATA FRAMES
'''


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
