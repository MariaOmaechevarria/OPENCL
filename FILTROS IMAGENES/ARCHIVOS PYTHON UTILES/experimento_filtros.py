import pyopencl as cl
import numpy as np
import pandas as pd
import os
import re
from PIL import Image, ImageFilter
from collections import defaultdict
import matplotlib.pyplot as plt

import funciones_filtros as ff
import kernels_filtros_imagenes as kernel
import determinar_mejor_local_size as mejor
import filtros as f


'''
FUNCIONES PARA GUARDAR DATA FRAMES EN FORMATO EXCEL
'''

# GUARDA DOS DATA FRAMES EN UN ARCHIVO

def guardar_dataframes_excel(resultados, best_results_df, base_save_dir, filtro_nombre, funcion_nombre):

    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    #funcion_dir = os.path.join(filtro_dir, funcion_nombre)
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


#GUARDA UN DATA FRAME EN FORMATO EXCEL
    
def guardar_dataframe_excel(resultados,base_save_dir):
 
    # Crear la estructura de directorios si no existe
    funcion_dir = os.path.join(base_save_dir)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Usar xlsxwriter como motor para formatear las celdas durante la escritura
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        # Guardar los DataFrames en hojas separadas
        resultados.to_excel(writer, sheet_name='Resultados Combinados', index=True)
       
        # Acceder al workbook y crear un formato para números con 6 decimales
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados Combinados']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")


'''
FUNCIONES COMUNES A LOS EXPERIMENTOS
'''

#FUNCION PARA OBTENER EL TAMAÑO DE UNA IMAGEN

def obtener_tamano_imagen(path):
    from PIL import Image
    with Image.open(path) as img:
        return img.size[0] * img.size[1]  # Ancho * Alto

# Función para extraer las dimensiones de las imágenes
def extraer_dimensiones(nombre_archivo):
    # Buscar un patrón como '128x128', '640x480', etc.
    dimensiones = re.findall(r'(\d+)x(\d+)', nombre_archivo)
    if dimensiones:
        # Convertir a entero y devolver como tupla (ancho, alto)
        return tuple(map(int, dimensiones[0]))
    else:
        # Si no se encuentran dimensiones, devolver un valor que permita ordenar correctamente
        return (0, 0)
    
#FUNCION QUE PARA UN DADO LOCAL SIZE CALCULA LOS TIEMPOS DE EJECUCION AL APLICAR UN FILTRO DADO PARA UN CIERTO KERNEL Y UNA CIERTA FUNCION.
#DEVUELVE UN DATA FRAME CON LOS VALORES CORRESPONDIENTES

def filtros_local_size_fijado(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, local_size):
    # Crear un diccionario para almacenar los tiempos de ejecución
    results = {os.path.basename(path): [] for path in lista_paths}

    if len(filtro) == 2:
        filtroX, filtroY = filtro[0], filtro[1]

        for path in lista_paths:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    (filtroX, filtroY),
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                results[os.path.basename(path)].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)}: {e}")
                results[os.path.basename(path)].append(None)  # Manejo de error

    else:
        for path in lista_paths:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    filtro,
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                
                results[os.path.basename(path)].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)}: {e}")
                results[os.path.basename(path)].append(None)  # Manejo de error

    # Crear DataFrame de resultados
    results_general = pd.DataFrame.from_dict(results, orient='index', columns=['Execution Time'])
    return results_general


#DADO UN DATA FRAME DETERMINA LOS MEJORES LOCAL SIZES PARA CADA IMAGEN. DEVUELVE UN DATA FRAME
def mejores_valores(results_combined):
    best_results = []

    # Iterar sobre las filas del DataFrame
    for index, row in results_combined.iterrows():
        # Encontrar el valor mínimo, ignorando NaN
        min_value = row.min()
        # Encontrar todas las columnas (local sizes) que tienen el valor mínimo
        min_local_sizes = row[row == min_value].index.tolist()

        # Agregar un único resultado por imagen, concatenando los tamaños locales en una cadena
        best_results.append({
            'Image Name': index,
            'Best Value': min_value,
            'Local Size': min_local_sizes  # Mantener los tamaños locales como lista
        })

    # Crear un DataFrame de los mejores resultados
    best_results_df = pd.DataFrame(best_results)
    
    return best_results_df
    

'''
FUNCIONES PARA REALIZAR EL EXPERIMENTO_MEJOR_LOCAL_SIZE Y EXPERIMENTO_1000VECES : PARA DISTINTOS LOCAL SIZES OBETENER LSO TIEMPOS DE EJECUCION Y 
DETERMINAR LOS MEJORES. SE HACE PARA NUMEROSOS FILTROS Y KERNELS
'''



#FUNCION PARA APLICAR A UNA LISTA DE IMAGENES UN FILTRO DADO CON LA FUNCION DADA PARA EL KERNEL DADO.
#DEVUELVE UN DATA FRAME CON LOS CAL SIZES ,LAS IMAGENES Y LOS TIEMPOS DE EJECUCION

def filtros_generales(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type):
    local_sizes = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    results = {size: [] for size in local_sizes}

    # Procesar cada imagen
    for path in lista_paths:
        for local_size in local_sizes:
            try:
                imagen_resultante, exec_time = aplicar_filtro_func(
                    path,
                    filtro,
                    kernel_code,
                    kernel_name,
                    device_type,
                    local_size
                )
                results[local_size].append(exec_time)
            except Exception as e:
                print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                results[local_size].append(None)  # O puedes agregar un valor especial para indicar error

    # Crear DataFrame
    results_general = pd.DataFrame(results, index=[os.path.basename(path) for path in lista_paths])

    # Agregar la columna con los nombres de las imágenes
    results_general.index.name = 'Image Name'

    return results_general

#DETERMINA LOS MEJORES LOCAL SIZES Y CALCULA PARA ESOS LOS TIEMPOS DE EJCUCION AL APLICAR EL FILTRO Y KERNEL DADO PARA LA LISTA DE IMAGENES
#DEVUELVE UN DATA FRAME CON LOS VALORES
#
def filtros_optimos(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_unit, processing_elements):
    # Inicializar el diccionario para almacenar los resultados
    results = defaultdict(lambda: defaultdict(list))  # Almacena resultados para cada tamaño local

    # Procesar cada imagen
    for path in lista_paths:
        try:
            # Abrir la imagen
            imagen = Image.open(path)
            imagen_np = np.array(imagen).astype(np.uint8)
            tam_x, tam_y = imagen_np.shape[:2]
            global_size = (tam_x, tam_y)

            # Obtener los tamaños locales óptimos
            local_sizes_optimos = mejor.optimal_local_size(global_size, compute_unit, processing_elements)

            for local_size in local_sizes_optimos:
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        path,
                        filtro,
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[local_size][os.path.basename(path)] = exec_time
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
                    results[local_size][os.path.basename(path)] = None  # O puedes agregar un valor especial para indicar error
        except Exception as e:
            print(f"Error al abrir la imagen {os.path.basename(path)}: {e}")
            # Agregar None para todos los tamaños locales óptimos en caso de error al abrir la imagen
            for local_size in mejor.optimal_local_size((1,1), compute_unit, processing_elements):
                results[local_size][os.path.basename(path)] = None

    # Crear DataFrame con los resultados
    results_optimal = pd.DataFrame({size: [results[size].get(os.path.basename(path), None) for path in lista_paths] 
                                         for size in results.keys()}, 
                                     index=[os.path.basename(path) for path in lista_paths])

    # Agregar la columna con los nombres de las imágenes
    results_optimal.index.name = 'Image Name'

    return results_optimal

#FUNCION QUE GRAFICA UN DATA FRAME DONDE EL EJE X SON LAS IAMGENES, EL EJE Y LOS TIEMPOS DE EJECUCION Y CADA LOCAL SIZE ES UNA LINEA DEL GRAFICO
def graficar_tiempos_ejecucion(data, columns_to_plot=None, save_path=None):
    plt.figure(figsize=(12, 8))

    if columns_to_plot:
        data = data[columns_to_plot]

    # Iterar sobre cada columna del DataFrame (cada local size)
    for local_size in data.columns:
        # Obtener los valores de tiempo correspondientes a cada imagen
        row_values = data[local_size].dropna().values  # Eliminamos NaN
        image_names = data.index[data[local_size].notna()]  # Nombres de las imágenes sin NaN
        
        # Graficar solo si hay datos
        if len(row_values) > 0:
            plt.plot(image_names, row_values, marker='o', label=f'Local Size: {local_size}')

    # Configuraciones de la gráfica
    plt.title('Tiempos de Ejecución por Tamaño de Trabajo')
    plt.xlabel('Nombre de la Imagen')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    #plt.yscale('log')  # Usar escala logarítmica si es necesario
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor legibilidad
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


#FUNCION QUE REALIZA EL EXPERIMENTO PARA UN FILTRO DADO,UN KERNEL Y UNA FUNCION. PARA NUMEROSOS LOCAL SIZES CALCULA LOS TIEMPOS DE EJECUCION PARA
# UNA LISTA DE IMAGENES.
#  DEVUELVE DOS DATA FRAMES Y TRES GRAFICOS

def experimento_filtros(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_units, processing_elements, filtro_nombre, funcion_nombre, base_save_dir='graficos'):

    # PARTE 1: APLICAR LOCAL SIZES GENERICAS
    results_general = filtros_generales(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type)

    # PARTE 2: APLICAR LOCAL SIZES OPTIMAS
    results_optimal = filtros_optimos(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_units, processing_elements)

    # PARTE 3: FUSIONAR LOS DOS DATA FRAMES
    results_combined = pd.merge(results_general, results_optimal, on='Image Name', how='outer')

    # Extraer el tamaño de la imagen y calcular el ancho y alto
    results_combined['Width'] = results_combined.index.to_series().str.extract(r'(\d+)x(\d+)').astype(int).apply(lambda x: x[0] * x[1], axis=1)

    # Ordenar por tamaño de imagen
    results_combined = results_combined.sort_values(by='Width')

    # Eliminar la columna temporal 'Width'
    results_combined = results_combined.drop(columns=['Width'])

    # PARTE 4: DEVOLVER LOS MEJORES VALORES PARA CADA FILA
    best_results_df = mejores_valores(results_combined)

    # Crear directorio para guardar los gráficos
    funcion_dir = os.path.join(base_save_dir, filtro_nombre)
    os.makedirs(funcion_dir, exist_ok=True)

    # PARTE 5: HACER Y GUARDAR UN GRAFICO COMBINADO
    combined_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(results_combined, save_path=combined_save_path)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_generales.png')
    graficar_tiempos_ejecucion(results_general, save_path=general_save_path)

    # PARTE 7: GRAFICAR SOLO LOS MEJORES RESULTADOS (excluyendo ciertos tamaños locales)
    excluded_columns = [(1, 1), (2, 2), (4, 4)]
    columns = [col for col in results_combined.columns if col not in excluded_columns]
    optimal_save_path = os.path.join(funcion_dir, 'tiempos_ejecucion_optimos.png')
    graficar_tiempos_ejecucion(results_combined, columns_to_plot=columns, save_path=optimal_save_path)

    # PARTE 8: Devolver los DataFrames
    return results_combined, best_results_df

#EJECUTA LA FUNCION EXPERIMENTO_FILTROS PARA UNA LISTA DE FILTROS,KERNELS Y FUNCIONES A APLICAR. DEVUELVE PARA CADA KERNEL DOS TABLAS Y TRES GRAFICOS

def ejecutar_experimentos(lista_paths, filtros,filtros_nombres ,aplicar_filtro_funcs, kernel_codes, kernel_names, device_type, compute_units, processing_elements, base_save_dir='graficos'):
  
    # Verificar que todas las listas tengan la misma longitud
    assert len(filtros) == len(aplicar_filtro_funcs) == len(kernel_codes) == len(kernel_names), "Las listas de filtros, funciones, kernels y nombres deben tener la misma longitud."

    for i, filtro in enumerate(filtros):
        aplicar_filtro_func = aplicar_filtro_funcs[i]
        kernel_code = kernel_codes[i]
        kernel_name = kernel_names[i]
        filtro_nombre = filtros_nombres[i] # Puedes personalizar el nombre según el filtro

        funcion_nombre = 'funcion_aplicada'  # Puedes personalizar el nombre según la función

        print(f"Ejecutando experimento para {filtro_nombre} con {funcion_nombre}")

        # Ejecutar el experimento de filtros
        resultados, best_results_df = experimento_filtros(
            lista_paths=lista_paths,
            filtro=filtro,
            aplicar_filtro_func=aplicar_filtro_func,
            kernel_code=kernel_code,
            kernel_name=kernel_name,
            device_type=device_type,
            compute_units=compute_units,
            processing_elements=processing_elements,
            filtro_nombre=filtro_nombre,
            funcion_nombre=funcion_nombre,
            base_save_dir=base_save_dir
        )


        # Guardar los DataFrames en Excel
        guardar_dataframes_excel(resultados, best_results_df, base_save_dir, filtro_nombre, funcion_nombre)



'''
FUNCIONES COMPARACION KERNELS USADAS EN Pruebas_kernels_filtros_local_size y en Experimento_distintos_filtros
DETERMINAR MEJOR KERNEL 
'''



# DADO UN DATA FRAME CON DISTINTOS KERNELS REALIZA UN GRAFICO DONDE EL EJE X SON LAS DIMENSIONES DE LAS IMAGENES, EL EJE Y LOS TIEMPOS DE EJECUCION Y CADA
# LINEA DEL GFRAFICO ES UN KERNEL DISTINTO

def graficar_tiempos_ejecucion_kernels(df,save_path=None):
    """
    Función para graficar los tiempos de ejecución de diferentes kernels para diferentes imágenes.
    
    Parámetros:
    df (DataFrame): Un DataFrame con los nombres de las imágenes y los tiempos de ejecución para cada kernel.
    """
    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar cada kernel
    plt.plot(df.index, df["kernel_filter_color"], marker='o', label='Filtro Color')
    plt.plot(df.index, df["kernel_filter_color_local"], marker='o', label='Filtro Color Memoria Local Ineficiente')
    plt.plot(df.index, df["kernel_filter_color_local2"], marker='o', label='Filtro Color Memoria Local Hebra Maestra')
    plt.plot(df.index, df["kernel_filter_color_local3"], marker='o', label='Filtro Color Memoria Local Organizado')
    plt.plot(df.index, df["kernel_filter_color_local4"], marker='o', label='Filtro Color Memoria Local Pixels a la vez ')
    plt.plot(df.index, df["kernel_filter_color_local_rectangular"], marker='o', label='Filtro Color Memoria Local Dividido')
    plt.plot(df.index, df["kernel_filter_color_rectangular"], marker='o', label='Filtro Color Dividido')
    # Personalizar el gráfico
    plt.title("Tiempos de Ejecución por Kernel")
    plt.xlabel("Nombre de la Imagen")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor visualización
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Ajustar el gráfico para evitar solapamientos
        # Guardar o mostrar la gráfica
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()


#FUNCION QUE DADA UNA LISTA DE KERNELS,LISTA DE FILTROS Y LISTA DE FUNCIONES ,CALCULA PARA CADA UNO LOS TIEMPOS DE EJECUCION CON UN LCOAL SIZE FIADO
#DEVUELVE AL FINAL UN DATA FRAME CON LOS VALORES PARA TODOS LOS KERNELS Y UN GRAFICO

def experimento_kernels(lista_paths, lista_filtro, lista_kernels, lista_nombres_kernels, lista_funciones, device_type, local_size, base_save_dir):
    # Inicializar el DataFrame de resultados
    resultados_finales = pd.DataFrame()

    for i in range(len(lista_kernels)):
        kernel_code=lista_kernels[i]
        kernel_name=lista_nombres_kernels[i]
        aplicar_filtro_func=lista_funciones[i]
        filtro=lista_filtro[i]

    
        resultados_kernel = filtros_local_size_fijado(
            lista_paths,
            filtro,
            aplicar_filtro_func,
            kernel_code,
            kernel_name,
            device_type,
            local_size
        )
        
        # Asignar resultados al DataFrame final
        resultados_finales[kernel_name] = resultados_kernel['Execution Time']

    # Asignar nombres de las imágenes como filas
    resultados_finales.index = resultados_kernel.index
    resultados_finales.index.name = 'Image Name'
    
        # Guardar los DataFrames en Excel
    guardar_dataframes_excel(resultados_finales,resultados_finales, base_save_dir, 'kernels','kernels')

    # Crear directorio para guardar gráficos si no existe
    os.makedirs(base_save_dir, exist_ok=True)

   
    save_path = os.path.join(base_save_dir, f"KERNELS_tiempos_ejecucion.png")
    image_names = [path.split('/')[-1] for path in lista_paths]
    graficar_tiempos_ejecucion_kernels(resultados_finales, save_path=save_path)

    return resultados_finales

#FUNCION PARA COMPARAR KERNELS , FIJADO UN LOCAL SIZE, CALCULA LOS TIEMPOS DE EJECUCION PARA UNA LISTA DE IMAGENES Y UNA LISTA DE FILTROS
# DEVUELVE UN DATA FRAME CON LOS VALORES Y UN GRAFICO

def comparar_filtros(kernels_codes, kernels_names, funciones, image_path, local_size, device_type, filtros1, filtros2,save_path):
    # Crear un diccionario para almacenar los resultados
    results = {name: [] for name in kernels_names}

    for i in range(len(kernels_codes)):
        kernel_code = kernels_codes[i]
        kernel_name = kernels_names[i]
        aplicar_filtro_func = funciones[i]

        for j in range(len(filtros1)):
            if kernel_name == "kernel_filter_color_local3":
                filtro = filtros1[j]
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        image_path,
                        filtro,
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[kernel_name].append(exec_time)
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(image_path)}: {e}")
                    results[kernel_name].append(None)  # Añadir None en caso de error
            else:
                filtroX, filtroY = filtros2[j]
                try:
                    imagen_resultante, exec_time = aplicar_filtro_func(
                        image_path,
                        (filtroX, filtroY),
                        kernel_code,
                        kernel_name,
                        device_type,
                        local_size
                    )
                    results[kernel_name].append(exec_time)
                except Exception as e:
                    print(f"Error al procesar {os.path.basename(image_path)}: {e}")
                    results[kernel_name].append(None)  # Añadir None en caso de error

    # Convertir el diccionario en un DataFrame
    df = pd.DataFrame.from_dict(results, orient='index').T
    guardar_dataframe_excel(df,save_path)

    # Renombrar el índice
    df.index = [str(i) for i in range(1, 33)]





    # Crear gráfico de líneas
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        if column=='kernel_filter_color_local3':
           label='Kernel Memoria Local'
           

        elif column=="kernel_filter_color_local_rectangular":
           label='Kernel Memoria Local Separado'

        else:
            label='Kernel Color Separado'
            
        plt.plot(df.index, df[column], marker='o', label=label)

    # Configurar el gráfico
    plt.title('Comparación de Tiempos de Ejecución por Kernel')
    plt.xlabel('Radio del Filtro')
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.legend(title='Kernels')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

# Mostrar gráfico
    plt.savefig(save_path+'comparacion_filtros.png')
    plt.show()

    return df




'''
FUNCIONES PARA REPETIR LOS MISMOS EXPERIMENTOS---> NO SE USA
'''


#FUNCION PARA REPETIR EL MISMO EXPERIMENTO PARA EL MISMO LOCAL SIZE MUCHAS VECES

def repetir_experimento(local_size, kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type):
    valores = []
    
    for i in range(100):
        try:
            # Procesar las imágenes con el tamaño local dado
            results = filtros_local_size_fijado(lista_paths, filter, aplicar_filtro_func, kernel_code, kernel_name, device_type, local_size).T
            valores.append(results)
        except Exception as e:
            # Mostrar un mensaje de error específico si ocurre
            print(f"Error al procesar con local_size {local_size}: {str(e)}")
    
    # Convertir la lista de DataFrames a un solo DataFrame concatenado
    if valores:
        df_final = pd.concat(valores, ignore_index=True)
        # Calcular la media por cada columna
        medias = df_final.mean()
    else:
        medias = pd.Series()  # Si no se procesó ningún resultado, devolver un Series vacío

    return medias




#FUNCION PARA REPETIR EXPERIMENTO PARA UNA LISTA DADA DE LOCAL SIZES. DEVUELVE DOS DATA FRAMES CON LOS VALORES Y VARIOS GRAFICOS.

def repetir_experimento_local_sizes(kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type,base_save_dir):
    local_sizes = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (1, 128), (128, 1), (2, 64), (64, 2), 
                   (32, 4), (4, 32), (8, 16), (16, 8)]

    # Inicializar DataFrame vacío con los nombres de las imágenes como índice
    df_final = pd.DataFrame(index=[os.path.basename(path) for path in lista_paths])

    for local_size in local_sizes:
        # Ejecutar el experimento con el local_size actual
        df_temp = repetir_experimento(local_size, kernel_code, kernel_name, lista_paths, filter, aplicar_filtro_func, device_type)

        # Verificar que df_temp no esté vacío
        if not df_temp.empty:
            # Convertir el local_size a un string para usarlo como nombre de columna
            local_size_str = f"{local_size[0]}x{local_size[1]}"
            # Asegurar que la columna del DataFrame tenga el nombre correcto
            if isinstance(df_temp, pd.Series):
                df_temp = df_temp.to_frame(name=local_size_str)
            else:
                df_temp.columns = [local_size_str]
            # Alinear por índice (nombres de las imágenes) y concatenar con df_final
            df_final = df_final.join(df_temp, how='outer')

    # Aplicar la función a los índices del DataFrame (los nombres de los archivos)
    df_final['Dimensiones'] = df_final.index.map(extraer_dimensiones)

    # Ordenar el DataFrame por las dimensiones (primero por ancho y luego por alto)
    df_final = df_final.sort_values(by=['Dimensiones'], ascending=True)

   # Eliminar la columna 'Dimensiones' si no quieres que aparezca en el DataFrame final
    df_final = df_final.drop(columns=['Dimensiones'])
    #Guardar data frame
    best_results_df=mejores_valores(df_final)
    
    #Guardar Data frames
    guardar_dataframes_excel(df_final, best_results_df, base_save_dir, 'filter', 'fun')

    #Hacer Graficos
    # PARTE 5: HACER Y GUARDAR UN GRAFICO COMBINADO
    combined_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_combined.png')
    graficar_tiempos_ejecucion(df_final, save_path=combined_save_path)

    # PARTE 6: HACER Y GUARDAR UN GRAFICO SOLO CON LOS RESULTADOS GENERALES
    general_save_path = os.path.join(base_save_dir, 'tiempos_ejecucion_generales.png')
    #graficar_tiempos_ejecucion(results_general, save_path=general_save_path)


    return df_final


