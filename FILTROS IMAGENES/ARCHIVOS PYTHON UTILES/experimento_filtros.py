import pyopencl as cl
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from collections import defaultdict
import matplotlib.pyplot as plt

import funciones_filtros as ff
import kernels_filtros_imagenes as kernel
import determinar_mejor_local_size as mejor
import filtros as f

import pandas as pd
import os

# Supongamos que 'resultados' y 'best_results_df' son tus DataFrames

def guardar_dataframes_excel(resultados, best_results_df, base_save_dir, filtro_nombre, funcion_nombre):
    """
    Guarda los DataFrames en un archivo Excel con múltiples hojas y formatea las celdas para mostrar 6 decimales.
    
    :param resultados: DataFrame con resultados combinados.
    :param best_results_df: DataFrame con los mejores resultados.
    :param base_save_dir: Directorio base para guardar los archivos.
    :param filtro_nombre: Nombre del filtro para organizar los archivos.
    :param funcion_nombre: Nombre de la función para organizar los archivos.
    """
    # Crear la estructura de directorios si no existe
    filtro_dir = os.path.join(base_save_dir, filtro_nombre)
    funcion_dir = os.path.join(filtro_dir, funcion_nombre)
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



def obtener_tamano_imagen(path):
    from PIL import Image
    with Image.open(path) as img:
        return img.size[0] * img.size[1]  # Ancho * Alto

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


def filtros_local_size_fijado(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, local_size):
    # Crear un diccionario para almacenar los tiempos de ejecución
    results = {os.path.basename(path): [] for path in lista_paths}
    
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
            results[os.path.basename(path)].append(exec_time)  # Agregar el tiempo de ejecución
        except Exception as e:
            print(f"Error al procesar {os.path.basename(path)} con local_size {local_size}: {e}")
            results[os.path.basename(path)].append(None)  # Manejo de error

    # Crear DataFrame de resultados
    results_general = pd.DataFrame.from_dict(results, orient='index', columns=['Execution Time'])
    results_general.index.name = 'Image Name'
    
    return results_general


def experimento_kernels(lista_paths, filtro, lista_kernels, lista_nombres_kernels, lista_funciones, device_type, local_size, base_save_dir):
    # Inicializar el DataFrame de resultados
    resultados_finales = pd.DataFrame()

    # Procesar cada kernel
    for kernel_code, kernel_name, aplicar_filtro_func in zip(lista_kernels, lista_nombres_kernels, lista_funciones):
        # Obtener resultados para el kernel actual
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

    # Personalizar el gráfico
    plt.title("Tiempos de Ejecución por Kernel")
    plt.xlabel("Nombre de la Imagen")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor visualización
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Ajustar el gráfico para evitar solapamientos

    # Mostrar el gráfico
    plt.show()
        # Guardar o mostrar la gráfica
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

    plt.close()








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

def graficar_columna_especifica(df, local_size, filtro_nombre, funcion_nombre, base_save_dir='graficos'):
    """
    Grafica una columna específica de un DataFrame y guarda el gráfico.

    :param df: DataFrame con MultiIndex en las columnas.
    :param local_size: Tupla indicando el tamaño local a graficar, e.g., (8, 8).
    :param filtro_nombre: Nombre del filtro aplicado.
    :param funcion_nombre: Nombre de la función utilizada.
    :param base_save_dir: Directorio base donde se guardarán los gráficos.
    """
    try:
        # Seleccionar la columna específica
        col_data = df[local_size]
    except KeyError:
        print(f"La columna {local_size} no existe en el DataFrame.")
        return
    
    # Crear la ruta completa
    filtro_dir = os.path.join(base_save_dir, filtro_nombre)
    funcion_dir = os.path.join(filtro_dir, funcion_nombre)
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir el nombre del archivo
    archivo = f'tiempo_ejecucion_{local_size[0]}x{local_size[1]}.png'
    save_path = os.path.join(funcion_dir, archivo)
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    col_data.plot(kind='bar', color='skyblue')
    plt.title(f'Tiempo de Ejecución para Tamaño Local {local_size} - Filtro: {filtro_nombre}')
    plt.xlabel('Nombre de la Imagen')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig(save_path)
    print(f"Gráfico guardado en {save_path}")
    
    plt.close()

def experimento_filtros(lista_paths, filtro, aplicar_filtro_func, kernel_code, kernel_name, device_type, compute_units, processing_elements, filtro_nombre, funcion_nombre, base_save_dir='graficos'):
    """
    Realiza el experimento de filtros y guarda los gráficos generados.

    :param lista_paths: Lista de rutas de imágenes.
    :param filtro: Filtro a aplicar.
    :param aplicar_filtro_func: Función para aplicar el filtro.
    :param kernel_code: Código del kernel de OpenCL.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo (e.g., GPU).
    :param compute_units: Número de unidades de cómputo.
    :param processing_elements: Elementos de procesamiento.
    :param filtro_nombre: Nombre del filtro para organización de gráficos.
    :param funcion_nombre: Nombre de la función para organización de gráficos.
    :param base_save_dir: Directorio base para guardar gráficos.
    :return: DataFrames `results_combined` y `best_results_df`.
    """
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
    filtro_dir = os.path.join(base_save_dir, filtro_nombre)
    funcion_dir = os.path.join(filtro_dir, funcion_nombre)
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

def ejecutar_experimentos(lista_paths, filtros, aplicar_filtro_funcs, kernel_codes, kernel_names, device_type, compute_units, processing_elements, base_save_dir='graficos'):
    """
    Ejecuta experimentos para múltiples filtros y funciones.

    :param lista_paths: Lista de rutas de imágenes.
    :param filtros: Lista de filtros a aplicar.
    :param aplicar_filtro_funcs: Lista de funciones para aplicar cada filtro.
    :param kernel_codes: Lista de códigos de kernels para cada filtro.
    :param kernel_names: Lista de nombres de kernels para cada filtro.
    :param device_type: Tipo de dispositivo (e.g., GPU).
    :param compute_units: Número de unidades de cómputo.
    :param processing_elements: Elementos de procesamiento.
    :param base_save_dir: Directorio base para guardar gráficos.
    """
    # Verificar que todas las listas tengan la misma longitud
    assert len(filtros) == len(aplicar_filtro_funcs) == len(kernel_codes) == len(kernel_names), "Las listas de filtros, funciones, kernels y nombres deben tener la misma longitud."

    for i, filtro in enumerate(filtros):
        aplicar_filtro_func = aplicar_filtro_funcs[i]
        kernel_code = kernel_codes[i]
        kernel_name = kernel_names[i]
        filtro_nombre = f'filtro_{i+1}'  # Puedes personalizar el nombre según el filtro

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
        

        #excel_save_path = os.path.join(base_save_dir, filtro_nombre, funcion_nombre, 'resultados.xlsx')
        #with pd.ExcelWriter(excel_save_path) as writer:
            #resultados.to_excel(writer, sheet_name='Resultados Combinados')
            #best_results_df.to_excel(writer, sheet_name='Mejores Resultados')

        #print(f"DataFrames guardados en {excel_save_path}")
