import numpy as np
import pyopencl as cl
from PIL import Image
import os
import time

''' FUNCIONES PARA APLICAR FILTROS
COMUNES A TODAS LAS FUNCIONES ESPECIFICAS POR KERNEL


'''

#FUNCION PARA PROCESAR IMAGEN, DEVUELVE LOS TAMAÑOS Y LAS IMAGENES IN Y OUT EN FORMATO NP

def procesar_imagen(image_path):
    imagen = Image.open(image_path)
    
    # Convertirla a un array de tres canales
    imagen_np = np.array(imagen).astype(np.uint8)

    # Dimensiones de la imagen
    tam_x, tam_y, _ = imagen_np.shape

    # Crear array para la imagen final
    imagen_out_np = np.empty_like(imagen_np)

    return tam_x, tam_y, imagen_np, imagen_out_np


#PREVIO AL KERNEL

def pre_compilar_kernel(context, kernel_code, kernel_name):
    # Crear el programa con el código fuente del kernel
    program = cl.Program(context, kernel_code).build()
    
    # Extraer el kernel del programa compilado
    kernel = cl.Kernel(program, kernel_name)
    
    return kernel


#PREPARACION DEL KERNEL, CREA EL KERNEL

def preparacion_kernel(device_type, kernel_code, kernel_name):
    # Plataforma y dispositivo
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]

    # Crear contexto y cola de comandos
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Crear el programa y compilarlo
    program = cl.Program(context, kernel_code).build()

    # Crear el kernel
    kernel = cl.Kernel(program, kernel_name)

    return platform, device, context, command_queue, program, kernel

#ESTABLECE ARGUMENTOS DEL KERNEL


def establecer_args_kernel(kernel, args):
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)

#EJECUTA EL KERNEL

def ejecutar_kernel(command_queue, kernel_filter, global_size, local_size):
    event = cl.enqueue_nd_range_kernel(command_queue, kernel_filter, global_size, local_size)
    event.wait()
    return event

#PREPARACION FILTROS

def pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size):
    # Procesar la imagen inicial y la final
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)

    # Preparación del kernel
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)

    # Obtener el tamaño máximo del grupo de trabajo
    wg_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)

    # Crear buffers de imagen
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np_out.nbytes)

    return context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue


#FUNCION QUE APLICA EL FILTRO, DEVUELVE LA IMAGEN FINAL Y EL TIEMPO DE EJECUCION

def aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_out_np, buffer_out):
    # Establecer argumentos del kernel
    establecer_args_kernel(kernel, args_kernel)

    # Ejecutar el kernel (ajustado para que reciba correctamente los argumentos)
    event = ejecutar_kernel(command_queue, kernel, global_size, local_size)

    # Leer el buffer de salida
    cl.enqueue_copy(command_queue, imagen_out_np, buffer_out)

    # Obtener el tiempo de ejecución
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    # Guardar y mostrar la imagen resultante
    imagen_resultante = Image.fromarray(imagen_out_np)

    return imagen_resultante, exec_time


'''
FUNCIONES QUE APLICAN DISTINTOS FILTROS COLOR SIN USAR MEMORIA LOCAL
'''

'''
APLICA FILTRO COLOR
'''


def aplicar_filtro_color(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Obtener las estructuras necesarias para ejecutar el kernel de filtros
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Crear buffer para el filtro
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    # Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    
    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time

'''
APLICA FILTRO COLOR EJECUTANDO EL KERNEL 1000 VECES
'''


def aplicar_filtro_color_100(image_path, filtro,kernel_code,kernel_name, device_type,local_size):
    # Cargar la imagen y convertirla en un array
    imagen = Image.open(image_path)
    imagen_np = np.array(imagen).astype(np.uint8)

    # Dimensiones de la imagen
    tam_x, tam_y, _ = imagen_np.shape
    
    # Preparación del contexto y del kernel
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context)
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    # Compilación del kernel
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    # Crear buffers
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np.nbytes)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]
    
    establecer_args_kernel(kernel, args_kernel)

    # Contar tiempo total
    start_time = time.time()

    # Ejecutar el kernel 100 veces
    for _ in range(1000):
        cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)

    # Esperar a que todas las operaciones terminen
    command_queue.finish()

    # Medir tiempo final
    end_time = time.time()

    # Calcular tiempo total y promedio
    total_time = end_time - start_time
    avg_time = total_time / 1000

    # Leer el buffer de salida
    imagen_out_np = np.empty_like(imagen_np)
    cl.enqueue_copy(command_queue, imagen_out_np, buffer_out)

    # Crear la imagen resultante
    imagen_resultante = Image.fromarray(imagen_out_np)

    return imagen_resultante, avg_time


'''
APLICA FILTRO COLOR DE CUALQUIER TAMAÑO EL FILTRO, NO NECESARIAMENTE CUADRADO
'''

def aplicar_filtro_color_cualquiera(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Obtener las estructuras necesarias para ejecutar el kernel de filtros
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Crear buffer para el filtro
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    # Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[1]),np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time

'''
APLICA EL FILTRO DIVIDIO, PRIMERO HORIZONTAL LUEGO VERTICAL
'''

def aplicar_filtro_color_dividido(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    filtroX, filtroY = filtro

    # Aplicar filtro horizontal
    imagen_in, exec_timeX = aplicar_filtro_color_cualquiera(
        image_path, 
        filtroX, 
        kernel_code, 
        kernel_name, 
        device_type, 
        local_size
    )

    #Aplicar Filtro Vertical

     # Convertirla a un array de tres canales
    imagen_np = np.array(imagen_in).astype(np.uint8)

    # Dimensiones de la imagen
    tam_x, tam_y, _ = imagen_np.shape

    # Crear array para la imagen final
    imagen_np_out = np.empty_like(imagen_np)

    # Plataforma y dispositivo
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]

    # Crear contexto y cola de comandos
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Crear el programa y compilarlo
    program = cl.Program(context, kernel_code).build()

    # Crear el kernel
    kernel = cl.Kernel(program, kernel_name)

    
    # Crear buffers
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroY)
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np.nbytes)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Argumentos del kernel

    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtroY.shape[1]),np.int32(filtroY.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    imagen_resultante, exec_timeY = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, (exec_timeX + exec_timeY)


'''
APLICA FILTRO MEDIAN: NO NECESITA FILTRO, SE HACE LA MEDIANA DE LOS VALROES
'''

def aplicar_filtro_median(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Obtener las estructuras necesarias para ejecutar el kernel de filtros
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time

'''
APLICA EL FILTRO SOBEL, NECESITA DOS FILTROS
'''

def aplicar_filtro_sobel(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Obtener las estructuras necesarias para ejecutar el kernel de filtros
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    #Establecer filtro
    filtroX,filtroY=filtro

    # Crear buffers para los filtros X y Y
    filtro_bufX = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroX)
    filtro_bufY = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroY)

    # Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, filtro_bufX, filtro_bufY, np.int32(filtroX.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time

'''
FUNCIONES QUE APLICAN DISTINTOS FILTROS COLOR USANDO MEMORIA LOCAL
'''


#Aplica filtro usando la memoria local

def aplicar_filtro_local(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Procesar la imagen inicial y la final
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Preparación del kernel
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Crear buffer para el filtro
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    #Crear memoria local
    dim=filtro.shape[0]
    centro = (dim - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centro) * (local_size_y + 2 * centro) * 3  # 3 es por los canales R, G, B

    # Define la memoria local
    local_mem = cl.LocalMemory(local_mem_size)

    # Argumentos del kernel (sin inicializar local_mem aquí)
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time


#Aplica filtro usando la memoria local para filtro de cualquier tamaño, no necesariamente cuadrado

def aplicar_filtro_local_cualquiera(image_path, filtro, kernel_code, kernel_name, device_type, local_size):
    # Procesar la imagen inicial y la final
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Preparación del kernel
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(image_path, kernel_code, kernel_name, device_type, local_size)

    # Crear buffer para el filtro
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    #Crear memoria local
    dimY=filtro.shape[0]
    dimX=filtro.shape[1]
    centroX = (dimX - 1) // 2
    centroY = (dimY - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centroY) * (local_size_y + 2 * centroX) * 3  # 3 es por los canales R, G, B

    # Define la memoria local
    local_mem = cl.LocalMemory(local_mem_size)

    # Argumentos del kernel (sin inicializar local_mem aquí)
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[1]), np.int32(filtro.shape[0]),np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)

    return imagen_resultante, exec_time





#APLICA FILTRO MEMORIA LOCAL DIVIDIDO, PRIMERO FILTRO HORIZONTAL LUEGO VERTICAL

def aplicar_filtro_local_dividido(image_path,filtro,kernel_code, kernel_name, device_type, local_size):
    filtroX,filtroY=filtro
    
    #Aplicar filtro horizontal
    imagen_in,exec_timeX=aplicar_filtro_local_cualquiera(image_path, filtroX, kernel_code, kernel_name, device_type, local_size)

    #Aplicar Filtro Vertical

        # Convertirla a un array de tres canales
    imagen_np = np.array(imagen_in).astype(np.uint8)

        # Dimensiones de la imagen
    tam_x, tam_y, _ = imagen_np.shape

        # Crear array para la imagen final
    imagen_np_out = np.empty_like(imagen_np)

    # Plataforma y dispositivo
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]

    # Crear contexto y cola de comandos
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Crear el programa y compilarlo
    program = cl.Program(context, kernel_code).build()

    # Crear el kernel
    kernel = cl.Kernel(program, kernel_name)
    
    #Crear memoria local
    dimY=filtroY.shape[0]
    dimX=filtroY.shape[1]
    centroX = (dimX - 1) // 2
    centroY = (dimY - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centroY) * (local_size_y + 2 * centroX) * 3  # 3 es por los canales R, G, B

    # Define la memoria local
    local_mem = cl.LocalMemory(local_mem_size)


    # Crear buffers
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroY)
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np.nbytes)

    # Establecer global_size
    global_size = (tam_x, tam_y)

    # Argumentos del kernel

     # Argumentos del kernel (sin inicializar local_mem aquí)
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtroY.shape[1]), np.int32(filtroY.shape[0]),np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_timeY = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    


    return imagen_resultante,(exec_timeX+exec_timeY)


