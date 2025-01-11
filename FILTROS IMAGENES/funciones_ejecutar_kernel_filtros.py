'''
FUNCIONES PARA EJECUTAR LOS KERNELS DE APLICACIÓN DE FILTROS A IMÁGENES
'''

#Librerias a importar
import numpy as np
import pyopencl as cl
from PIL import Image
import time

'''
FUNCIONES PARA APLICAR FILTROS COMUNES A TODAS LAS FUNCIONES ESPECÍFICAS POR KERNEL
'''

def procesar_imagen(image_path: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Procesa una imagen para obtener sus dimensiones y la convierte en un array NumPy.

    :param image_path: Ruta de la imagen a procesar.
    :return: Dimensiones de la imagen (tam_x, tam_y), imagen original como array y array vacío para salida.
    """
    imagen = Image.open(image_path)
    imagen_np = np.array(imagen).astype(np.uint8)
    tam_x, tam_y, _ = imagen_np.shape
    imagen_out_np = np.empty_like(imagen_np)
    return tam_x, tam_y, imagen_np, imagen_out_np

def pre_compilar_kernel(context: cl.Context, kernel_code: str, kernel_name: str) -> cl.Kernel:
    """
    Compila un kernel OpenCL y lo devuelve.

    :param context: Contexto de OpenCL.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :return: Kernel compilado.
    """
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)
    return kernel

def preparacion_kernel(device_type: cl.device_type, kernel_code: str, kernel_name: str) -> tuple:
    """
    Prepara el contexto, cola de comandos y compila el kernel.

    :param device_type: Tipo de dispositivo (CPU, GPU, etc.).
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :return: Plataforma, dispositivo, contexto, cola de comandos, programa compilado y kernel.
    """
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)
    return platform, device, context, command_queue, program, kernel

def establecer_args_kernel(kernel: cl.Kernel, args: list) -> None:
    """
    Establece los argumentos de un kernel.

    :param kernel: Kernel al que se le asignan los argumentos.
    :param args: Lista de argumentos para el kernel.
    """
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)

def ejecutar_kernel(command_queue: cl.CommandQueue, kernel_filter: cl.Kernel, global_size: tuple[int, int], local_size: tuple[int, int]) -> cl.Event:
    """
    Ejecuta un kernel OpenCL.

    :param command_queue: Cola de comandos para la ejecución.
    :param kernel_filter: Kernel a ejecutar.
    :param global_size: Tamaño global de la ejecución.
    :param local_size: Tamaño local de la ejecución.
    :return: Evento de ejecución del kernel.
    """
    event = cl.enqueue_nd_range_kernel(command_queue, kernel_filter, global_size, local_size)
    event.wait()
    return event

def pre_filtros(image_path: str, kernel_code: str, kernel_name: str, device_type: cl.device_type, local_size: tuple[int, int]) -> tuple:
    """
    Prepara la configuración necesaria para aplicar un filtro, incluyendo contexto, buffers y kernel.

    :param image_path: Ruta de la imagen a procesar.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local del trabajo.
    :return: Configuración necesaria para aplicar un filtro.
    """
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np_out.nbytes)
    return context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue

def aplicar_filtro(kernel: cl.Kernel, args_kernel: list, global_size: tuple[int, int], local_size: tuple[int, int],
                   command_queue: cl.CommandQueue, imagen_out_np: np.ndarray, buffer_out: cl.Buffer) -> tuple[Image.Image, float]:
    """
    Aplica un filtro utilizando un kernel OpenCL.

    :param kernel: Kernel OpenCL.
    :param args_kernel: Argumentos del kernel.
    :param global_size: Tamaño global de la ejecución.
    :param local_size: Tamaño local de la ejecución.
    :param command_queue: Cola de comandos.
    :param imagen_out_np: Array de salida de la imagen.
    :param buffer_out: Buffer de salida OpenCL.
    :return: Imagen resultante y tiempo de ejecución del kernel.
    """
    establecer_args_kernel(kernel, args_kernel)
    event = ejecutar_kernel(command_queue, kernel, global_size, local_size)
    cl.enqueue_copy(command_queue, imagen_out_np, buffer_out)
    exec_time = 1e-9 * (event.profile.end - event.profile.start)
    imagen_resultante = Image.fromarray(imagen_out_np)
    return imagen_resultante, exec_time


'''
FUNCIÓN PARA APLICAR UN KERNEL BÁSICO A IMAGENES EN COLOR
'''

def aplicar_filtro_color(image_path: str, filtro: np.ndarray, kernel_code: str, kernel_name: str,
                         device_type: cl.device_type, local_size: tuple[int, int]) -> tuple[Image.Image, float]:
    """
    Aplica un filtro de color utilizando un kernel OpenCL.

    :param image_path: Ruta de la imagen a procesar.
    :param filtro: Filtro a aplicar.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local del trabajo.
    :return: Imagen resultante y tiempo de ejecución promedio.
    """
    #Obetener todos los valores previos a ejecutar el kernel
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(
        image_path, kernel_code, kernel_name, device_type, local_size)
    
    #Global size
    global_size = (tam_x, tam_y)

    #Crear el buffer del filtro
    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    #Argumentos del kernel
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]
    
    #Aplicar el kernel del filtro color
    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    return imagen_resultante, exec_time


'''
APLICA FILTRO COLOR EJECUTANDO EL KERNEL 1000 VECES
'''

def aplicar_filtro_color_100(image_path:str, filtro:list,kernel_code:str,kernel_name:str, device_type:str,local_size:tuple)->tuple[Image.Image, float]:
    """
    Aplica un filtro a una imagen utilizando OpenCL, ejecutando el kernel 1000 veces para medir el tiempo promedio de ejecución.

    Parámetros:
    - image_path (str): Ruta de la imagen a procesar.
    - filtro (ndarray): Filtro que se aplicará a la imagen.
    - kernel_code (str): Código del kernel OpenCL.
    - kernel_name (str): Nombre del kernel OpenCL.
    - device_type (str): Tipo de dispositivo en el que se ejecuta el kernel (CPU o GPU).
    - local_size (int): Tamaño del workgroup en el que se ejecutará el kernel.

    Retorna:
    - imagen_resultante (Image): Imagen resultante después de aplicar el filtro.
    - avg_time (float): Tiempo promedio de ejecución por iteración después de ejecutar el kernel 1000 veces.
    """
    # Cargar la imagen y convertirla en un array
    imagen = Image.open(image_path)
    imagen_np = np.array(imagen).astype(np.uint8)

    # Dimensiones de la imagen
    tam_x, tam_y, _ = imagen_np.shape
    
    # Preparación del contexto y del kernel
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
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

def aplicar_filtro_color_cualquiera(image_path:str, filtro:np.ndarray, kernel_code:str, kernel_name:str, device_type:str, local_size:tuple)->tuple[Image.Image, float]:

    """
    Aplica un filtro a la imagen especificada usando OpenCL.

    Parámetros:
    image_path (str): Ruta de la imagen a procesar.
    filtro (numpy.ndarray): El filtro a aplicar, debe ser un array 2D.
    kernel_code (str): Código fuente del kernel OpenCL.
    kernel_name (str): Nombre del kernel en el código fuente.
    device_type (str): Tipo de dispositivo OpenCL (ej. 'GPU' o 'CPU').
    local_size (tuple): Tamaño del workgroup para la ejecución del kernel.

    Retorna:
    imagen_resultante (numpy.ndarray): Imagen filtrada.
    exec_time (float): Tiempo de ejecución del kernel.
    """
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

def aplicar_filtro_color_dividido(image_path:str, filtro:tuple, kernel_code:str, kernel_name:str, device_type:str, local_size:tuple[int,int])->tuple[Image.Image, float]:
     
    """
    Aplica un filtro dividido (horizontal seguido de vertical) a una imagen utilizando OpenCL.

    Esta función aplica primero un filtro horizontal a la imagen y luego aplica un filtro vertical
    sobre la imagen resultante del filtro horizontal. Los dos filtros se ejecutan utilizando OpenCL,
    y los tiempos de ejecución de ambos filtros se suman para obtener el tiempo total.

    Parámetros:
    image_path (str): Ruta de la imagen a la que se le aplicarán los filtros.
    filtro (tuple): Un par de matrices de filtro (filtroX, filtroY), donde filtroX es el filtro horizontal 
                    y filtroY es el filtro vertical.
    kernel_code (str): Código fuente del kernel OpenCL que implementa el filtro.
    kernel_name (str): Nombre del kernel OpenCL que se ejecutará.
    device_type (str): Tipo de dispositivo OpenCL a utilizar (por ejemplo, "CPU" o "GPU").
    local_size (tuple): Tamaño del workgroup para ejecutar el kernel.

    Retorna:
    tuple: Una tupla que contiene:
        - imagen_resultante (numpy.ndarray): Imagen resultante después de aplicar ambos filtros (horizontal y vertical).
        - exec_time_total (float): El tiempo total de ejecución en segundos (suma de los tiempos de ejecución de ambos filtros).
    """




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

def aplicar_filtro_median(image_path:str, filtro, kernel_code:str, kernel_names:str, device_type:str, local_size:tuple[int,int])->tuple[Image.Image, float]:

    """
    Aplica un filtro de mediana a una imagen utilizando OpenCL.

    Esta función utiliza un kernel OpenCL para aplicar un filtro de mediana sobre la imagen especificada. 
    El filtro se aplica utilizando una implementación basada en OpenCL, donde la imagen es procesada en bloques 
    y el valor de cada píxel es reemplazado por el valor mediano de sus vecinos en un vecindario definido 
    por el filtro.

    Parámetros:
    image_path (str): Ruta de la imagen a la que se le aplicará el filtro de mediana.
    filtro (object): El filtro que se aplicará, generalmente de tipo matriz. Este parámetro no se utiliza directamente
                     en esta versión de la función, pero es parte de la firma de la función.
    kernel_code (str): Código fuente del kernel OpenCL que implementa el filtro de mediana.
    kernel_name (str): Nombre del kernel en el código fuente OpenCL que se ejecutará.
    device_type (str): Tipo de dispositivo OpenCL a utilizar (por ejemplo, "CPU" o "GPU").
    local_size (tuple): Tamaño del workgroup para la ejecución del kernel.

    Retorna:
    tuple: Una tupla que contiene:
        - imagen_resultante (numpy.ndarray): Imagen resultante después de aplicar el filtro de mediana.
        - exec_time (float): El tiempo de ejecución en segundos del filtro de mediana.
    """



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
def aplicar_filtro_sobel(image_path: str, filtro: tuple[np.ndarray, np.ndarray], kernel_code: str, kernel_name: str,
                         device_type: cl.device_type, local_size: tuple[int, int]) -> tuple[Image.Image, float]:
    """
    Aplica el filtro Sobel a una imagen utilizando dos filtros (horizontal y vertical) y un kernel OpenCL.

    :param image_path: Ruta de la imagen a procesar.
    :param filtro: Tupla con los filtros Sobel en direcciones X e Y.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL (CPU, GPU, etc.).
    :param local_size: Tamaño local del trabajo.
    :return: Imagen resultante y tiempo de ejecución.
    """
    context, kernel, buffer_in, buffer_out, tam_x, tam_y, imagen_np, imagen_np_out, command_queue = pre_filtros(
        image_path, kernel_code, kernel_name, device_type, local_size)

    global_size = (tam_x, tam_y)
    filtroX, filtroY = filtro

    filtro_bufX = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroX)
    filtro_bufY = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroY)

    args_kernel = [buffer_in, buffer_out, filtro_bufX, filtro_bufY, np.int32(filtroX.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0])]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    return imagen_resultante, exec_time


'''
FUNCIONES QUE APLICAN DISTINTOS FILTROS COLOR USANDO MEMORIA LOCAL
'''


'''
APLICA FILTRO USANDO LA MEMORIA LOCAL
'''
def aplicar_filtro_local(image_path: str, filtro: np.ndarray, kernel_code: str, kernel_name: str,
                         device_type: cl.device_type, local_size: tuple[int, int]) -> tuple[Image.Image, float]:
    """
    Aplica un filtro a una imagen utilizando memoria local en un kernel OpenCL.

    :param image_path: Ruta de la imagen a procesar.
    :param filtro: Filtro a aplicar.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local del trabajo.
    :return: Imagen resultante y tiempo de ejecución.
    """
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)
    global_size = (tam_x, tam_y)

    context, kernel, buffer_in, buffer_out, _, _, _, _, command_queue = pre_filtros(
        image_path, kernel_code, kernel_name, device_type, local_size)

    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    dim = filtro.shape[0]
    centro = (dim - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centro) * (local_size_y + 2 * centro) * 3  # 3 canales RGB

    local_mem = cl.LocalMemory(local_mem_size)

    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    return imagen_resultante, exec_time


'''
APLICA FILTRO USANDO MEMORIA LOCAL PARA CUALQUIER TAMAÑO DE FILTRO
'''
def aplicar_filtro_local_cualquiera(image_path: str, filtro: np.ndarray, kernel_code: str, kernel_name: str,
                                    device_type: cl.device_type, local_size: tuple[int, int]) -> tuple[Image.Image, float]:
    """
    Aplica un filtro a una imagen utilizando memoria local para filtros de cualquier tamaño.

    :param image_path: Ruta de la imagen a procesar.
    :param filtro: Filtro a aplicar.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local del trabajo.
    :return: Imagen resultante y tiempo de ejecución.
    """
    tam_x, tam_y, imagen_np, imagen_np_out = procesar_imagen(image_path)
    global_size = (tam_x, tam_y)

    context, kernel, buffer_in, buffer_out, _, _, _, _, command_queue = pre_filtros(
        image_path, kernel_code, kernel_name, device_type, local_size)

    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtro)

    dimY, dimX = filtro.shape
    centroX, centroY = (dimX - 1) // 2, (dimY - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centroY) * (local_size_y + 2 * centroX) * 3  # 3 canales RGB

    local_mem = cl.LocalMemory(local_mem_size)

    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtro.shape[1]), np.int32(filtro.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_time = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    return imagen_resultante, exec_time


'''
APLICA FILTRO USANDO MEMORIA LOCAL DIVIDIDO: PRIMERO FILTRO HORIZONTAL Y LUEGO VERTICAL
'''
def aplicar_filtro_local_dividido(image_path: str, filtro: tuple[np.ndarray, np.ndarray], kernel_code: str, kernel_name: str,
                                  device_type: cl.device_type, local_size: tuple[int, int]) -> tuple[Image.Image, float]:
    """
    Aplica un filtro a una imagen utilizando memoria local y dividiendo el procesamiento en dos pasos: 
    filtro horizontal y luego filtro vertical.

    :param image_path: Ruta de la imagen a procesar.
    :param filtro: Filtros en direcciones horizontal y vertical.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel.
    :param device_type: Tipo de dispositivo OpenCL.
    :param local_size: Tamaño local del trabajo.
    :return: Imagen resultante y tiempo total de ejecución (suma de los tiempos de los dos pasos).
    """
    filtroX, filtroY = filtro

    imagen_in, exec_timeX = aplicar_filtro_local_cualquiera(image_path, filtroX, kernel_code, kernel_name, device_type, local_size)

    imagen_np = np.array(imagen_in).astype(np.uint8)
    tam_x, tam_y, _ = imagen_np.shape
    imagen_np_out = np.empty_like(imagen_np)

    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    dimY, dimX = filtroY.shape
    centroX, centroY = (dimX - 1) // 2, (dimY - 1) // 2
    local_size_x, local_size_y = local_size
    local_mem_size = (local_size_x + 2 * centroY) * (local_size_y + 2 * centroX) * 3  # 3 canales RGB

    local_mem = cl.LocalMemory(local_mem_size)

    filtro_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filtroY)
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imagen_np)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imagen_np.nbytes)

    global_size = (tam_x, tam_y)
    args_kernel = [buffer_in, buffer_out, filtro_buf, np.int32(filtroY.shape[1]), np.int32(filtroY.shape[0]), np.int32(imagen_np.shape[1]), np.int32(imagen_np.shape[0]), local_mem]

    imagen_resultante, exec_timeY = aplicar_filtro(kernel, args_kernel, global_size, local_size, command_queue, imagen_np_out, buffer_out)
    return imagen_resultante, (exec_timeX + exec_timeY)
