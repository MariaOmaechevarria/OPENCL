import numpy as np
import pyopencl as cl


'''
FUNCIONES COMUNES PARA MANIPULACIÓN DE MATRICES Y KERNELS EN OPENCL
'''

def preparacion_kernel2(device_type, kernel_code, kernel_name):
    """
    Configura el entorno OpenCL y compila un kernel.

    :param device_type: Tipo de dispositivo OpenCL (e.g., cl.device_type.GPU).
    :param kernel_code: Código fuente del kernel en OpenCL.
    :param kernel_name: Nombre del kernel en el código fuente.
    :return: Tupla (platform, device, context, command_queue, program, kernel).
    """
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

def preparacion_kernel(device_type, kernel_code, kernel_name):
    # Buscar una plataforma que tenga un dispositivo del tipo especificado
    for platform in cl.get_platforms():
        try:
            device = platform.get_devices(device_type=device_type)[0]
            print(f"Seleccionada plataforma: {platform}")
            print(f"Seleccionado dispositivo: {device.name}")
            break
        except IndexError:
            # Si no encuentra un dispositivo de ese tipo en la plataforma, sigue buscando
            continue
    else:
        raise RuntimeError("No se encontró ninguna plataforma con el dispositivo especificado.")

    # Crear contexto y cola de comandos
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Crear el programa y compilarlo
    program = cl.Program(context, kernel_code).build()

    # Crear el kernel
    kernel = cl.Kernel(program, kernel_name)

    return platform, device, context, command_queue, program, kernel



def establecer_args_kernel(kernel, args):
    """
    Configura los argumentos de un kernel.

    :param kernel: Instancia del kernel compilado.
    :param args: Lista de argumentos a pasar al kernel.
    """
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)


def ejecutar_kernel(command_queue, kernel_filter, global_size, local_size):
    """
    Ejecuta un kernel OpenCL y mide su tiempo de ejecución.

    :param command_queue: Cola de comandos de OpenCL.
    :param kernel_filter: Kernel a ejecutar.
    :param global_size: Tamaño global de los datos.
    :param local_size: Tamaño local de los datos.
    :return: Evento OpenCL del kernel.
    """
    event = cl.enqueue_nd_range_kernel(command_queue, kernel_filter, global_size, local_size)
    event.wait()
    return event


def crear_buffers_matrices(A, B, context, dim):
    """
    Crea buffers OpenCL para dos matrices de entrada y una de salida.

    :param A: Matriz de entrada A.
    :param B: Matriz de entrada B.
    :param context: Contexto OpenCL.
    :param dim: Dimensiones de las matrices.
    :return: Tupla (bufA, bufB, bufC, C).
    """
    C = np.zeros((dim, dim), dtype=np.int32)

    bufA = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    bufC = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

    return bufA, bufB, bufC, C


def aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC):
    """
    Aplica un kernel a los datos y devuelve los resultados.

    :param kernel: Kernel a ejecutar.
    :param args_kernel: Lista de argumentos del kernel.
    :param global_size: Tamaño global de los datos.
    :param local_size: Tamaño local de los datos.
    :param command_queue: Cola de comandos de OpenCL.
    :param C: Matriz de salida.
    :param bufC: Buffer de salida.
    :return: Tiempo de ejecución y la matriz resultante.
    """
    establecer_args_kernel(kernel, args_kernel)
    event = ejecutar_kernel(command_queue, kernel, global_size, local_size)
    cl.enqueue_copy(command_queue, C, bufC).wait()
    exec_time = 1e-9 * (event.profile.end - event.profile.start)
    return exec_time, C


'''
FUNCIONES ESPECÍFICAS PARA DIFERENTES IMPLEMENTACIONES DE MULTIPLICACIÓN DE MATRICES
'''

def mult_mat_basica(dim, local_size, device_type, kernel_code, kernel_name, A, B):
    """
    Multiplicación básica de matrices utilizando OpenCL.

    :param dim: Dimensión de las matrices.
    :param local_size: Tamaño del grupo de trabajo local.
    :param device_type: Tipo de dispositivo OpenCL.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel en el código.
    :param A: Matriz A.
    :param B: Matriz B.
    :return: Tiempo de ejecución y matriz resultante.
    """
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)
    global_size = (dim, dim)
    bufA, bufB, bufC, C = crear_buffers_matrices(A, B, context, dim)
    args_kernel = [np.int32(dim), bufA, bufB, bufC]
    exec_time, C = aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC)
    return exec_time, C


def mult_mat_local(dim, local_size, device_type, kernel_code, kernel_name, A, B):
    """
    Multiplicación de matrices utilizando memoria local para A.

    :param dim: Dimensión de las matrices.
    :param local_size: Tamaño del grupo de trabajo local.
    :param device_type: Tipo de dispositivo OpenCL.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel en el código.
    :param A: Matriz A.
    :param B: Matriz B.
    :return: Tiempo de ejecución y matriz resultante.
    """
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)
    global_size = (dim, dim)
    bufA, bufB, bufC, C = crear_buffers_matrices(A, B, context, dim)
    num_elements = dim // local_size[0]
    local_mem_size = local_size[0] * num_elements * np.dtype(np.int32).itemsize
    local_A = cl.LocalMemory(local_mem_size)
    args_kernel = [np.int32(dim), bufA, bufB, bufC, local_A]
    exec_time, C = aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC)
    return exec_time, C


def mult_mat_local_tiles(dim, local_size, device_type, kernel_code, kernel_name, A, B):
    """
    Multiplicación de matrices utilizando memoria local para A y B con división en tiles.

    :param dim: Dimensión de las matrices.
    :param local_size: Tamaño del grupo de trabajo local.
    :param device_type: Tipo de dispositivo OpenCL.
    :param kernel_code: Código fuente del kernel.
    :param kernel_name: Nombre del kernel en el código.
    :param A: Matriz A.
    :param B: Matriz B.
    :return: Tiempo de ejecución y matriz resultante.
    """
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)
    global_size = (dim, dim)
    bufA, bufB, bufC, C = crear_buffers_matrices(A, B, context, dim)
    local_mem_size = local_size[0] * local_size[1] * np.dtype(np.int32).itemsize
    local_A = cl.LocalMemory(local_mem_size)
    local_B = cl.LocalMemory(local_mem_size)
    args_kernel = [np.int32(dim), bufA, bufB, bufC, local_A, local_B]
    exec_time, C = aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC)
    return exec_time, C



