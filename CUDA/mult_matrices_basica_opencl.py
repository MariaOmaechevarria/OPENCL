# KERNEL MULTIPLICACIÓN MATRICES BÁSICO
import numpy as np
import pyopencl as cl
import os
import pandas as pd

# Código del kernel para multiplicación de matrices básico
MatrixMul_kernel1 = """
__kernel void MatrixMul_kernel1(int dim, __global float* A, __global float* B, __global float* C) {
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    int resultado = 0;

    for (int i = 0; i < dim; i++) {
        resultado += A[fila * dim + i] * B[i * dim + columna];
    }

    C[fila * dim + columna] = resultado;
}
"""

# FUNCIÓN PARA PREPARAR EL KERNEL (PLATAFORMA, DISPOSITIVO, CONTEXTO, COLA Y KERNEL)
def preparacion_kernel(
    device_type: cl.device_type, 
    kernel_code: str, 
    kernel_name: str
) -> tuple[cl.Platform, cl.Device, cl.Context, cl.CommandQueue, cl.Program, cl.Kernel]:
    """
    Prepara el entorno de OpenCL, compila el kernel y devuelve los objetos necesarios para ejecutarlo.

    Inputs:
    - device_type (cl.device_type): Tipo de dispositivo OpenCL (CPU/GPU).
    - kernel_code (str): Código fuente del kernel.
    - kernel_name (str): Nombre del kernel dentro del código.

    Outputs:
    - tuple: 
        - cl.Platform: Plataforma seleccionada.
        - cl.Device: Dispositivo seleccionado.
        - cl.Context: Contexto creado.
        - cl.CommandQueue: Cola de comandos con perfilado habilitado.
        - cl.Program: Programa OpenCL compilado.
        - cl.Kernel: Kernel listo para ser ejecutado.
    """
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


# FUNCIÓN PARA ESTABLECER LOS ARGUMENTOS DEL KERNEL
def establecer_args_kernel(kernel: cl.Kernel, args: list) -> None:
    """
    Asigna argumentos al kernel.

    Inputs:
    - kernel (cl.Kernel): Kernel al que se asignarán los argumentos.
    - args (list): Lista de argumentos que se pasarán al kernel.

    Outputs:
    - None
    """
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)


# FUNCIÓN PARA EJECUTAR EL KERNEL
def ejecutar_kernel(
    command_queue: cl.CommandQueue, 
    kernel_filter: cl.Kernel, 
    global_size: tuple[int], 
    local_size: tuple[int]
) -> cl.Event:
    """
    Ejecuta el kernel y espera a que finalice.

    Inputs:
    - command_queue (cl.CommandQueue): Cola de comandos para enviar la tarea.
    - kernel_filter (cl.Kernel): Kernel que se ejecutará.
    - global_size (tuple[int]): Tamaño global de los hilos.
    - local_size (tuple[int]): Tamaño local (workgroup) de los hilos.

    Outputs:
    - cl.Event: Evento que contiene información del perfilado de ejecución.
    """
    event = cl.enqueue_nd_range_kernel(command_queue, kernel_filter, global_size, local_size)
    event.wait()
    return event


# FUNCIÓN PARA CREAR BUFFERS PARA LAS MATRICES
def crear_buffers_matrices(
    A: np.ndarray, 
    B: np.ndarray, 
    context: cl.Context, 
    dim: int
) -> tuple[cl.Buffer, cl.Buffer, cl.Buffer, np.ndarray]:
    """
    Crea los buffers necesarios para almacenar las matrices en la GPU.

    Inputs:
    - A (np.ndarray): Matriz A (primer operando).
    - B (np.ndarray): Matriz B (segundo operando).
    - context (cl.Context): Contexto de OpenCL.
    - dim (int): Dimensión de las matrices.

    Outputs:
    - tuple:
        - cl.Buffer: Buffer de la matriz A.
        - cl.Buffer: Buffer de la matriz B.
        - cl.Buffer: Buffer de la matriz C (resultado).
        - np.ndarray: Matriz C inicializada con ceros.
    """
    # Crear matriz C inicializada con ceros
    C = np.zeros((dim, dim), dtype=np.float32)

    # Crear buffers
    bufA = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    bufC = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

    return bufA, bufB, bufC, C


# FUNCIÓN PARA APLICAR EL KERNEL
def aplicar_kernel(
    kernel: cl.Kernel, 
    args_kernel: list, 
    global_size: tuple[int], 
    local_size: tuple[int], 
    command_queue: cl.CommandQueue, 
    C: np.ndarray, 
    bufC: cl.Buffer
) -> tuple[float, np.ndarray]:
    """
    Establece argumentos, ejecuta el kernel y obtiene los resultados.

    Inputs:
    - kernel (cl.Kernel): Kernel a ejecutar.
    - args_kernel (list): Argumentos del kernel.
    - global_size (tuple[int]): Tamaño global de los hilos.
    - local_size (tuple[int]): Tamaño local (workgroup) de los hilos.
    - command_queue (cl.CommandQueue): Cola de comandos para enviar tareas.
    - C (np.ndarray): Matriz de salida inicializada.
    - bufC (cl.Buffer): Buffer de salida para la matriz C.

    Outputs:
    - tuple:
        - float: Tiempo de ejecución del kernel en segundos.
        - np.ndarray: Matriz C con los resultados de la multiplicación.
    """
    # Establecer argumentos del kernel
    establecer_args_kernel(kernel, args_kernel)

    # Ejecutar el kernel
    event = ejecutar_kernel(command_queue, kernel, global_size, local_size)

    # Leer el buffer de salida
    cl.enqueue_copy(command_queue, C, bufC).wait()

    # Obtener el tiempo de ejecución
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return exec_time, C


# FUNCIÓN PARA REALIZAR MULTIPLICACIÓN DE MATRICES BÁSICA
def mult_mat_basica(
    dim: int, 
    local_size: tuple[int], 
    device_type: cl.device_type, 
    kernel_code: str, 
    kernel_name: str, 
    A: np.ndarray, 
    B: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Realiza la multiplicación de matrices utilizando un kernel OpenCL básico.

    Inputs:
    - dim (int): Dimensión de las matrices.
    - local_size (tuple[int]): Tamaño local (workgroup) de los hilos.
    - device_type (cl.device_type): Tipo de dispositivo OpenCL (CPU/GPU).
    - kernel_code (str): Código fuente del kernel.
    - kernel_name (str): Nombre del kernel dentro del código.
    - A (np.ndarray): Matriz A (primer operando).
    - B (np.ndarray): Matriz B (segundo operando).

    Outputs:
    - tuple:
        - float: Tiempo de ejecución del kernel en segundos.
        - np.ndarray: Matriz resultante de la multiplicación.
    """
    # Preparar el kernel y obtener los objetos necesarios
    platform, device, context, command_queue, program, kernel = preparacion_kernel(device_type, kernel_code, kernel_name)

    # Tamaño global
    global_size = (dim, dim)

    # Crear buffers para las matrices
    bufA, bufB, bufC, C = crear_buffers_matrices(A, B, context, dim)

    # Configurar los argumentos del kernel
    args_kernel = [np.int32(dim), bufA, bufB, bufC]

    # Ejecutar el kernel y obtener los resultados
    exec_time, C = aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC)

    return exec_time, C

