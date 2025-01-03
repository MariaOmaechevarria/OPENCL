'''
MULTIPLICACIÓN DE MATRICES EN OPENCL
'''

# Librerias a importar
import numpy as np
import pyopencl as cl


# Código del kernel para multiplicación de matrices básico
MatrixMul_kernel = """
__kernel void MatrixMul_kernel(int dim, __global float* A, __global float* B, __global float* C) {
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    int resultado = 0;

    for (int i = 0; i < dim; i++) {
        resultado += A[fila * dim + i] * B[i * dim + columna];
    }

    C[fila * dim + columna] = resultado;
}
"""

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
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]

    # Crear contexto y cola de comandos
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Crear el programa y compilarlo
    program = cl.Program(context, kernel_code).build()

    # Crear el kernel
    kernel = cl.Kernel(program, kernel_name)

    # Tamaño global
    global_size = (dim, dim)

    # Crear buffers para las matrices
    # Crear matriz C inicializada con ceros
    C = np.zeros((dim, dim), dtype=np.float32)

    # Crear buffers
    bufA = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    bufC = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

    # Configurar los argumentos del kernel
    args_kernel = [np.int32(dim), bufA, bufB, bufC]

    # Ejecutar el kernel y obtener los resultados
    for i, arg in enumerate(args_kernel):
        kernel.set_arg(i, arg)

    # Ejecutar el kernel
    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()

    # Leer el buffer de salida
    cl.enqueue_copy(command_queue, C, bufC).wait()

    # Obtener el tiempo de ejecución
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return exec_time, C

