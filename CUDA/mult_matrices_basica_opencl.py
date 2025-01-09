'''
MULTIPLICACIÓN DE MATRICES EN OPENCL: Código en el kernel y host para realizar la multiplicación de matrices en OpenCL
'''

# Librerias a importar
import numpy as np
import pyopencl as cl


# Código del kernel para multiplicación de matrices básico
MatrixMul_kernel = """
__kernel void MatrixMul_kernel(int dim, __global float* A, __global float* B, __global float* C) {
    //Obtener fila y columna work del work item
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    //Inicializar variable resultado
    int resultado = 0;

    //Bucle para realizar la multiplicación, recorrer fila y columna
    for (int i = 0; i < dim; i++) {
        resultado += A[fila * dim + i] * B[i * dim + columna];
    }
    
    //Almacenar resultado en C
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
    # Obtener la  plataforma y el dispositivo
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

    # Crear matriz C inicializada con ceros
    C = np.zeros((dim, dim), dtype=np.float32)

    # Crear buffers para las matrices
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

