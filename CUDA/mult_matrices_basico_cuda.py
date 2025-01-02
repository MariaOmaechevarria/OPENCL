'''
MULTIPLICACIÓN DE MATRICES EN CUDA
'''


#Librerias a importar
import os
import pycuda.driver as cuda
import pycuda.compiler as SourceModule
import numpy as np

# Configura las rutas necesarias para Visual Studio y el SDK de Windows
vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
sdk_include_path = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
sdk_lib_path = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"

# Agrega las rutas a la variable PATH
os.environ["PATH"] += f";{vs_path};{sdk_include_path};{sdk_lib_path}"

# Kernel CUDA para multiplicación de matrices
MatrixMul_kernel = """
__global__ void MatrixMul_kernel(int dim, float* A, float* B, float* C) {
    // Identificar la fila y columna del elemento de la matriz
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int columna = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < dim && columna < dim) {
        float resultado = 0.0f;

        // Calcular el producto de la fila de A y la columna de B
        for (int i = 0; i < dim; i++) {
            resultado += A[fila * dim + i] * B[i * dim + columna];
        }

        // Almacenar el resultado en la matriz C
        C[fila * dim + columna] = resultado;
    }
}
"""

# FUNCIÓN PARA EJECUTAR EL KERNEL Y MEDIR TIEMPO CON CUDA EVENTS
def ejecutar_kernel(
    dim: int,
    A: np.ndarray,
    B: np.ndarray,
    block_value: tuple[int, int, int],
    grid_value: tuple[int, int, int]
) -> tuple[float, np.ndarray]:
    """
    Ejecuta el kernel CUDA para multiplicación de matrices y mide el tiempo de ejecución.

    Inputs:
    - dim (int): Dimensión de las matrices cuadradas.
    - A (np.ndarray): Matriz A (primer operando), en formato de punto flotante (float32).
    - B (np.ndarray): Matriz B (segundo operando), en formato de punto flotante (float32).
    - block_value (tuple[int, int, int]): Dimensiones del bloque de hilos (X, Y, Z).
    - grid_value (tuple[int, int, int]): Dimensiones de la malla de bloques (X, Y, Z).

    Outputs:
    - tuple[float, np.ndarray]:
        - float: Tiempo de ejecución del kernel en segundos.
        - np.ndarray: Matriz resultante de la multiplicación.
    """
    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)  # Seleccionar el primer dispositivo CUDA disponible
    context = device.make_context()  # Crear un contexto para gestionar la GPU

    try:
        # Compilar el kernel CUDA
        mod = SourceModule.SourceModule(MatrixMul_kernel)
        kernel = mod.get_function("MatrixMul_kernel")

        # Crear una matriz de salida inicializada con ceros
        C = np.zeros((dim, dim), dtype=np.float32)

        # Reservar memoria en la GPU para las matrices
        A_gpu = cuda.mem_alloc(A.nbytes)  # Reservar memoria para la matriz A
        B_gpu = cuda.mem_alloc(B.nbytes)  # Reservar memoria para la matriz B
        C_gpu = cuda.mem_alloc(C.nbytes)  # Reservar memoria para la matriz C

        # Transferir datos desde el host (CPU) a la GPU
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        # Crear eventos para medir el tiempo de ejecución
        start_event = cuda.Event()  # Evento de inicio
        end_event = cuda.Event()  # Evento de fin

        # Lanzar el kernel
        start_event.record()  # Registrar el inicio del evento
        kernel(
            np.int32(dim), A_gpu, B_gpu, C_gpu,
            block=block_value, grid=grid_value
        )
        end_event.record()  # Registrar el fin del evento

        # Sincronizar los eventos para asegurar que la ejecución ha finalizado
        end_event.synchronize()

        # Calcular el tiempo de ejecución en milisegundos
        tiempo_ms = start_event.time_till(end_event)  # Tiempo en ms

        # Transferir los datos de la matriz C desde la GPU al host (CPU)
        cuda.memcpy_dtoh(C, C_gpu)

        # Retornar el tiempo de ejecución (en segundos) y la matriz resultante
        return tiempo_ms / 1000.0, C
    finally:
        # Asegurarse de liberar el contexto para evitar bloqueos
        context.pop()



