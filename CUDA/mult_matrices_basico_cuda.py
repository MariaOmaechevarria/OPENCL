import os
import pycuda.driver as cuda
import pycuda.compiler as SourceModule
import numpy as np
import time

# Configura las rutas necesarias para Visual Studio y el SDK de Windows
vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
sdk_include_path = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
sdk_lib_path = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"

# Agrega las rutas a la variable PATH
os.environ["PATH"] += f";{vs_path};{sdk_include_path};{sdk_lib_path}"

# Kernel CUDA para multiplicación de matrices
MatrixMul_kernel = """
__global__ void MatrixMul_kernel(int dim, float* A, float* B, float* C) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int columna = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < dim && columna < dim) {
        float resultado = 0.0f;

        for (int i = 0; i < dim; i++) {
            resultado += A[fila * dim + i] * B[i * dim + columna];
        }

        C[fila * dim + columna] = resultado;
    }
}
"""

# Función para ejecutar el kernel y medir tiempo con CUDA events
def ejecutar_kernel(dim, A, B, block_value, grid_value):
    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()

    try:
        # Compilar kernel
        mod = SourceModule.SourceModule(MatrixMul_kernel)
        kernel = mod.get_function("MatrixMul_kernel")

        # Crear matrices
        C = np.zeros((dim, dim), dtype=np.float32)

        # Transferir datos a la GPU
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(C.nbytes)

        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        # Crear eventos para medir tiempo
        start_event = cuda.Event()
        end_event = cuda.Event()

        # Lanzar kernel
        start_event.record()  # Registrar inicio
        kernel(
            np.int32(dim), A_gpu, B_gpu, C_gpu,
            block=block_value, grid=grid_value
        )
        end_event.record()  # Registrar fin

        # Sincronizar eventos
        end_event.synchronize()

        # Calcular tiempo
        tiempo_ms = start_event.time_till(end_event)  # Tiempo en milisegundos

        # Copiar resultados de la GPU
        cuda.memcpy_dtoh(C, C_gpu)

        return tiempo_ms / 1000.0, C  # Convertir a segundos
    finally:
        # Asegurarse de liberar el contexto
        context.pop()

'''
# Configuración para la multiplicación de matrices
dim = 8  # Dimensión de la matriz
A = np.random.rand(dim, dim).astype(np.float32)
B = np.random.rand(dim, dim).astype(np.float32)

block_value = (8, 8, 1)
grid_value = (dim // 8, dim // 8, 1)

# Ejecutar y medir tiempo
tiempo, resultado = ejecutar_kernel(dim, A, B, block_value, grid_value)

print(f"Tiempo de ejecución: {tiempo:.6f} segundos")
'''