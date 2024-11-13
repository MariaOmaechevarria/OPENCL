import pycuda.driver as cuda
import pycuda.compiler as SourceModule
import numpy as np
import time


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

def ejecutar_kernel(dim,A,B,block_value,grid_value):

    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    # Compilar kernel
    mod = SourceModule.SourceModule(MatrixMul_kernel)
    kernel = mod.get_function("MatrixMul_kernel")

    # Crear A, B, C

    C = np.zeros((dim, dim), dtype=np.float32)

    # Transferir datos a la GPU
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)

    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)
    cuda.memcpy_htod(C_gpu, C)

    # Ejecutar kernel y medir tiempo
    start = time.time()
    kernel(np.int32(dim), A_gpu, B_gpu, C_gpu, block=block_value, grid=grid_value)  
    cuda.Context.synchronize()  # Asegurarse de que el kernel ha terminado
    end = time.time()

    # Copiar resultados de C
    cuda.memcpy_dtoh(C, C_gpu)

    # Limpiar
    context.pop()

    return (end-start),C