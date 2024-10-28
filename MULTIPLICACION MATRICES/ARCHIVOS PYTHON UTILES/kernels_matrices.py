
#KERNEL MULTIPLICACION MATRICES TILES

MatrixMul_Local_Memory="""
__kernel void MatrixMul_Local_Memory(int N,__global float* A, __global float* B, __global float* C, __local float* sh_A, __local float* sh_B) {
    // Obtener la información de los índices
    int by = get_group_id(1);  // blockIdx.y
    int bx = get_group_id(0);  // blockIdx.x

    int ty = get_local_id(1);  // threadIdx.y
    int tx = get_local_id(0);  // threadIdx.x

    // Asumiendo TILE_WIDTH es el tamaño de grupo local (local workgroup size)
    int TILE_WIDTH = get_local_size(0);  // Debe ser igual a get_local_size(1)

    // C[i,j]
    int i = TILE_WIDTH * by + ty;
    int j = TILE_WIDTH * bx + tx;

    // Inicializar el valor de la celda de C
    float value = 0.0f;

    // Loop para la multiplicación de matrices en bloques
    for (int phase = 0; phase < N / TILE_WIDTH; phase++) {
        // Cargar los sub-bloques (tiles) de A y B en la memoria local
        sh_A[ty * TILE_WIDTH + tx] = A[i * N + (phase * TILE_WIDTH + tx)];
        sh_B[ty * TILE_WIDTH + tx] = B[(phase * TILE_WIDTH + ty) * N + j];
        
        // Sincronizar los hilos para asegurar que toda la memoria local esté cargada
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calcular el producto punto de los sub-bloques
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty * TILE_WIDTH + k] * sh_B[k * TILE_WIDTH + tx];
        }

        // Sincronizar los hilos antes de cargar el siguiente bloque
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Asignar el valor calculado a la matriz C
    C[i * N + j] = value;
}


"""


#KERNEL MULTIPLICACION MATRICES MEMORIA LOCAL A

MatrixMul_kernel_localA_coallesced="""
    __kernel void MatrixMul_kernel_localA_coallesced(int dim,__global float *A,__global float *B,__global float *C,__local float *lA)
{
 //Get the index of the work-item
 int iCol = get_global_id(0);
 int iRow = get_global_id(1);
 int localIdx = get_local_id(0);
 int localSizex = get_local_size(0);
 
 float result = 0.0f;
 int numElements = dim/localSizex;
 
 for(int i=0; i<numElements ; i++)
 {
    lA[i*localSizex + localIdx] = A[iRow*dim + i*localSizex +localIdx];
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 
 for(int i=0;i< dim;++i)
 {
         result += lA[i]*B[i*dim + iCol];
 }
 C[iRow*dim + iCol] = result;
}
"""



#KERNEL MULTIPLICACION MATRICES BASICO

MatrixMul_kernel1="""__kernel void MatrixMul_kernel1(int dim, __global int* A, __global int* B, __global int* C) {
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    int resultado = 0;

    for (int i = 0; i < dim; i++) {
        resultado += A[fila * dim + i] * B[i * dim + columna];
    }

    C[fila * dim + columna] = resultado;
}
"""