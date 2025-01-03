'''
KERNEL MULTIPLICACIÓN MATRICES
'''

#KERNEL MULTIPLICACION MATRICES TILES: Realiza la multiplicación por bloques usando la memoria local

MatrixMul_Local_Tiles="""
__kernel void MatrixMul_Local_Tiles(int N,__global float* A, __global float* B, __global float* C, __local float* local_A, __local float* local_B) {
    // Obtener la información de los índices

    //Índice del grupo
    int group_id_y = get_group_id(1);  
    int group_id_x = get_group_id(0);  
    
    //Índice Local de la hebras
    int local_id_y = get_local_id(1);  
    int local_id_x = get_local_id(0);  

    // Asumiendo TILE_WIDTH= local_size(0)= local_size(1)
    int TILE_WIDTH = get_local_size(0);  // Debe ser igual a get_local_size(1)

    // Indices del elemento a obtener C[i,j]
    int i = TILE_WIDTH * group_id_y + local_id_y; //(global_size(1))
    int j = TILE_WIDTH * group_id_x + local_id_x; //(global_size(0))

    // Inicializar el valor de la celda de C
    float value = 0.0f;

    // Bucle para la multiplicación de matrices en bloques
    for (int phase = 0; phase < N / TILE_WIDTH; phase++) {

        // Cargar los sub-bloques (tiles) de A y B en la memoria local
        local_A[local_id_y * TILE_WIDTH + local_id_x] = A[i * N + (phase * TILE_WIDTH + local_id_x)];
        local_B[local_id_y * TILE_WIDTH + local_id_x] = B[(phase * TILE_WIDTH + local_id_y) * N + j];
        
        // Sincronizar los hilos para asegurar que toda la memoria local esté cargada
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calcular el producto punto de los sub-bloques
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += local_A[local_id_y * TILE_WIDTH + k] * local_B[k * TILE_WIDTH + local_id_x];
        }

        // Sincronizar los hilos antes de cargar el siguiente bloque
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Asignar el valor calculado a la matriz C
    C[i * N + j] = value;
}


"""


#KERNEL MULTIPLICACION MATRICES MEMORIA LOCAL A : ALMACENA EN LA MEMORIA LOCAL LOS VALORES DE A NECESARISO

MatrixMul_kernel_local_A2="""
    __kernel void MatrixMul_kernel_local_A2(int dim,__global float *A,__global float *B,__global float *C,__local float *local_A)
{
 // Obtener la información de los índices

 //Indices globales
 int global_id_x = get_global_id(0);
 int global_id_y = get_global_id(1); //(row)

 //Indice local size
 int localIdx = get_local_id(0);

// Tamaño grupo trabajo
 int localSizex = get_local_size(0);
 
 //Inicializar variable resultado
 float result = 0.0f;

 //Obtener número de elementos para almacenar
 int numElements = dim/localSizex;
 
 //Acceder a la memoria global y guardar en la memoria local en lA
 for(int i=0; i<numElements ; i++)
 {
    local_A[i*localSizex + localIdx] = A[global_id_y*dim + i*localSizex +localIdx];
 }
 //Barrera para sincronizar las hebras
 barrier(CLK_LOCAL_MEM_FENCE);
 
 //Bucle para realizar la mulriplicación
 for(int i=0;i< dim;++i)
 {
         result += local_A[i]*B[i*dim + global_id_x];
 }
 //Almacenar resultado
 C[global_id_y*dim + global_id_x] = result;
}
"""

MatrixMul_kernel_local_A="""
    __kernel void MatrixMul_kernel_local_A(int dim,__global float *A,__global float *B,__global float *C,__local float *lA)
{
 // Obtener la información de los índices
 int iCol = get_global_id(0);
 int iRow = get_global_id(1);
 int localIdx = get_local_id(0);
 int localSizex = get_local_size(0);
 
 //Inicializar variable resultado
 float result = 0.0f;

 //Obtener número de elementos para almacenar
 int numElements = dim/localSizex;
 
 //Acceder a la memoria global y guardar en la memoria local en lA
 for(int i=0; i<numElements ; i++)
 {
    lA[i*localSizex + localIdx] = A[iRow*dim + i*localSizex +localIdx];
 }
 //Barrera para sincronizar las hebras
 barrier(CLK_LOCAL_MEM_FENCE);
 
 //Bucle para realizar la mulriplicación
 for(int i=0;i< dim;++i)
 {
         result += lA[i]*B[i*dim + iCol];
 }
 C[iRow*dim + iCol] = result;
}
"""



#KERNEL MULTIPLICACION MATRICES BASICO

MatrixMul_kernel="""__kernel void MatrixMul_kernel(int dim, __global int* A, __global int* B, __global int* C) {
    
    //Obtener IDs del work item
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    
    //Inicializar resultado
    int resultado = 0;
    
    //Recorrer la fila y columna correspondiente
    for (int i = 0; i < dim; i++) {
        resultado += A[global_x * dim + i] * B[i * dim + global_y];
    }
    
    //Establecer resultado en C
    C[global_x * dim + global_y] = resultado;
}
"""