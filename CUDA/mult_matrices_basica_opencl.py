#KERNEL MULTIPLICACION MATRICES BASICO
import numpy as np
import pyopencl as cl
import os
import pandas as pd

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

#FUNCION PREPRATIVOS ANTES DE CREAR KERNEL+ CREAR KERNEL

def preparacion_kernel(device_type, kernel_code, kernel_name):
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


# ESTABLECER ARGUMENTOS KERNEL
def establecer_args_kernel(kernel, args):
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)

#EJECUTAR KERNEL
def ejecutar_kernel(command_queue, kernel_filter, global_size, local_size):
    event = cl.enqueue_nd_range_kernel(command_queue, kernel_filter, global_size, local_size)
    event.wait()
    return event


#CREA BUFFERS DE LAS MATRICES 

def crear_buffers_matrices(A,B,context,dim):
    #Crear Buffers Matrices
    C = np.zeros((dim, dim), dtype=np.int32)

    # Crear Buffers
    bufA = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    bufC = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes) 
    return bufA,bufB,bufC,C


#DADO UN KERNEL, ESTABLECE ARGUMENTOS,LO EJECUTA Y DEVUELVE RESULTADOS

def aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC):
    # Establecer argumentos del kernel
    establecer_args_kernel(kernel, args_kernel)

    # Ejecutar el kernel (ajustado para que reciba correctamente los argumentos)
    event = ejecutar_kernel(command_queue, kernel, global_size, local_size)

    # Leer el buffer de salida
    cl.enqueue_copy(command_queue, C, bufC).wait()

    # Obtener el tiempo de ejecución
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return  exec_time,C


'''
FUNCIONES ESPECIFICAS PARA CADA TIPO DE KERNEL
'''

#FUNCION PARA REALIZAR MULTIPLICACION BASICA MATRICES

def mult_mat_basica(dim,local_size,device_type,kernel_code,kernel_name,A,B):

    platform, device, context, command_queue, program, kernel=preparacion_kernel(device_type, kernel_code, kernel_name)

    #global size
    global_size=(dim,dim)

    #Crear Buffers Matrices
    bufA,bufB,bufC,C=crear_buffers_matrices(A,B,context,dim)

    #Args kernel
    args_kernel=[np.int32(dim),bufA,bufB,bufC]

    #Ejecutar kernel
    exec_time,C=aplicar_kernel(kernel, args_kernel, global_size, local_size, command_queue, C, bufC)

    return exec_time,C



def guardar_dataframes_excel(resultados,base_save_dir):
   
    # Crear la estructura de directorios si no existe
    funcion_dir = base_save_dir
    os.makedirs(funcion_dir, exist_ok=True)
    
    # Definir la ruta completa del archivo Excel
    excel_save_path = os.path.join(funcion_dir, 'resultados.xlsx')
    
    # Usar xlsxwriter como motor para formatear las celdas durante la escritura
    with pd.ExcelWriter(excel_save_path, engine='xlsxwriter') as writer:
        # Guardar los DataFrames en hojas separadas
        resultados.to_excel(writer, sheet_name='Resultados', index=True)
       
        # Acceder al workbook y crear un formato para números con 6 decimales
        workbook = writer.book
        float_format = workbook.add_format({'num_format': '0.000000'})  # 6 decimales

        # Formatear 'Resultados Combinados'
        worksheet = writer.sheets['Resultados ']
        # Iterar sobre las columnas (empezando en la segunda columna si la primera es índice)
        for idx, col in enumerate(resultados.columns, start=1):  # start=1 para saltar la columna de índice
            worksheet.set_column(idx, idx, 15, float_format)  # 15 es el ancho de columna opcional

    print(f"DataFrames guardados y formateados en Excel en {excel_save_path}")