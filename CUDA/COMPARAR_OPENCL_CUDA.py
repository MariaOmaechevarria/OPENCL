import pycuda.driver as cuda
import pycuda.compiler as SourceModule
import numpy as np
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pyopencl as cl
import mult_matrices_basico_cuda as cuda
import mult_matrices_basica_opencl as opencl


#COMPARAR OPENCL Y CUDA MULTIPLICANDO MATRICES DE DISTINTOS TAMAÑOS

def comparar(path):

    device_type=device_type = cl.device_type.GPU 

    #Global size y local size
    local_size=(8,8)

    #Block y grid size
    block=(8, 8,1)

    dims=[8,16,32,64,128,256,512,1024,2048,4096,8192]

    results=[]

    for dim in dims:
        A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
        B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)

        grid=(dim//8, dim//8)
        
        exec_time_cl,C_cl=opencl.mult_mat_basica(dim,local_size,device_type,opencl.MatrixMul_kernel1,"MatrixMul_kernel1",A,B)
        exec_time_cuda,C_cuda=cuda.ejecutar_kernel(dim,A,B,block,grid)

         # Añadir los resultados a la lista
        results.append({
            "Dimensión": dim,
            "Tiempo OpenCL (s)": exec_time_cl,
            "Tiempo CUDA (s)": exec_time_cuda
        })

    # Crear un DataFrame con los resultados
    df_results = pd.DataFrame(results)

    # Mostrar la tabla de resultados
    opencl.guardar_dataframes_excel(df_results,path)

    #Hacer un grafico
        # Graficar los tiempos de ejecución
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Dimensión"], df_results["Tiempo OpenCL (s)"], label="OpenCL", marker='o')
    plt.plot(df_results["Dimensión"], df_results["Tiempo CUDA (s)"], label="CUDA", marker='s')

    # Personalizar el gráfico
    plt.xlabel("Dimensión de la Matriz")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.title("Comparación de Tiempos de Ejecución entre OpenCL y CUDA")
    plt.xscale('log')
    plt.xticks(dims)  # Mostrar solo los valores de dims en el eje X
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()


