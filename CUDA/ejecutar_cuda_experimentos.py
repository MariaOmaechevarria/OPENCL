'''
ARCHIVO QUE EJECUTA 2 EXPERIMENTOS:
   - CUDA VS OPENCL
   - CUDA MULTIPLICACION MATRICES
'''

'''
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
import matplotlib.ticker as ticker
import mult_matrices_basica_opencl as opencl
import experimentos_cuda as ex
'''

#Importar libreria
import os

#Archivo con funciones experimentos cuda/opencl
import funciones_experimentos_cuda as ex

#Ruta del archivo (MODIFICAR)
path="C:/Users/Eevee/OPENCL/"

# Crear directorio para guardar los gráficos
save_path = os.path.join(path, 'CUDA/RESULTADOS')
os.makedirs(save_path, exist_ok=True)

#EJECUTAR LA COMPARACION DE OPENCL VS CUDA
ex.comparar(save_path)

#EJECUTAR EL KERNEL DE CUDA CON MUCHOS EJEMPLOS:
ex.experimento_matrices(save_path,funcion_nombre='kernel_cuda')