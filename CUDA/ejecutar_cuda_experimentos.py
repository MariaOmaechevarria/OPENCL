'''
ARCHIVO QUE EJECUTA 2 EXPERIMENTOS:
   - CUDA VS OPENCL
   - CUDA MULTIPLICACION MATRICES
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
import funciones_experimentos_cuda as ex


#Importar libreria
import os

'''

# Configura las rutas necesarias para Visual Studio y el SDK de Windows
vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
sdk_include_path = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
sdk_lib_path = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"

# Agrega las rutas a la variable PATH
os.environ["PATH"] += f";{vs_path};{sdk_include_path};{sdk_lib_path}"
'''


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

'''
$env:Path="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64;$env:Path"        
PS C:\Users\Eevee\OPENCL> $env:Path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;$env:Path"

'''