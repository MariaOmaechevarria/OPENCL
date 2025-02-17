'''
EJECUTA DOS EXPERIMENTOS DE MULTIPLICACIÓN DE MATRICES:
  - DETERMINAR LOCAL SIZE OPTIMO
  - DETERMINAR KERNEL OPTIMO

'''


#Importar librerias
import numpy as np
import pyopencl as cl
import pandas as pd
import matplotlib as plt
import os

#Importar archivo con funciones para hacer los experimentos
import funciones_experimento_matrices as em

#Importar archivo funciones ejecutar kernels
import funciones_ejecutar_kernel_matrices as fm

#Importar Kernel matrices
import kernels_matrices as km

# Path para guardar los archivos (ADAPTAR SEGUN EL ORDENADOR)
path = os.getcwd()

# Datos GPU (ADAPTAR SEGUN LA GPU)
compute_units = 82  
processing_elements = 128

# Datos DEVICE
device_type = cl.device_type.GPU 

#Lista kernels
kernel_codes=[km.MatrixMul_kernel,km.MatrixMul_kernel_local_A,km.MatrixMul_Local_Tiles]
kernel_names=["MatrixMul_kernel","MatrixMul_kernel_local_A","MatrixMul_Local_Tiles"]
aplicar_funcs=[fm.mult_mat_basica,fm.mult_mat_local,fm.mult_mat_local_tiles]

'''
EXPERIMENTO LOCAL SIZE OPTIMO
'''
#Crear path para almacenar resultados experimento determinar local size optimo
base_save_dir = os.path.join(path, "MULTIPLICACION MATRICES/RESULTADOS/")
os.makedirs(base_save_dir, exist_ok=True)

# Ejecutar los experimentos de estudiar local size optimo
em.ejecutar_experimentos(aplicar_funcs, kernel_codes, kernel_names, device_type, compute_units, processing_elements, base_save_dir)

'''
EXPERIMENTO KERNEL OPTIMO
'''

# Directorio base para guardar los resultados de determinar el kernel optimo
base_save_dir = os.path.join(path, "MULTIPLICACION MATRICES/RESULTADOS/Comparacion kernels")
os.makedirs(base_save_dir, exist_ok=True)

#Local size para el experimento
local_size=(8,8)

# Ejecutar experimento
em.experimento_kernels( kernel_codes, kernel_names, aplicar_funcs, device_type, local_size, base_save_dir)