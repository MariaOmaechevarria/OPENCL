#ARCHIVO PARA EJECUTAR MULTIPLICACIÓN DE MATRICES BASICA EN PYOPENCL EN GOOGLE COLLAB (TANTO CPU COMO GPU)

#NECESARIO PARA EJECUTAR PYOPENCL EN GOOGLE COLLAB (APROX 5 MINUTOS)


!sudo apt update
!sudo apt purge *nvidia* -y
!sudo apt install nvidia-driver-530 -y


!pip install pyopencl
!apt-get install -y pocl-opencl-icd ocl-icd-libopencl1


#IMPORTAR LIBRERIAS Y FUNCIONES

import pyopencl as cl
import numpy as np
import pandas as pd
import kernels_matrices as km

#Para guardar archivos en google collab
'''
from google.colab import drive
drive.mount('/content/drive')
'''


"""FUNCION EN PYOPENCL QUE REALIZA LA MULTIPLICACION DE DOS MATRICES DADAS DE CIERTA DIMENSION"""

def mult_mat_basica(dim:int,local_size:tuple,device_type,program_text,A,B):

  # Plataforma
  platform = cl.get_platforms()[0]

  # Dispositivo (GPU)
  device = platform.get_devices(device_type=device_type)[0]

  # Crear contexto con el dispositivo seleccionado
  context = cl.Context([device])

  # Crear una cola de comandos
  command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

  # Crear el programa y compilarlo
  program = cl.Program(context, program_text)
  try:
       program.build()
  except Exception as e:
    print("Build log:")
    print(program.get_build_info(device, cl.program_build_info.LOG))
    raise e

  # Crear el kernel
  mult_kernel = cl.Kernel(program, 'MatrixMul_kernel1')

  # Obtener el tamaño máximo de workgroup
  max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
  print(f"Tamaño máximo de workgroup: {max_work_group_size}")

  # Crear matrices
  C = np.zeros((dim, dim), dtype=np.int32)

  # Crear Buffers
  bufA = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
  bufB = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
  bufC = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)  # C solo necesita espacio de escritura

  # Argumentos del kernel
  mult_kernel.set_arg(0, np.int32(dim))  # Dimensión como int32
  mult_kernel.set_arg(1, bufA)            # Buffer A
  mult_kernel.set_arg(2, bufB)            # Buffer B
  mult_kernel.set_arg(3, bufC)            # Buffer C

  # Ejecutar el kernel y registrar el evento
  global_size = (dim, dim)  # Tamaño global
  event = cl.enqueue_nd_range_kernel(command_queue, mult_kernel, global_size, local_size)

  # Esperar a que se complete el evento
  event.wait()

  # Obtener el tiempo de ejecución 
  exec_time = 1e-9 * (event.profile.end - event.profile.start)

  # Leer el buffer C
  cl.enqueue_copy(command_queue, C, bufC).wait()  # Asegúrate de que la operación se complete


  return exec_time,C

"""FUNCION MAIN QUE REALIZA LA MULT DE MATRICES PARA VARIAS DIMENSIONES Y DISTINTOS LOCAL SIZE(Nº WORK ITEMS EN CADA WORK_GROUP)"""

def main():

  device_type=cl.device_type.GPU

  index = [(f"({2 ** i}/{2 ** i})" if i != 0 else "(1/1)") for i in range(0, 5)]
  columns = [2 ** i for i in range(1, 14)]  # 2^1 a 2^13 (de 2 a 8192)
  results_df = pd.DataFrame(index=index, columns=columns)
  i=1
  while i<=32:

    local_size=(i,i)
    dim=i

    while dim<=8192:

       A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
       B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)

       exec_time,C=mult_mat_basica(dim,local_size,device_type,km.MatrixMul_kernel1,A,B)
       print(A,B,C)

       results_df.loc[f"({i}/{i})", dim] = exec_time if exec_time is not None else "NP"

       dim*=2

       del A,B

    i*=2

  #Guardar los resultados en google collab
  results_df.to_csv('/content/drive/My Drive/Colab Notebooks/TFG_OPENCL/MULTIPLICACION DE MATRICES/Mult_Mat_Basica_GPU.csv',index=True)

  return results_df


results_df=main()
