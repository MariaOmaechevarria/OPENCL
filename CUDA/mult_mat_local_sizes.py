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


def aplicar_kernel_local_sizes():

  index = [(f"({2 ** i}/{2 ** i})" if i != 0 else "(1/1)") for i in range(0, 5)]
  columns = [2 ** i for i in range(1, 14)]  # 2^1 a 2^13 (de 2 a 8192)
  results_df = pd.DataFrame(index=index, columns=columns)
  i=1
  while i<=32:

    local_size=(i,i)
    if i==1:
        dim=2

    else:
      dim=i

    while dim<=8192:

       A = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
       B = np.random.randint(0, 10, size=(dim, dim)).astype(np.int32)
       
       grid = (dim // i, dim // i)
       block = (i, i,1)

       exec_time,C=cuda.ejecutar_kernel(dim,A,B,block,grid)

       results_df.loc[f"({i}/{i})", dim] = exec_time if exec_time is not None else "NP"


       dim*=2

       del A,B


    i*=2
    
  

  return results_df
