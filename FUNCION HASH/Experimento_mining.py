import pyopencl as cl
import numpy as np
import os
import pandas as pd
import struct
from Mining_GPU import kernel_mining,mining_GPU


# Target para probar

target = np.uint64( 0x00FFFFFFFFFFFFFFFF)

# Nonmbre del kernel
kernel_name = "kernel_mining"

#Device type
device_type = cl.device_type.GPU

# Simulación de campos del encabezado
version = 2  # Versión del bloque (4 bytes)
prev_block_hash = "0000000000000000000babae9ed8f7bb2b8d3f9f97bba97b8b8b8b8b8b8b8b8b"  # Hash del bloque anterior (simulado, 32 bytes en hex)
merkle_root = "4d5f5c9ac7ed8f96b0e8a6b3b1c1a3e5f7d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"  # Merkle root simulado (32 bytes en hex)
timestamp = 1633046400  # Timestamp Unix (ejemplo)
bits = 0x17148edf  # Dificultad en bits (ejemplo de Bitcoin)
nonce = 2083236893  # Nonce simulado

# Construcción del encabezado en binario
header = (
    str(version).encode('utf-8') +  # Convertir version a bytes
    bytes.fromhex(prev_block_hash) +  # Convertir prev_block_hash a bytes desde hexadecimal
    bytes.fromhex(merkle_root) +  # Convertir merkle_root a bytes desde hexadecimal
    timestamp.to_bytes(4, 'little') +  # Convertir timestamp a 4 bytes (little-endian)
    bits.to_bytes(4, 'little') +  # Convertir bits a 4 bytes (little-endian)
    nonce.to_bytes(4, 'little')  # Convertir nonce a 4 bytes (little-endian)
)

#Distintas global y local sizes
global_sizes=[(2**3,),(2**4,),(2**5,),(2**6,),(2**7,),(2**8,),(2**9,),(2**10,),(2**12,),(2**15,),(2**20,)]
local_sizes=[(1,),(2,),(4,),(8,),(16,),(32,),(64,)]


results_gen=[]
for global_size in global_sizes:
    result=[]
    for local_size in local_sizes:
      exec_time, result_nonce = mining_GPU(kernel_name, kernel_mining, device_type, header, target,global_size,local_size)
        
      result.append({
        'Global Size':global_size,
        'Execution Time (s)': exec_time})
    results_gen.append(result)

# Crear DataFrame con los resultados
df = pd.DataFrame(results_gen)
print(df)
