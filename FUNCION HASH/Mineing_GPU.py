import pyopencl as cl
import numpy as np
import os
import pandas as pd
import struct



def mining_GPU(kernel_name, kernel_code, device_type, header, target,global_size,local_size):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
   
    # Crear el kernel
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

   
    # Tamaño del header del bloque que se va a minar
    HEADER_SIZE = len(header)
   
    # Creación de buffers necesarios
    header_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header)
    result_nonce = np.zeros(1, dtype=np.uint32)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result_nonce.nbytes)
    found = np.zeros(1, dtype=np.int32)
    found_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=found)
    lock = np.zeros(1, dtype=np.int32)
    lock_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=lock)
    winner_id = np.array([-1], dtype=np.int32)
    winner_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=winner_id)

    # Buffer auxiliar para poder calcular la función SHA256
    mensaje_procesado = np.zeros(64, dtype=np.uint32)
    mensaje_procesado_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=mensaje_procesado)
   
    # Buffer para almacenar hash obtenido
    hash_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=32)
    hash_final = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=32)
   
    # Valor inicial del nonce, siempre 0
    start_nonce = 0
   
    # Convertir el target a buffer de bytes
    target_buffer = np.zeros(4, dtype=np.uint64)
    target_buffer[0] = target
    target_buffer_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target_buffer)
   
    # Establecer argumentos kernel
    kernel.set_arg(0, header_buffer)
    kernel.set_arg(1, np.uint32(start_nonce))
    kernel.set_arg(2, result_buffer)
    kernel.set_arg(3, target_buffer[0])
    #kernel.set_arg(4, found_buffer)
    #kernel.set_arg(5, lock_buffer)
    kernel.set_arg(4, winner_buffer)
    kernel.set_arg(5, mensaje_procesado_buffer)
    kernel.set_arg(6, hash_final)
    kernel.set_arg(7, np.uint64(HEADER_SIZE))
   
    # Ejecutar kernel
    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()
   
    # Copiar valor del nonce final
    hash = np.zeros(32, dtype=np.uint8)
    cl.enqueue_copy(command_queue, result_nonce, result_buffer).wait()
    cl.enqueue_copy(command_queue, hash, hash_final).wait()
   
    # Obtener tiempos
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return exec_time, result_nonce

# Lista de targets desde más fácil hasta más difícil
targets = [np.uint64(0x00FFFFFFFFFFFFFFFF)]

# Ejecutar kernel
kernel_name = "kernel_mining"
device_type = cl.device_type.GPU

result = []

header = 'abc'
header_bytes = header.encode('utf-8')


# Simulación de campos del encabezado
version = 2  # Versión del bloque (4 bytes)
prev_block_hash = "0000000000000000000babae9ed8f7bb2b8d3f9f97bba97b8b8b8b8b8b8b8b8b"  # Hash del bloque anterior (simulado, 32 bytes en hex)
merkle_root = "4d5f5c9ac7ed8f96b0e8a6b3b1c1a3e5f7d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"  # Merkle root simulado (32 bytes en hex)
timestamp = 1633046400  # Timestamp Unix (ejemplo)
bits = 0x17148edf  # Dificultad en bits (ejemplo de Bitcoin)
nonce = 2083236893  # Nonce simulado

# Construcción del encabezado en binario
header = (
    struct.pack("<I", version) +
    bytes.fromhex(prev_block_hash) +
    bytes.fromhex(merkle_root) +
    struct.pack("<I", timestamp) +
    struct.pack("<I", bits) +
    struct.pack("<I", nonce)
)
print(header)


for target in targets:
    # Llamada a la función de minería
    exec_time, result_nonce = mining_GPU(kernel_name, kernel_mining, device_type, header_bytes, target)

    print(f"Execution time: {exec_time} s")
    print(f"Found nonce: {result_nonce[0]}")
    
    result.append({
        'Target': target,
        'Execution Time (s)': exec_time,
        'Nonce': result_nonce[0]
    })

# Crear DataFrame con los resultados
df = pd.DataFrame(result)
print(df)
