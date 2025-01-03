'''
ARCHIVO CON LA FUNCION PARA EJECUTAR EL KERNEL DE MINERÍA DE UN BLOQUE DEL BLOCKCHAIN
'''


import pyopencl as cl
import numpy as np
import hashlib




# Validación del nonce
def validate_nonce(block: bytearray, nonce: int, target: int) -> tuple[bool, bytes]:
    """
    Valida si un nonce específico genera un hash que cumple con el objetivo (target).

    Inputs:
    - block (bytearray): Bloque de datos que incluye información como transacciones y el nonce.
    - nonce (int): Número entero que se utiliza como intento de solución.
    - target (int): Valor objetivo para el hash.

    Outputs:
    - tuple[bool, bytes]: 
        - bool: Indica si el hash generado es menor que el objetivo (target).
        - bytes: Hash generado (32 bytes).
    """
    # Convertir el nonce en bytes en formato little-endian
    nonce_bytes = int(nonce).to_bytes(4, byteorder='little')
    # Reemplazar los bytes correspondientes al nonce en el bloque
    block[80:84] = nonce_bytes
    # Calcular el doble hash SHA-256 del bloque
    hash_value = hashlib.sha256(block).digest()
    hash_value = hashlib.sha256(hash_value).digest()
    # Convertir el hash en un número entero para comparación
    hash_int = int.from_bytes(hash_value, byteorder='big')
    # Verificar si el hash es menor que el objetivo
    return hash_int < target, hash_value

# Minería con OpenCL (múltiples iteraciones del kernel)
def mining_GPU(
    kernel_code: str,
    kernel_name: str,
    block: bytearray,
    target: np.ndarray,
    global_size: tuple[int],
    local_size: tuple[int],
    device_type: cl.device_type,
    max_iterations: int = 10,
) -> tuple[float | None, int | None, bytes | None]:
    """
    Ejecuta minería utilizando un kernel OpenCL en la GPU.
    

    Inputs:
    - kernel_code (str): Código del kernel OpenCL.
    - kernel_name (str): Nombre del kernel dentro del código.
    - block (bytearray): Bloque de datos con información para calcular el hash.
    - target (np.ndarray): Array de tipo uint32 que contiene el objetivo (target).
    - global_size (tuple[int]): Tamaño total de los hilos en el espacio global.
    - local_size (tuple[int]): Tamaño de los hilos en cada grupo de trabajo (workgroup).
    - device_type (cl.device_type): Tipo de dispositivo OpenCL (CPU o GPU).
    - max_iterations (int): Número máximo de intentos para encontrar un nonce válido.

    Outputs:
    - tuple[float | None, int | None, bytes | None]:
        - float | None: Tiempo de ejecución del kernel en segundos (si se encuentra un nonce).
        - int | None: Nonce encontrado (si es válido).
        - bytes | None: Hash generado correspondiente al nonce (si es válido).
    """
    # Inicialización de OpenCL
    platform = cl.get_platforms()[0]  # Seleccionar la primera plataforma
    device = platform.get_devices(device_type=device_type)[0]  # Seleccionar el dispositivo (CPU/GPU)
    context = cl.Context([device])  # Crear un contexto para el dispositivo
    # Crear una cola de comandos con perfilado para medir tiempos de ejecución
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Configuración inicial de datos
    nonce = np.array([0xFFFFFFFF], dtype=np.uint32)  # Inicializar el nonce al valor máximo
    debug_hash = np.zeros(8, dtype=np.uint32)  # Espacio para almacenar el hash calculado en el kernel

    # Buffers de memoria en la GPU
    block_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=block)
    target_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
    nonce_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=nonce)
    debug_hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_hash.nbytes)

    # Compilación del kernel
    program = cl.Program(context, kernel_code).build()  # Compilar el código OpenCL
    kernel = cl.Kernel(program, kernel_name)
  
    kernel.set_arg(0, block_buffer)
    kernel.set_arg(1, target_buffer)
    kernel.set_arg(2, nonce_buffer)
    kernel.set_arg(3, debug_hash_buffer)

    
    # Ejecutar kernel
    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()  # Esperar a que termine la ejecución

    # Leer los resultados desde la GPU
    cl.enqueue_copy(command_queue, nonce, nonce_buffer)
    cl.enqueue_copy(command_queue, debug_hash, debug_hash_buffer)

    # Medir el tiempo de ejecución
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    # Validar el nonce encontrado
    if nonce[0] != 0xFFFFFFFF:
        is_valid, hash_value = validate_nonce(block, nonce[0], int.from_bytes(target.tobytes(), byteorder='big'))
        #is_valid=True
        if is_valid:
            return exec_time, nonce[0], hash_value  # Retornar si es válido

    return None, None, None
