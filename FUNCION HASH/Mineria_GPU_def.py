import pyopencl as cl
import numpy as np
import hashlib
import struct

# Kernel actualizado con depuración
kernel_mining = """

//Constantes necesarias para SHA256

__constant uint H[8] = { 
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
};

__constant uint k[64] = { 
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 
};

// Mini funciones para operaciones de SHA-256

uint Ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

uint Maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

uint Sigma0(uint x) {
    return (x >> 2 | x << 30) ^ (x >> 13 | x << 19) ^ (x >> 22 | x << 10);
}

uint Sigma1(uint x) {
    return (x >> 6 | x << 26) ^ (x >> 11 | x << 21) ^ (x >> 25 | x << 7);
}

uint sigma0(uint x) {
    return (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3);
}

uint sigma1(uint x) {
    return (x >> 17 | x << 15) ^ (x >> 19 | x << 13) ^ (x >> 10);
}

// Implementación de SHA-256

void loc_sha256(__private const unsigned char *input, unsigned long len, __private uint *output) {
    uint w[64];
    size_t bloques = (len / 64) + (len % 64 ? 1 : 0);
    
    //Iniciamos con los valores de las constantes H

    uint hi[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    //Vamos recorriendo los bloques del mensaje

    for (size_t i = 0; i < bloques; i++) {

       //Inicializamos las variables de la a...h con los valores de hi

        uint a = hi[0], b = hi[1], c = hi[2], d = hi[3];
        uint e = hi[4], f = hi[5], g = hi[6], h = hi[7];

        for (size_t t = 0; t < 64; t++) {
            if (t < 16) {
               //Los primeros 16 valores los inicializamos con las priemras 16 palabras(32 bites)

                w[t] = ((uint)input[(i * 64) + t * 4 + 0] << 24) |
                       ((uint)input[(i * 64) + t * 4 + 1] << 16) |
                       ((uint)input[(i * 64) + t * 4 + 2] << 8) |
                       ((uint)input[(i * 64) + t * 4 + 3]);
            } else {
               //Las sigueintes (16 a 64) se calculan con ciertas operaciones

                w[t] = w[t - 16] + sigma0(w[t - 15]) + w[t - 7] + sigma1(w[t - 2]);
            }
            
            //Realizamos 64 iterraciones y calculamos las letras de la a..h

            uint t1 = h + Sigma1(e) + Ch(e, f, g) + k[t] + w[t];
            uint t2 = Sigma0(a) + Maj(a, b, c);

            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        hi[0] += a; hi[1] += b; hi[2] += c; hi[3] += d;
        hi[4] += e; hi[5] += f; hi[6] += g; hi[7] += h;
    }
    //Por ultimo ,al final del ultimo bloque, se obtiene el hash final

    for (int i = 0; i < 8; i++) {
        output[i] = hi[i];
    }
}

// Kernel principal

__kernel void kernel_mining(
    __global unsigned char *block_raw,
    __global uint *target,
    __global uint *nonce,
    __global uint *debug_hash
) {
    //Instruccion para ver si alguna hebra ha conseguido ya el objetivo
    if (*nonce != 0xFFFFFFFF) return;
    
    //Cada hebra inicializa el nonce con su global_id()
    uint cur_nonce = get_global_id(0);

    //Variable auxiliar para almacenar el mensaje
    __private unsigned char my_raw[128] = {0};

    //Copiar el mensaje a la variable auxiliar
    for (int i = 0; i < 80; i++) my_raw[i] = block_raw[i];

    //Instruccion para ver si alguna hebra ha conseguido ya el objetivo
     if (*nonce != 0xFFFFFFFF) return;
    
    //Añadir el nonce a la variable auxiliar que almcena el mensaje
    for (size_t i = 0; i < 4; i++) {
        my_raw[80 + i] = (cur_nonce >> (8 * i)) & 0xFF;
    }

    //Instruccion para ver si alguna hebra ha conseguido ya el objetivo
    if (*nonce != 0xFFFFFFFF) return;

    //Bariable auxiloar para almacenar el hash
    uint hash[8] = {0};

    //Calcular el hash SHA256
    loc_sha256(my_raw, 128, hash);

    //Instruccion para ver si alguna hebra ha conseguido ya el objetivo
     if (*nonce != 0xFFFFFFFF) return;
    
     
    //Copiar el hash a la variable auxiliar para devolverlo
    for (int i = 0; i < 8; i++) {
        debug_hash[i] = hash[i];
    }

    //Instruccion para ver si alguna hebra ha conseguido ya el objetivo
    if (*nonce != 0xFFFFFFFF) return;
    

    //Comprobacion de si el hash obtenido es menor que el target
    //Recorremos los 8  valores del hash ()
    for (int i = 7; i >= 0; i--) {

        //Concatenemos los valores
        uint big_hi = ((hash[i] & 0xFF) << 24) |
                      ((hash[i] & 0xFF00) << 8) |
                      ((hash[i] & 0xFF0000) >> 16) |
                      ((hash[i] & 0xFF000000) >> 24);

        //Comparamos con el target
        if (target[i] > big_hi) {
            //Si es menor exito lo hemos conseguido,actualizamos el nonce
            atomic_xchg(nonce, cur_nonce);
            return;
        }
        //Si es mayor hemos perdido, esa hebra acaba
        if (hash[i] > target[i]) return;

        //Si es igual seguimos en el bucle
    }
}



"""

import numpy as np
import pyopencl as cl
import hashlib
import struct


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
    Realiza múltiples intentos ajustando el rango de nonces entre iteraciones.

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
    kernel = program.__getattr__(kernel_name)  # Obtener el kernel por su nombre
    kernel.set_arg(0, block_buffer)
    kernel.set_arg(1, target_buffer)
    kernel.set_arg(2, nonce_buffer)
    kernel.set_arg(3, debug_hash_buffer)

    # Ejecución en múltiples iteraciones
    for iteration in range(0):
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
            if is_valid:
                return exec_time, nonce[0], hash_value  # Retornar si es válido

        # Ajustar el rango de nonces para la siguiente iteración
        block[76:80] = struct.pack('<I', int.from_bytes(block[76:80], 'little') + global_size[0])

    # Si no se encuentra un nonce válido después de todas las iteraciones
    print("No se encontró un nonce válido.")
    return None, None, None
