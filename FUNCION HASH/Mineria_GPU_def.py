import pyopencl as cl
import numpy as np
import hashlib
import struct

# Kernel actualizado con depuración
kernel_mining = """
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

    uint hi[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    for (size_t i = 0; i < bloques; i++) {
        uint a = hi[0], b = hi[1], c = hi[2], d = hi[3];
        uint e = hi[4], f = hi[5], g = hi[6], h = hi[7];

        for (size_t t = 0; t < 64; t++) {
            if (t < 16) {
                w[t] = ((uint)input[(i * 64) + t * 4 + 0] << 24) |
                       ((uint)input[(i * 64) + t * 4 + 1] << 16) |
                       ((uint)input[(i * 64) + t * 4 + 2] << 8) |
                       ((uint)input[(i * 64) + t * 4 + 3]);
            } else {
                w[t] = w[t - 16] + sigma0(w[t - 15]) + w[t - 7] + sigma1(w[t - 2]);
            }

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
    if (*nonce != 0xFFFFFFFF) return;

    uint cur_nonce = get_global_id(0);
    __private unsigned char my_raw[128] = {0};
    for (int i = 0; i < 80; i++) my_raw[i] = block_raw[i];
     if (*nonce != 0xFFFFFFFF) return;

    for (size_t i = 0; i < 4; i++) {
        my_raw[80 + i] = (cur_nonce >> (8 * i)) & 0xFF;
    }
    if (*nonce != 0xFFFFFFFF) return;
    uint hash[8] = {0};
    loc_sha256(my_raw, 128, hash);
     if (*nonce != 0xFFFFFFFF) return;

    for (int i = 0; i < 8; i++) {
        debug_hash[i] = hash[i];
    }
    if (*nonce != 0xFFFFFFFF) return;

    for (int i = 7; i >= 0; i--) {
        uint big_hi = ((hash[i] & 0xFF) << 24) |
                      ((hash[i] & 0xFF00) << 8) |
                      ((hash[i] & 0xFF0000) >> 16) |
                      ((hash[i] & 0xFF000000) >> 24);

        if (target[i] > big_hi) {
            atomic_xchg(nonce, cur_nonce);
            return;
        }

        if (hash[i] > target[i]) return;
    }
}



"""

import pyopencl as cl
import numpy as np
import hashlib

   # Validación del nonce
def validate_nonce(block, nonce, target):
        nonce_bytes = int(nonce).to_bytes(4, byteorder='little')
        block[80:84] = nonce_bytes
        hash_value = hashlib.sha256(block).digest()
        hash_value = hashlib.sha256(hash_value).digest()
        hash_int = int.from_bytes(hash_value, byteorder='big')
        return hash_int < target, hash_value

def mining_GPU2(kernel_code,kernel_name, block, target, global_size, local_size,device_type):
 
    # Inicialización de OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Compilación del programa
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    # Configuración de datos
    nonce = np.array([0xFFFFFFFF], dtype=np.uint32)  # Inicialización del nonce a su valor máximo
    debug_hash = np.zeros(8, dtype=np.uint32)  # Para almacenar el hash calculado en el kernel

    # Buffers de memoria
    block_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=block)
    target_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
    nonce_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=nonce)
    debug_hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_hash.nbytes)

    # Configuración del kernel

    kernel.set_arg(0, block_buffer)
    kernel.set_arg(1, target_buffer)
    kernel.set_arg(2, nonce_buffer)
    kernel.set_arg(3, debug_hash_buffer)

    # Ejecución del kernel
    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()

    # Leer resultados
    cl.enqueue_copy(command_queue, nonce, nonce_buffer)
    cl.enqueue_copy(command_queue, debug_hash, debug_hash_buffer)

    exec_time = 1e-9 * (event.profile.end - event.profile.start)


    print(f"Nonce encontrado: {nonce[0]}")
    is_valid, hash_value = validate_nonce(block, nonce[0], int.from_bytes(target.tobytes(), byteorder='big'))
    if is_valid:
        print(f"Nonce válido con hash: {hash_value.hex()}")
    else:
        print(f"Nonce no válido.")

    return exec_time,nonce[0], hash_value




def mining_GPU(kernel_code, kernel_name, block, target, global_size, local_size, device_type, max_iterations=10):
     # Inicialización de OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Crear buffers
    nonce = np.array([0xFFFFFFFF], dtype=np.uint32)
    debug_hash = np.zeros(8, dtype=np.uint32)
    
    block_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=block)
    target_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
    nonce_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=nonce)
    debug_hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_hash.nbytes)
    
    program = cl.Program(context, kernel_code).build()
    kernel = program.__getattr__(kernel_name)
    kernel.set_arg(0, block_buffer)
    kernel.set_arg(1, target_buffer)
    kernel.set_arg(2, nonce_buffer)
    kernel.set_arg(3, debug_hash_buffer)

    for iteration in range(max_iterations):
        # Ejecución del kernel
        event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
        event.wait()

        # Leer resultados
        cl.enqueue_copy(command_queue, nonce, nonce_buffer)
        cl.enqueue_copy(command_queue, debug_hash, debug_hash_buffer)

        exec_time = 1e-9 * (event.profile.end - event.profile.start)

        if nonce[0] != 0xFFFFFFFF:
            

            is_valid, hash_value = validate_nonce(block, nonce[0], int.from_bytes(target.tobytes(), byteorder='big'))
            if is_valid:
                
                return exec_time,nonce[0], hash_value

            else:
                print(f"Nonce no válido.")

        # Ajustar rango para la siguiente iteración
        block[76:80] = struct.pack('<I', int.from_bytes(block[76:80], 'little') + global_size[0])

    print("No se encontró un nonce válido.")
    return None, None,None
