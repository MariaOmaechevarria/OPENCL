import pyopencl as cl
import numpy as np
import hashlib


import pyopencl as cl
import numpy as np
import hashlib

# Kernel actualizado

import pyopencl as cl
import numpy as np
import hashlib

# Kernel actualizado con depuración y correcciones
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

uint rotr(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

uint S0(uint x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint S1(uint x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

uint Ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

uint Maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

uint E0(uint x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

uint E1(uint x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

void preprocesar(__private uchar* data, __global uint* data_p, ulong original_byte_len) {
    ulong bit_len = original_byte_len * 8;
    ulong total_size = original_byte_len + 1 + ((56 - (original_byte_len + 1) % 64) % 64) + 8;

    for (int i = 0; i < total_size / 4; i++) {
        data_p[i] = 0;
    }

    for (int i = 0; i < original_byte_len; i++) {
        data_p[i / 4] |= (data[i] << (24 - (i % 4) * 8));
    }

    data_p[original_byte_len / 4] |= 0x80 << (24 - (original_byte_len % 4) * 8);
    data_p[(total_size / 4) - 2] = (uint)(bit_len >> 32);
    data_p[(total_size / 4) - 1] = (uint)(bit_len & 0xFFFFFFFF);
}

void SHA256(__private uchar* mensaje, __private uchar* hash, ulong original_byte_len, __global uint* mensaje_procesado) {
    preprocesar(mensaje, mensaje_procesado, original_byte_len);

    uint h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = H[i];
    }

    ulong total_size = original_byte_len + 1 + ((56 - (original_byte_len + 1) % 64) % 64) + 8;
    int num_blocks = total_size / 64;

    for (int i = 0; i < num_blocks; i++) {
        uint w[64];
        for (int j = 0; j < 16; j++) {
            w[j] = mensaje_procesado[i * 16 + j];
        }

        for (int j = 16; j < 64; j++) {
            w[j] = S1(w[j-2]) + w[j-7] + S0(w[j-15]) + w[j-16];
        }

        uint a = h[0], b = h[1], c = h[2], d = h[3];
        uint e = h[4], f = h[5], g = h[6], h_temp = h[7];

        for (int j = 0; j < 64; j++) {
            uint tmp1 = h_temp + E1(e) + Ch(e, f, g) + k[j] + w[j];
            uint tmp2 = E0(a) + Maj(a, b, c);
            h_temp = g;
            g = f;
            f = e;
            e = d + tmp1;
            d = c;
            c = b;
            b = a;
            a = tmp1 + tmp2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_temp;
    }

    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 3] = h[i] & 0xFF;
        hash[i * 4 + 2] = (h[i] >> 8) & 0xFF;
        hash[i * 4 + 1] = (h[i] >> 16) & 0xFF;
        hash[i * 4] = (h[i] >> 24) & 0xFF;
    }
}

__kernel void kernel_mining(__global const char* data, 
                            uint start_nonce, 
                            __global uint* new_nonce, 
                            __global const ulong* target, 
                            __global int* winner_id,
                            __global uint* mensaje_procesado, 
                            __global uchar* winning_hash,
                            ulong data_len,
                            __global uchar* debug_data,
                            __global uchar* debug_hash) {

    uint nonce = start_nonce + get_global_id(0); 
    uint global_id = get_global_id(0); 

    __private uchar hash_buffer[32];

    if (*winner_id != -1) {
        return;
    }

    const ulong max_len = 256; 
    uchar combined_data_local[max_len];
    ulong new_len = data_len;

    // Copiar el header inicial
    for (ulong i = 0; i < data_len; i++) {
        combined_data_local[i] = data[i];
    }

    // Convertir el nonce en caracteres y añadirlo al final
    uint temp_nonce = nonce;
    uchar nonce_buffer[10];
    int nonce_length = 0;

    do {
        nonce_buffer[nonce_length++] = '0' + (temp_nonce % 10);
        temp_nonce /= 10;
    } while (temp_nonce > 0);

    for (int i = 0; i < nonce_length; i++) {
        combined_data_local[data_len + i] = nonce_buffer[nonce_length - 1 - i];
    }
    new_len += nonce_length;

    // Depuración: Copiar datos combinados en debug_data
    for (ulong i = 0; i < new_len; i++) {
        debug_data[i] = combined_data_local[i];
    }

    // Calcular SHA256
    SHA256(combined_data_local, hash_buffer, new_len, mensaje_procesado);

    // Depuración: Copiar el hash generado
    for (int i = 0; i < 32; i++) {
        debug_hash[i] = hash_buffer[i];
    }

    // Comprobar si el hash cumple con el target
    bool valid_nonce = true;
    for (int i = 0; i < 32; i++) {
        uchar target_byte = (uchar)((target[i / 8] >> (8 * (7 - (i % 8)))) & 0xFF);
        if (hash_buffer[i] > target_byte) {
            valid_nonce = false;
            break;
        } else if (hash_buffer[i] < target_byte) {
            break;
        }
    }

    // Si es válido, intentar escribir el resultado
    if (valid_nonce) {
        if (atomic_cmpxchg(winner_id, -1, global_id) == -1) {
            *new_nonce = nonce;
            for (int i = 0; i < 32; i++) {
                winning_hash[i] = hash_buffer[i];
            }
        }
    }
}





"""

# Python code para ejecutar el kernel y depurar
def mining_GPU(kernel_name, kernel_code, device_type, header, target, global_size, local_size):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    HEADER_SIZE = len(header)

    header_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header)
    result_nonce = np.zeros(1, dtype=np.uint32)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result_nonce.nbytes)
    winner_id = np.array([-1], dtype=np.int32)
    winner_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=winner_id)
    mensaje_procesado = np.zeros(64, dtype=np.uint32)
    mensaje_procesado_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=mensaje_procesado)
    hash_final = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=32)


    
    debug_data = np.zeros(256, dtype=np.uint8)
    debug_data_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_data.nbytes)
    debug_hash = np.zeros(32, dtype=np.uint8)
    debug_hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_hash.nbytes)

    target_buffer = np.zeros(4, dtype=np.uint64)
    target_buffer[0] = target & 0xFFFFFFFFFFFFFFFF
    target_buffer[1] = (target >> 64) & 0xFFFFFFFFFFFFFFFF
    target_buffer[2] = (target >> 128) & 0xFFFFFFFFFFFFFFFF
    target_buffer[3] = (target >> 192) & 0xFFFFFFFFFFFFFFFF
    target_buffer_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target_buffer)

    kernel.set_arg(0, header_buffer)
    kernel.set_arg(1, np.uint32(0))
    kernel.set_arg(2, result_buffer)
    kernel.set_arg(3, target_buffer_buffer)
    kernel.set_arg(4, winner_buffer)
    kernel.set_arg(5, mensaje_procesado_buffer)
    kernel.set_arg(6, hash_final)
    kernel.set_arg(7, np.uint64(HEADER_SIZE))
    kernel.set_arg(8, debug_data_buffer)
    kernel.set_arg(9, debug_hash_buffer)


    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()

    hash = np.zeros(32, dtype=np.uint8)
    cl.enqueue_copy(command_queue, result_nonce, result_buffer).wait()
    cl.enqueue_copy(command_queue, hash, hash_final).wait()
    cl.enqueue_copy(command_queue, debug_data, debug_data_buffer).wait()

    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    print("Datos combinados depurados (hex):", ''.join(f'{byte:02x}' for byte in debug_data))

    return exec_time, result_nonce, hash,debug_data, debug_hash


# Configuración del header y target
header = bytes.fromhex('0abc')
target = 0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
global_size = (2**10,)
local_size = (1,)



def validate_nonce(header, nonce, target):
    combined = header + str(nonce).encode('utf-8')
    hash_value = hashlib.sha256(combined).digest()
    return hash_value, int.from_bytes(hash_value, byteorder='big') < target

# Ejecutar kernel
exec_time, result_nonce, hash, debug_data, debug_hash = mining_GPU(
    "kernel_mining", kernel_mining, cl.device_type.GPU, header, target, global_size, local_size
)

# Mostrar resultados
print(f"Execution time: {exec_time}")
print(f"Nonce encontrado: {result_nonce[0]}")
print(f"Hash encontrado (kernel): {''.join(f'{byte:02x}' for byte in hash)}")

# Validar datos combinados y hash
combined_data = bytes([byte for byte in debug_data if byte != 0])
print(f"Datos combinados (hex): {combined_data.hex()}")
print(f"Hash depurado (kernel): {''.join(f'{byte:02x}' for byte in debug_hash)}")

# Validar nonce fuera del kernel
hash_outside, nonce_valido = validate_nonce(header, result_nonce[0], target)
print(f"SHA256 calculado fuera del kernel: {hash_outside.hex()}")
if nonce_valido:
    print("El nonce encontrado es válido.")
else:
    print("El nonce encontrado no es válido. Revisar el cálculo en el kernel.")

