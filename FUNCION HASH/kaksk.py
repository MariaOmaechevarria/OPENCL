import pyopencl as cl
import numpy as np
import hashlib

# Kernel actualizado
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

void SHA256(__private uchar* mensaje, __private uchar* hash, ulong len, __global uint* mensaje_procesado) {
    // Implementación completa de SHA256 como ya tienes
}

__kernel void kernel_mining(__global const char* data, 
                            uint start_nonce, 
                            __global uint* new_nonce, 
                            __global const ulong* target, 
                            __global int* winner_id,
                            __global uint* mensaje_procesado, 
                            __global uchar* winning_hash,
                            ulong data_len,
                            __global uchar* debug_data) {
    uint nonce = start_nonce + get_global_id(0); 
    __private uchar hash_buffer[32];
    uchar combined_data[256];

    // Construcción de datos combinados
    int new_len = data_len;
    for (ulong i = 0; i < data_len; i++) {
        combined_data[i] = data[i];
    }
    uint temp_nonce = nonce;
    uchar nonce_str[10];
    int nonce_length = 0;
    do {
        nonce_str[nonce_length++] = '0' + (temp_nonce % 10);
        temp_nonce /= 10;
    } while (temp_nonce > 0);
    for (int i = 0; i < nonce_length; i++) {
        combined_data[data_len + i] = nonce_str[nonce_length - 1 - i];
    }
    new_len += nonce_length;

    // Depuración: copiar datos combinados
    for (int i = 0; i < new_len; i++) {
        debug_data[i] = combined_data[i];
    }

    // Calcular SHA256
    SHA256(combined_data, hash_buffer, new_len, mensaje_procesado);

    // Validar hash contra el target
    bool valid = true;
    for (int i = 0; i < 32; i++) {
        uchar target_byte = (uchar)((target[i / 8] >> (8 * (7 - (i % 8)))) & 0xFF);
        if (hash_buffer[i] > target_byte) {
            valid = false;
            break;
        }
    }

    if (valid && atomic_cmpxchg(winner_id, -1, get_global_id(0)) == -1) {
        *new_nonce = nonce;
        for (int i = 0; i < 32; i++) {
            winning_hash[i] = hash_buffer[i];
        }
    }
}
"""

# Python host code para ejecutar el kernel
def mining_GPU(kernel_name, kernel_code, device_type, header, target, global_size, local_size):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    result_nonce = np.array([-1], dtype=np.uint32)
    winner_id = np.array([-1], dtype=np.int32)

    header_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header)
    nonce_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result_nonce.nbytes)
    target_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
    winner_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=winner_id)

    debug_data = np.zeros(256, dtype=np.uint8)
    debug_data_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug_data.nbytes)

    kernel.set_arg(0, header_buf)
    kernel.set_arg(1, np.uint32(0))
    kernel.set_arg(2, nonce_buf)
    kernel.set_arg(3, target_buf)
    kernel.set_arg(4, winner_buf)
    kernel.set_arg(5, debug_data_buf)

    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    cl.enqueue_copy(queue, result_nonce, nonce_buf).wait()
    cl.enqueue_copy(queue, debug_data, debug_data_buf).wait()

    return result_nonce[0], debug_data

# Ejecutar kernel y encontrar nonce
header = b'\x0a\xbc'
target = np.array([0x00ffffffffffffffffffffffffffffffff], dtype=np.uint64)

nonce, debug_data = mining_GPU("kernel_mining", kernel_mining, cl.device_type.GPU, header, target, (1024,), (1,))

if nonce != -1:
    print(f"Nonce válido encontrado: {nonce}")
else:
    print("No se encontró un nonce válido. Revisa el kernel.")
