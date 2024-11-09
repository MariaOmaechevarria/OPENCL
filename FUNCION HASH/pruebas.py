import ctypes
kernel_check_hash="""
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

void preprocesar(__global char* data, __global uint* data_p, ulong original_byte_len) {
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

void SHA256(__global char* mensaje, __global uchar* hash, ulong original_byte_len, __global uint* mensaje_procesado) {
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

__kernel void kernel_check_hash(__global char* data, __global uchar* hash_buffer, __global uint* mensaje_procesado, ulong data_len) {
    SHA256(data, hash_buffer, data_len, mensaje_procesado);
}

"""
import pyopencl as cl
import numpy as np

def check_hash_GPU(kernel_name, kernel_code, device_type, data):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
   
    # Crear el kernel

    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)
   
    # Tamaño global para una única ejecución
    global_size = (1,)
    local_size = None
   
    # Tamaño del data a ser procesado
    DATA_SIZE = len(data)
    data_size_np = np.array([DATA_SIZE], dtype=np.uint64)

    # Creación de buffers necesarios
    data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=32)
    mensaje_procesado = np.zeros(64, dtype=np.uint32)
    mensaje_procesado_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=mensaje_procesado)
    data_size_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_size_np)

    # Establecer argumentos kernel
    kernel.set_arg(0, data_buffer)
    kernel.set_arg(1, hash_buffer)
    kernel.set_arg(2, mensaje_procesado_buffer)
    kernel.set_arg(3, data_size_buffer)

    # Ejecutar kernel
    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()
   
    # Copiar valor del hash final
    hash = np.zeros(32, dtype=np.uint8)
    cl.enqueue_copy(command_queue, hash, hash_buffer).wait()
   
    # Obtener tiempos
    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return exec_time, hash

# Datos de prueba
data = "abc"
data_bytes = data.encode('utf-8')

# Ejecutar kernel
kernel_name = "kernel_check_hash"
device_type = cl.device_type.GPU

# Llamada a la función de minería
exec_time, hash = check_hash_GPU(kernel_name, kernel_check_hash, device_type, data_bytes)

# Convertir resultado hash a hexadecimal
hexadecimal = ''.join(f'{num:02X}' for num in hash)

print(f"Execution time: {exec_time} s")
print(f"Hash: {hexadecimal}")