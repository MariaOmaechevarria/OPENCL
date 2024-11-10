import pyopencl as cl
import numpy as np
import os
import pandas as pd
import struct
import kernel_mining

kernel_mining ="""
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
                            ulong target, 
                            __global int* winner_id,  // Variable atómica para asegurar exclusividad
                            __global uint* mensaje_procesado, 
                            __global uchar* winning_hash,
                            ulong data_len) {

    uint nonce = start_nonce + get_global_id(0); // Cada hebra tiene un nonce único basado en su ID global
    uint global_id = get_global_id(0); // Obtener el ID global de la hebra
    __private uchar hash_buffer[32];
       // Verificar si ya se ha encontrado un nonce válido
    if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }
    const ulong max_len = 256; // Tamaño máximo adecuado para combined_data

    // Buffer de ayuda para almacenar los dígitos del nonce
    uchar nonce_buffer[10];
    int nonce_length = 0;
     if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }

    // Extraer los dígitos del nonce y almacenarlos en nonce_buffer
    uint copia_nonce = nonce;
    if (nonce == 0) {
        nonce_buffer[nonce_length++] = '0';
    }
    while (copia_nonce > 0) {
        nonce_buffer[nonce_length++] = (uchar)((copia_nonce % 10) + '0');
        copia_nonce /= 10;
    }

    // Invertir los dígitos en nonce_buffer
    for (int i = 0; i < nonce_length / 2; i++) {
        uchar temp = nonce_buffer[i];
        nonce_buffer[i] = nonce_buffer[nonce_length - 1 - i];
        nonce_buffer[nonce_length - 1 - i] = temp;
    }

    // Concatenar el data y el nonce en una cadena de bytes
    uchar combined_data_local[max_len];
    ulong new_len = data_len + nonce_length;

    // Copiar el data al comienzo de combined_data_local
    for (ulong i = 0; i < data_len; i++) {
        combined_data_local[i] = data[i];
    }

    // Añadir el nonce al final de combined_data_local
    for (int i = 0; i < nonce_length; i++) {
        combined_data_local[data_len + i] = nonce_buffer[i];
    }
       // Verificar si ya se ha encontrado un nonce válido
    if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }

    // Llamar a la función SHA256 con los datos combinados
    SHA256(combined_data_local, hash_buffer, new_len, mensaje_procesado);

     if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }
    // Convertir los primeros 16 bytes del hash a un valor entero
    ulong hash_value_high = 0;
    ulong hash_value_low = 0;
    for (int i = 0; i < 8; i++) {
        hash_value_high = (hash_value_high << 8) | hash_buffer[i];
    }
    for (int i = 8; i < 16; i++) {
        hash_value_low = (hash_value_low << 8) | hash_buffer[i];
    }
     if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }
    // Convertir el target a dos valores enteros
    ulong target_high = target >> 64;
    ulong target_low = target & 0xFFFFFFFFFFFFFFFF;
    if (*winner_id !=-1) {
        // Si ya se encontró un nonce, no hacer nada y terminar
        return;
    }
    // Comparar el hash con el target
    if (hash_value_high < target_high || (hash_value_high == target_high && hash_value_low < target_low)) {
        // Intentar establecer el ID del ganador
        if (atomic_cmpxchg(winner_id, -1, global_id) == -1) { // Solo el primer hilo que lo consigue se convierte en el ganador
            if (*winner_id == global_id){
            *new_nonce = nonce; // Guarda el nonce que cumplió la condición
            }
        }
    }
}


"""



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

