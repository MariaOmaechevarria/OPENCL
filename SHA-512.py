import pyopencl as cl
import struct
import numpy as np

def preprocesar(data):
    if isinstance(data, str):
        data = data.encode('utf-8')

    original_byte_len = len(data)
    original_bit_len = original_byte_len * 8

    data += b'\x80'
    
    while (len(data) * 8) % 1024 != 896:
        data += b'\x00'
    
    data += original_bit_len.to_bytes(16, 'big')
    
    return data

def bytes_a_hex(data, spaced=False):
    if spaced:
        return ' '.join(f'{byte:02x}' for byte in data)
    else:
        return ''.join(f'{byte:02x}' for byte in data)

kernel_SHA512 = """
ulong rotr(ulong x, int n) {
        return (x >> n) | (x << (64 - n));
    }

    ulong shr(ulong x, int n) {
     return x >> n;
    }


    ulong SumaA(ulong a) {
        return rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
    }

    ulong SumaE(ulong e) {
        return rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
    }

    ulong Sigma0(ulong x) {
        return rotr(x, 1) ^ rotr(x, 8) ^ shr(x, 7);
    }

    ulong Sigma1(ulong x) {
        return rotr(x, 19) ^ rotr(x, 61) ^ shr(x, 6);
    }   
    ulong Ch(ulong e, ulong f, ulong g) {
        return (e & f) ^ (~e & g);
    }

    ulong Maj(ulong a, ulong b, ulong c) {
        return (a & b) ^ (a & c) ^ (b & c);
    }

__kernel void kernel_SHA512(__global const ulong *message, 
                             __global ulong *hash, 
                             int num_blocks, __global ulong *bufferValores, __global ulong *K) {

    int block_id = get_global_id(0);
    
    
    for (int i = 0; i < num_blocks; i++) {
        ulong a, b, c, d, e, f, g, h;
        ulong W[80];

        for(int t = 0; t < 16; t++){
            W[t] = message[i * 16 + t];
        }

        for(int t = 16; t < 80; t++){
            W[t] = Sigma1(W[t-2]) + W[t-7] + Sigma0(W[t-15]) + W[t-16];
        }

        a = bufferValores[0];
        b = bufferValores[1];
        c = bufferValores[2];
        d = bufferValores[3];
        e = bufferValores[4];
        f = bufferValores[5];
        g = bufferValores[6];
        h = bufferValores[7];

        for(int t = 0; t < 80; t++){
            ulong T1 = h + SumaE(e) + Ch(e, f, g) + K[t] + W[t];
            ulong T2 = SumaA(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        bufferValores[0] += a;
        bufferValores[1] += b;
        bufferValores[2] += c;
        bufferValores[3] += d;
        bufferValores[4] += e;
        bufferValores[5] += f;
        bufferValores[6] += g;
        bufferValores[7] += h;
    }

    for (int k = 0; k < 8; k++) {
        hash[k] = bufferValores[k];
    }
}
"""
def dividir_en_bloques(hex_data, block_size):
    return [hex_data[i:i + block_size] for i in range(0, len(hex_data), block_size)]

def SHA512(mensaje, device_type, kernel_code, kernel_name):
   # Preprocesar el mensaje
    mensaje_hex = bytes_a_hex(preprocesar(mensaje))

     # Dividir la cadena hexadecimal en bloques de 16 caracteres
    bloques = dividir_en_bloques(mensaje_hex, 16)

    # Crear el array con representación hexadecimal de 64 bits
    mensaje_array = np.array([int(bloque, 16) for bloque in bloques], dtype=np.uint64)

    K = np.array([
        0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f,
        0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
        0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242,
        0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
        0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
        0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
        0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275,
        0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
        0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f,
        0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
        0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc,
        0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
        0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6,
        0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
        0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
        0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
        0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99,
        0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
        0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc,
        0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
        0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915,
        0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
        0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba,
        0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
        0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
        0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
        0x5fcb6fab3ad6faec, 0x6c44198c4a475817
    ], dtype=np.uint64)

    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=device_type)[0]
    context = cl.Context([device])
    command_queue = cl.CommandQueue(context, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, kernel_name)

    H = np.array([
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
    ], dtype=np.uint64)

    num_blocks = len(mensaje_array) // 16

    mensaje_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mensaje_array)
    hash_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=8 * np.dtype(np.uint64).itemsize)
    buffer_valores = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=H)
    buffer_valores_K = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=K)

    kernel.set_arg(0, mensaje_buffer)
    kernel.set_arg(1, hash_buffer)
    kernel.set_arg(2, np.int32(num_blocks))
    kernel.set_arg(3, buffer_valores)
    kernel.set_arg(4, buffer_valores_K)

    global_size = (1,)
    local_size = (1,)

    event = cl.enqueue_nd_range_kernel(command_queue, kernel, global_size, local_size)
    event.wait()

    hash_result = np.empty(8, dtype=np.uint64)
    cl.enqueue_copy(command_queue, hash_result, hash_buffer).wait()
    hash_concatenado = ''.join(f'{h:016x}' for h in hash_result)


    exec_time = 1e-9 * (event.profile.end - event.profile.start)

    return exec_time, hash_concatenado

mensaje = 'shhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh'
device_type = cl.device_type.GPU
kernel_code = kernel_SHA512
kernel_name = "kernel_SHA512"
tiempo, hash_result = SHA512(mensaje, device_type, kernel_code, kernel_name)
print(tiempo,hash_result)
