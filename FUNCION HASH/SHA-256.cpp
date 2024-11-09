#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

const std::array<uint32_t, 8> H = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372,
    0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19
};

const std::array<uint32_t, 64> k = {
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

std::vector<uint8_t> preprocesar(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> data_p = data;

    size_t original_byte_len = data_p.size();
    size_t original_bit_len = original_byte_len * 8;

    // Agregar el bit '1'
    data_p.push_back(0x80);

    // Agregar ceros hasta que la longitud total sea 448 mod 512
    while ((data_p.size() * 8) % 512 != 448) {
        data_p.push_back(0x00);
    }

    // Agregar la longitud original del mensaje como un entero de 64 bits
    for (int i = 7; i >= 0; --i) {
        data_p.push_back((original_bit_len >> (i * 8)) & 0xFF);
    }

    return data_p;
}

std::string bytes_a_hex(const std::vector<uint8_t>& data) {
    std::ostringstream oss;
    for (uint8_t byte : data) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    return oss.str();
}

uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

uint32_t S0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint32_t S1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

void SHA256(const std::vector<uint8_t>& mensaje) {
    std::vector<uint8_t> mensaje_procesado = preprocesar(mensaje);
    size_t num_blocks = mensaje_procesado.size() / 64;

    std::array<uint32_t, 8> valores_intermedios = H;

    for (size_t i = 0; i < num_blocks; ++i) {
        std::array<uint32_t, 64> W;
        for (size_t t = 0; t < 16; ++t) {
            W[t] = (mensaje_procesado[i * 64 + t * 4] << 24) |
                   (mensaje_procesado[i * 64 + t * 4 + 1] << 16) |
                   (mensaje_procesado[i * 64 + t * 4 + 2] << 8) |
                   (mensaje_procesado[i * 64 + t * 4 + 3]);
        }

        for (size_t t = 16; t < 64; ++t) {
            W[t] = S1(W[t - 2]) + W[t - 7] + S0(W[t - 15]) + W[t - 16];
        }

        uint32_t a = valores_intermedios[0];
        uint32_t b = valores_intermedios[1];
        uint32_t c = valores_intermedios[2];
        uint32_t d = valores_intermedios[3];
        uint32_t e = valores_intermedios[4];
        uint32_t f = valores_intermedios[5];
        uint32_t g = valores_intermedios[6];
        uint32_t h = valores_intermedios[7];

        for (size_t t = 0; t < 64; ++t) {
            uint32_t T1 = h + S1(e) + Ch(e, f, g) + k[t] + W[t];
            uint32_t T2 = S0(a) + Maj(a, b, c);

            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        valores_intermedios[0] += a;
        valores_intermedios[1] += b;
        valores_intermedios[2] += c;
        valores_intermedios[3] += d;
        valores_intermedios[4] += e;
        valores_intermedios[5] += f;
        valores_intermedios[6] += g;
        valores_intermedios[7] += h;
    }

    // Imprimir el hash final
    for (const auto& v : valores_intermedios) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << v;
    }
    std::cout << std::endl;
}




