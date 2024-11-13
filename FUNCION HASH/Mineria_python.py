#!/usr/bin/env python3  # Cambia a python3 si es necesario
import hashlib
import time

max_nonce = 2 ** 32  # 4 billion

def proof_of_work(header, difficulty_bits):
    # Calculate the difficulty target
    target = 2 ** (256 - difficulty_bits)
    for nonce in range(max_nonce):
        hash_result = hashlib.sha256((str(header) + str(nonce)).encode()).hexdigest()
        # Check if this is a valid result, below the target
        if int(hash_result, 16) < target:
            print("Success with nonce %d" % nonce)
            print("Hash is %s" % hash_result)
            return (hash_result, nonce)
    print("Failed after %d tries" % nonce)
    return nonce


  # Simulación de campos del encabezado
version = 2  # Versión del bloque (4 bytes)
prev_block_hash = "0000000000000000000babae9ed8f7bb2b8d3f9f97bba97b8b8b8b8b8b8b8b8b"
merkle_root = "4d5f5c9ac7ed8f96b0e8a6b3b1c1a3e5f7d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
timestamp = 1633046400
bits = 0x17148edf
nonce = 2083236893

    # Construcción del encabezado en binario
header = (
        str(version).encode('utf-8') +
        bytes.fromhex(prev_block_hash) +
        bytes.fromhex(merkle_root) +
        timestamp.to_bytes(4, 'little') +
        bits.to_bytes(4, 'little') +
        nonce.to_bytes(4, 'little')
    )

hash_result = hashlib.sha256((str(header) + str(2083236893)).encode()).hexdigest()
print(hash_result)
print(int(hash_result, 16))
target= 0x00000000000000000000000000000000000000000000000000000000000000FF
print(target)
print(int(hash_result, 16) < target)

'''


if __name__ == '__main__':
    nonce = 0
    hash_result = ''
    # Difficulty from 0 to 31 bits
    for difficulty_bits in range(32):
        difficulty = 2 ** difficulty_bits
        print("Difficulty: %d (%d bits)" % (difficulty, difficulty_bits))
        print("Starting search...")
        # Checkpoint the current time
        start_time = time.time()
        # We fake a block of transactions - just a string
        new_block = 'test block with transactions' + hash_result
        # Find a valid nonce for the new block
        (hash_result, nonce) = proof_of_work(new_block, difficulty_bits)
        # Checkpoint how long it took to find a result
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time: %.4f seconds" % elapsed_time)
        if elapsed_time > 0:  # Estimate the hashes per second
            hash_power = float(nonce) / elapsed_time
            print("Hashing Power: %d hashes per second" % hash_power)
'''