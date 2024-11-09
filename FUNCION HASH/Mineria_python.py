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

import numpy as np

hash_result = hashlib.sha256((str("abc10") ).encode()).hexdigest()
target=np.uint64(0x00FFFFFFFFFFFFFFFF)
print( int(hash_result, 16) < target)
print(hash_result)

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
