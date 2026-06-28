#!/usr/bin/env python3


# did r2 -A re_challenge
# pdf @main
# then from there pdf @ fcn.00001380
# brute forced after exit
# here we are skull emoji
def hash_input(input_str):

    hash_val = 0x12345678
    
    if input_str:
        counter = 0
        for char in input_str:
            hash_val = ((hash_val << 5) + hash_val) ^ (ord(char) + counter * 7)
            hash_val &= 0xFFFFFFFF
            counter += 1
    
    output_bytes = []
    for shift in range(16):
        hash_val = (hash_val * 0x19660d + 0x3c6ef35f) & 0xFFFFFFFF
        byte_val = (hash_val >> shift) & 0xFF
        output_bytes.append(byte_val)
    
    return bytes(output_bytes)

target = bytes([0xa0, 0xbf, 0x34, 0xc1, 0xfd, 0x49, 0xf0, 0xca, 
                0xc6, 0x5a, 0x7b, 0x05, 0x99, 0x11, 0xf5, 0x46])

import string
import itertools

candidates = [
    "ND_3r", # n word
    "mgUa", # uhhh
    "ND_3rmgUa",
    "mgUaND_3r",
    "password", # lol
    "admin",
    "root",
    "flag",
    "secret",
    "key",
    "pass",
    "re_challenge",
    "goodjob",
    "goodjob!", # thought maybe you were sneaky
]

for candidate in candidates:
    result = hash_input(candidate)
    if result == target:
        print(f"'{candidate}'")
        print(f"{result.hex()}")
        exit(0)

charset = string.ascii_lowercase + string.digits + "_!@#"

for length in range(1, 10):
    print(f"Trying length {length}...")
    for attempt in itertools.product(charset, repeat=length):
        password = ''.join(attempt)
        result = hash_input(password)
        
        if result == target:
            print(f"'{password}'")
            print(f"{result.hex()}")
            exit(0)
        
        if hash(password) % 100000 == 0:
            print(f"  Tested: {password}", end='\r')
