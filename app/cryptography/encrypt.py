from .algorithms.AES import encrypt as AES_encrypt, decrypt as AES_decrypt
from .algorithms.Twofish import encrypt as Twofish_encrypt, decrypt as Twofish_decrypt
from .algorithms.Serpent import encrypt as Serpent_encrypt, decrypt as Serpent_decrypt
import base64
import os
from typing import Callable

ENCRYPTION_ALGORITHMS = [AES_encrypt, Twofish_encrypt, Serpent_encrypt]
DECRYPTION_ALGORITHMS = [AES_decrypt, Twofish_decrypt, Serpent_decrypt]

def divide_evenly(text: str) -> tuple[str, str, str]:
    n = len(text) // 3
    return (text[:n], text[n:2*n], text[2*n:])


def split_three_ways(text: str) -> tuple[str, str, str]:
    return (
        text[0::3],
        text[1::3],
        text[2::3]
    )

def join_three_ways(a: str, b: str, c: str) -> str:
    res = [''] * (len(a) + len(b) + len(c))
    res[0::3] = a
    res[1::3] = b
    res[2::3] = c
    return ''.join(res)

def encrypt_part(encrypt: Callable[[str, str], str], text: str) -> tuple[str, str]:
    key = os.urandom(16)
    
    # Encrypt the text
    ct_bytes = encrypt(key, text)
    
    # Combine IV and ciphertext, and encode as Base64
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    key_b64 = base64.b64encode(key).decode('utf-8')
    
    return (ct, key_b64)

def encrypt(text: str) -> tuple[str, str]:
    parts = split_three_ways(text)
    
    cipher = ''
    key = ''
    for i in range(3):
        ct, key_b64 = encrypt_part(ENCRYPTION_ALGORITHMS[i], parts[i])
        cipher += ct
        key += key_b64

    return (cipher, key)

def decrypt_part(algorithm:str, ct: str, key: str) -> str:
    ct = base64.b64decode(ct.encode('utf-8'))
    key = base64.b64decode(key.encode('utf-8'))

    pt_bytes = algorithm(key, ct)

    return pt_bytes.decode('utf-8')

def decrypt(text: str, key: str) -> str:
    cts = divide_evenly(text)
    keys = divide_evenly(key)

    pts = []
    for i in range(3):
        pts.append(decrypt_part(DECRYPTION_ALGORITHMS[i], cts[i], keys[i]))

    return join_three_ways(*pts)