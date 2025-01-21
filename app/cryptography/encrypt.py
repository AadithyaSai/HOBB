from .algorithms.AES import encrypt as AES_encrypt, decrypt as AES_decrypt
from .algorithms.Twofish import encrypt as Twofish_encrypt, decrypt as Twofish_decrypt
from .algorithms.Serpent import encrypt as Serpent_encrypt, decrypt as Serpent_decrypt
import base64
import os
from typing import Callable

ENCRYPTION_ALGORITHMS = [AES_encrypt, Twofish_encrypt, Serpent_encrypt]

def split_three_ways(text: str) -> tuple[str, str, str]:
    return (
        text[0::3],
        text[1::3],
        text[2::3]
    )

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
        ct, key_b64 = encrypt_part(ENCRYPTION_ALGORITHMS[i].encrypt, parts[i])
        cipher += ct
        key += key_b64

    return (cipher, key)

def decrypt_part(algorithm:str, ct: str, key: str) -> str:
    ct = base64.b64decode(ct.encode('utf-8'))
    key = base64.b64decode(key.encode('utf-8'))

    iv = ct[:16]  # First 16 bytes are the IV
    ct_bytes = ct[16:]  # Remaining bytes are the ciphertext

    # Decrypt the text
    cipher = botan3.SymmetricCipher(algorithm, encrypt=False)
    cipher.set_key(key)
    cipher.start(iv)
    pt_bytes = cipher.finish(ct_bytes)

    return pt_bytes.decode('utf-8')

def decrypt(text: str, key: str) -> str:
    cts = split_three_ways(text)
    keys = split_three_ways(key)

    plaintext = ''
    for i in range(3):
        plaintext += decrypt_part(ENCRYPTION_ALGORITHMS[i], cts[i], keys[i])

    return plaintext