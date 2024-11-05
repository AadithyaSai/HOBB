import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encrypt(text: str) -> tuple[str, str]:
    # Generate a random key
    key = get_random_bytes(16)

    # Encrypt the text
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(text.encode('utf-8'), AES.block_size))
    ct = base64.b64encode(cipher.iv + ct_bytes).decode('utf-8')

    return (ct, base64.b64encode(key).decode('utf-8'))

def decrypt(text: str, key: str) -> str:
    # Decode the text
    ct = base64.b64decode(text.encode('utf-8'))

    # Decrypt the text
    iv = ct[:AES.block_size]
    cipher = AES.new(base64.b64decode(key), AES.MODE_CBC, iv=iv)
    pt = unpad(cipher.decrypt(ct[AES.block_size:]), AES.block_size)

    return pt.decode('utf-8')