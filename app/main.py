from fastapi import FastAPI

from app.cryptography import *

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("encrypt")
def encrypt(plaintext: str, image: str) -> dict:
    # Encrypt the text and perform steganography on key
    encrypted_text, key = encrypt(plaintext)
    stego_image = steg_encode(image, key)

    return {"ciphertext": encrypted_text, "stego_image": stego_image}

@app.get("decrypt")
def decrypt(ciphertext: str, stego_image: str) -> str:
    # Perform steganography on key and decrypt the text
    key = steg_decode(stego_image)
    decrypted_text = decrypt(ciphertext, key)

    return decrypted_text
