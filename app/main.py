from typing import BinaryIO

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


from app.cryptography import *

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/encrypt/")
async def encrypt_data(image: UploadFile, plaintext: str = Form()) -> dict[str, str]:
    # Encrypt the text and perform steganography on key
    encrypted_text, key = encrypt(plaintext)
    stego_image = steg_encode(image.file, key)

    return {"ciphertext":encrypted_text, "stego_image":stego_image}

@app.post("/decrypt/")
async def decrypt_data(stego_image: UploadFile, ciphertext: str = Form()) -> dict[str, str]:
    # Perform steganography on key and decrypt the text
    key = steg_decode(stego_image.file)
    decrypted_text = decrypt(ciphertext, key)

    return {"plaintext": decrypted_text}
