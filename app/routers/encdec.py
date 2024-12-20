from fastapi import FastAPI, Request, UploadFile, Form, Response, status,HTTPException, Depends,APIRouter
from fastapi.responses import HTMLResponse
from fastapi.params import Body
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .. import models,schemas,utils
from sqlalchemy.orm import Session
from ..database import engine,SessionLocal, get_db
from app.cryptography import *

templates = Jinja2Templates(directory="app/templates")
router = APIRouter(tags=["Encryption/Decryption"])


@router.post("/encrypt/")
async def encrypt_data(image: UploadFile, plaintext: str = Form()) -> dict[str, str]:
    # Encrypt the text and perform steganography on key
    encrypted_text, key = encrypt(plaintext)
    stego_image = steg_encode(image.file, key)

    return {"ciphertext":encrypted_text, "stego_image":stego_image}

@router.post("/decrypt/")
async def decrypt_data(stego_image: UploadFile, ciphertext: str = Form()) -> dict[str, str]:
    # Perform steganography on key and decrypt the text
    key = steg_decode(stego_image.file)
    decrypted_text = decrypt(ciphertext, key)

    return {"plaintext": decrypted_text}

