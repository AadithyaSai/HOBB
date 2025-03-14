from pydantic import BaseModel,EmailStr
from datetime import datetime
from typing import Optional

class Usercreate(BaseModel):
    email : EmailStr
    password : str

class UserOut(BaseModel):
    id:int
    email:EmailStr
    created_at : datetime

    class Config:
        orm_mode  = True

class UserLogin(BaseModel):
    email: EmailStr
    password : str

class Token(BaseModel):
    access_token:str
    token_type:str

class TokenData(BaseModel):
    id:Optional[str] = None

class ResetPasswordSchema(BaseModel):
    token: str
    new_password: str

class Message(BaseModel):
    message:str