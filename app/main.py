from typing import BinaryIO,Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Request, UploadFile, Form, Response, status,HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.params import Body
from pydantic import BaseModel
from random import randrange
import time
from . import models,schemas,utils
from sqlalchemy.orm import Session
from .database import engine,SessionLocal, get_db
from app.cryptography import *
from .routers import encdec,user
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")
 
app.include_router(encdec.router)
app.include_router(user.router)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

while True:
    try:
        conn=psycopg2.connect(host='localhost',database='cryptobase',user='postgres',password='postgres',cursor_factory=RealDictCursor)
        cursor=conn.cursor()
        print("Connected to DB")
        break
    except Exception as error:
        print("Connection to db failed")
        print("Error : ",error)
        time.sleep(3)

@app.get("/sqlalchemy")
def testposts(db:Session = Depends(get_db)):
    posts=db.query(models.Post).all()
    return {"status":posts}

