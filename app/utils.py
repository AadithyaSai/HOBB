import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")

def hash(password:str):
    return pwd_context.hash(password)

def verify(plain_pwd,hashed_pwd):
    return pwd_context.verify(plain_pwd,hashed_pwd)

def generate_reset_token():
    return secrets.token_urlsafe(32)

def token_expiry():
    return datetime.datetime.now(datetime.UTC) + timedelta(minutes=30)  # Token expires in 30 minutes

def send_reset_email(email: str, token: str):
    msg = EmailMessage()
    msg["Subject"] = "Password Reset Request"
    msg["From"] = "your_email@example.com"
    msg["To"] = email
    msg.set_content(f"Click the link to reset your password: http://localhost:8000/reset-password/{token}")

    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login("your_email@example.com", "your_email_password")
        server.send_message(msg)