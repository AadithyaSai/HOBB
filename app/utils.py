import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, UTC
from passlib.context import CryptContext
from .settings import settings

pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")

def hash(password:str):
    return pwd_context.hash(password)

def verify(plain_pwd,hashed_pwd):
    return pwd_context.verify(plain_pwd,hashed_pwd)

def generate_reset_token():
    return secrets.token_urlsafe(32)

def token_expiry():
    return datetime.now(UTC) + timedelta(minutes=30)  # Token expires in 30 minutes

def send_reset_email(email: str, token: str):
    msg = EmailMessage()
    msg["Subject"] = "Password Reset Request"
    msg["From"] = settings.SMTP_EMAIL
    msg["To"] = email
    msg.set_content(f"Click the link to reset your password: {settings.FRONTEND_URL}/auth?reset=true&email={email}&token={token}")

    with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:  # FIXME
        server.starttls()
        server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
        server.send_message(msg)