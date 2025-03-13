from .. import models,schemas,utils
from fastapi import FastAPI, Request, Response, status,HTTPException, Depends,APIRouter
from sqlalchemy.orm import Session
from ..database import engine,SessionLocal, get_db
from passlib.context import CryptContext
from datetime import datetime, UTC

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/users",tags=["User"])


@router.post("/",status_code=status.HTTP_201_CREATED,response_model = schemas.UserOut)
def create_user(user:schemas.Usercreate,db:Session=Depends(get_db)):
    hashed_password = utils.hash(user.password)
    user.password = hashed_password

    new_user=models.User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.get("/{id}",response_model = schemas.UserOut)
def get_user(id:int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == id).first()

    if not user:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND,detail = f"User with id {id} not found")

    return user

@router.post("/forgot-password", response_model=schemas.Message)
def forgot_password(email: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND,detail = f"User with email {email} not found")
    
    # Generate new token and expiry
    reset_token = utils.generate_reset_token()
    expiry = utils.token_expiry()

    # Remove any existing reset token for the user
    db.query(models.PasswordResetToken).filter(models.PasswordResetToken.user_id == user.id).delete()

    # Create new token record
    reset_entry = models.PasswordResetToken(user_id=user.id, token=reset_token, expires_at=expiry)
    db.add(reset_entry)
    db.commit()

    utils.send_reset_email(user.email, reset_token)
    return {"message": "Password reset email sent"}


@router.post("/reset-password", response_model=schemas.Message)
def reset_password(data: schemas.ResetPasswordSchema, db: Session = Depends(get_db)):
    token_entry = db.query(models.PasswordResetToken).filter(models.PasswordResetToken.token == data.token).first()

    if not token_entry or token_entry.expires_at < datetime.now(UTC).replace(tzinfo=None):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    
    # Update user password
    user = db.query(models.User).filter(models.User.id == token_entry.user_id).first()
    user.password_hash = pwd_context.hash(data.new_password)
    
    # Remove used token
    db.delete(token_entry)
    db.commit()

    return {"message": "Password reset successful"}