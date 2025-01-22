from fastapi import APIRouter, Depends, status, HTTPException,Response
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .. import database,schemas,models,utils,oauth2

router = APIRouter(tags=['Authentication'])

#since we use OAuth2PasswordRequestForm here below, if we test in postman we have to send the data not as raw, but as form-data
@router.post('/login',response_model=schemas.Token)
def login(user_credentials:OAuth2PasswordRequestForm = Depends() ,db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == user_credentials.username).first()  #username is essentially the email, its just called username cause we used OAuth2PasswordRequestForm
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail=f"Invalid Credentials")
    
    if not utils.verify(user_credentials.password,user.password):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail=f"Invalid Credentials")
    
    #create token jwt and return
    access_token = oauth2.create_access_token(data={"user_id":user.id})
    return {"access_token":access_token,"token_type":"bearer"}
