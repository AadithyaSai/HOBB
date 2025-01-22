from jose import JWTError,jwt
from datetime import datetime,timedelta,timezone
from . import schemas,database
from fastapi import Depends,status,HTTPException
from fastapi.security import OAuth2PasswordBearer
#secret key, algorithm to use(sha256), expiration time of token(user cant be logged in forever)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='login')  #tokenurl is the login endpoint..here it is /login

SECRET_KEY = "09d25e094fdikshf556c818166b7a9563b93faidjslf0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data:dict):
    to_encode = data.copy()
    
    expire = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp" : expire})

    encoded_jwt = jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM) #creates token
    return encoded_jwt

def verify_access_token(token:str,credentials_exception):
    try:
        payload = jwt.decode(token,SECRET_KEY,[ALGORITHM])
        id:str = payload.get("user_id")

        if id is None:
           raise credentials_exception
        token_data = schemas.TokenData(id=id)

    except JWTError:
        raise credentials_exception
    
    return token_data
    
#(below fn)pass as dependency to path opn, thus taking token from reqst automatically,extract id for us, verify token validity 
def get_current_user(token:str=Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail=f"Could not validate the credentials",headers={"WWW-Authenticate":"Bearer"})

    return verify_access_token(token,credentials_exception)

#anytime we have an endpoint that needs to be protected (user needs to be authorized to access it), we add the above fn as a dependency, so token needs to be provided to access that endpoint
#68 01:35
