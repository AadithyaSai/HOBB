from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_URL: str
    SMTP_SERVER: str
    SMTP_PORT: str
    SMTP_EMAIL: str
    SMTP_PASSWORD: str
    FRONTEND_URL: str
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()