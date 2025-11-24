import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
