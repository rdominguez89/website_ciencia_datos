import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB file size limit
    ALLOWED_EXTENSIONS = {'csv'}