import os

MAX = 1000000

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') 
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB file size limit
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Babel configuration
    BABEL_DEFAULT_LOCALE = 'en'
    BABEL_SUPPORTED_LOCALES = ['en', 'es']
    BABEL_TRANSLATION_DIRECTORIES = 'translations'