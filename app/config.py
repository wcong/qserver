"""
Flask application configuration
"""
import os


class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Qlib configuration
    QLIB_DATA_PATH = os.environ.get('QLIB_DATA_PATH', '/app/data/stock/cn_data')
    
    # Training configuration
    MODEL_OUTPUT_PATH = os.environ.get('MODEL_OUTPUT_PATH', '/app/data/models')
    TRAIN_OUTPUT_PATH = os.environ.get('TRAIN_OUTPUT_PATH', '/app/data/train')
