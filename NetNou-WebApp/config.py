"""Application configuration settings."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Base configuration for the application."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Database Configuration
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'sqlite:///' + str(BASE_DIR / 'instance' / 'app.db')
    
    # Camera Configuration
    CAMERA_INDEX = 0
    
    # Face Recognition Configuration
    FACE_DETECTION_BACKEND = 'opencv'  # Options: opencv, ssd, mtcnn, retinaface, mediapipe
    ANALYZE_EVERY_N_FRAMES = 20
    FACE_RESIZE_FACTOR = 0.3
    FACE_ANALYZE_ACTIONS = ['emotion', 'age', 'gender']
    
    # Paths to models
    NN_WEIGHTS_PATH = str(BASE_DIR / 'models' / 'engagement_nn_weights.npz')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URI = os.environ.get('DATABASE_URI')
    
    # More secure settings
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True 