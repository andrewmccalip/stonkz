"""Configuration settings for the AI Stock Prediction System"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    # OpenRouter API
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # Hugging Face API
    HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN', '')
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Data settings
    DATA_PATH = 'databento/ES/'
    DEFAULT_DATA_FILE = 'glbx-mdp3-20100606-20250822.ohlcv-1m.csv'
    
    # Market hours (Eastern Time)
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    EXTENDED_HOURS_START = "04:00"
    EXTENDED_HOURS_END = "20:00"
    
    # Technical indicator parameters
    RSI_PERIOD = 14
    SMA_PERIODS = [20, 50]
    EMA_PERIODS = [12, 26]
    BB_PERIOD = 20
    BB_STD = 2
    STOCH_PERIODS = (14, 3, 3)
    
    # AI Models to use
    AI_MODELS = [
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-opus",
        "google/gemini-pro",
        "mistralai/mistral-large",
        "cohere/command-r-plus"
    ]
    
    # TimesFM Model Configuration
    TIMESFM_CONTEXT_LENGTH = 512      # Context length for TimesFM model
    TIMESFM_HORIZON_LENGTH = 128      # Maximum prediction horizon
    TIMESFM_MODEL_REPO = "pfnet/timesfm-1.0-200m-fin"  # Financial fine-tuned model
    TIMESFM_BACKEND = "cpu"           # Force CPU backend for local inference
    
    # Prediction settings
    PREDICTION_INTERVAL_MINUTES = 1
    MAX_PREDICTION_HOURS = 8  # Maximum hours into the future to predict
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_TIMEOUT_SECONDS = 300  # 5 minutes

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Override with production settings
    
# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
