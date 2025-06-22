# config.py

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'material_recognition_model.h5'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Image Parameters ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3 # RGB

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 15 # Increased epochs for potentially better training
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3 # Initial learning rate for Adam

# --- UI Parameters ---
APP_TITLE = "Material Recognition AI"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600