# main.py

import tkinter as tk
from app_ui import MaterialRecognitionApp
import os
from config import MODELS_DIR

def create_initial_dirs():
    """Ensures necessary directories exist at startup."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    # The data directory is handled by data_loader, but a minimal dummy setup
    # for first run can be helpful if it's completely empty.
    # For robust app: instruct user to prepare data/


if __name__ == "__main__":
    create_initial_dirs()
    root = tk.Tk()
    app = MaterialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Ensure clean shutdown
    root.mainloop()