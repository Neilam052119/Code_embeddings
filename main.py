import time
from datetime import datetime
script_start_time = datetime.now()


import subprocess
import os
# import pandas as pd
# from datetime import datetime
# from PIL import Image
# from paddleocr import PaddleOCR, draw_ocr
# from transformers import AutoImageProcessor, TableTransformerForObjectDetection
# import matplotlib.pyplot as plt
# import numpy as np
import torch
import fitz
import re
print("PyMuPDF is successfully imported")

# Step 1: Execute crop.py
print("Executing crop.py...")
try:
    # Get the path to the Python interpreter in the virtual environment
    venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')
    subprocess.run([venv_python, "crop.py"], check=True)
    print("crop.py execution completed.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing crop.py: {e}")

# Step 2: Execute keywords.py
print("Executing keywords.py...")
try:
    # Get the path to the Python interpreter in the virtual environment
    venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')
    subprocess.run([venv_python, "keywords.py"], check=True)
    print("keywords.py execution completed.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing keywords.py: {e}")

# # Define paths
# main_folder = r"C:\Users\neela\OneDrive\Desktop\Code_embeddings\Code_merge\Delhi"
# model_weight_path = r"C:\Users\neela\OneDrive\Desktop\Code_embeddings\Code_merge\last (1).pt"

