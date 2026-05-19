import torch
import torch.nn as nn
import torchaudio
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Enhanced Configuration for Saraga Dataset
class SaragaConfig:
    # === PATHS ===
    BASE_PATH = "saraga_carnatic/"
    DATASET_PATH = "saraga_carnatic/dataset/"
    AUDIO_PATH = "saraga_carnatic/dataset/audio/"
    ANNOTATIONS_PATH = "saraga_carnatic/dataset/annotations/"
    MELODY_PATH = "saraga_carnatic/dataset/annotations/melody/"
    METADATA_PATH = "saraga_carnatic/dataset/metadata/"
    
    MODEL_PATH = "models/"
    OUTPUT_PATH = "output/"
    LOGS_PATH = "logs/"
    
    # === AUDIO PROCESSING ===
    SAMPLE_RATE = 44100  # Saraga uses 44.1kHz (high quality)
    TARGET_SR = 22050    # We'll downsample for processing efficiency
    HOP_LENGTH = 512     # ~23ms frames at 22050 Hz
    
    # === SARAGA SPECIFIC ===
    MELODY_HOP_SIZE = 0.0029  # Saraga's annotation hop size (2.9ms)
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for pitch
    
    # === MODEL PARAMETERS ===
    SEQUENCE_LENGTH = 200    # Longer sequences for better context
    HIDDEN_SIZE = 256        # Larger for better learning
    NUM_LAYERS = 3           # Deeper network
    DROPOUT = 0.4
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    
    # === CARNATIC MUSIC PARAMETERS ===
    # All 12 swaras (we'll filter by raga later)
    ALL_SWARAS = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
    
    # Bhairavi scale (komal re, ga, dha, ni)
    BHAIRAVI_SWARAS = ['S', 'r', 'g', 'M', 'P', 'd', 'n']
    
    # Frequency ranges for each swara (in semitones from Sa)
    SWARA_SEMITONES = {
        'S': 0,    # Sa
        'r': 1,    # Komal Re  
        'R': 2,    # Shuddha Re
        'g': 3,    # Komal Ga
        'G': 4,    # Shuddha Ga  
        'm': 5,    # Komal Ma
        'M': 6,    # Shuddha Ma
        'P': 7,    # Pa
        'd': 8,    # Komal Dha
        'D': 9,    # Shuddha Dha
        'n': 10,   # Komal Ni
        'N': 11    # Shuddha Ni
    }
    
    # === DEVICE CONFIGURATION ===
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === KAGGLE DATASET ===
    KAGGLE_DATASET = "desolationofsmaug/saraga-carnatic-music-dataset"

def setup_project_structure():
    """
    Creates all necessary folders for the project.
    Think of this as creating organized folders on your computer.
    """
    directories = [
        SaragaConfig.BASE_PATH,
        SaragaConfig.DATASET_PATH,
        SaragaConfig.AUDIO_PATH,
        SaragaConfig.ANNOTATIONS_PATH,
        SaragaConfig.MELODY_PATH,
        SaragaConfig.METADATA_PATH,
        SaragaConfig.MODEL_PATH,
        SaragaConfig.OUTPUT_PATH,
        SaragaConfig.LOGS_PATH
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def download_saraga_dataset():
    """
    Downloads the Saraga dataset from Kaggle.
    This is like downloading a ZIP file and extracting it.
    """
    try:
        import kaggle
        print("Downloading Saraga Carnatic Dataset from Kaggle...")
        print("This might take 10-20 minutes depending on your internet speed.")
        
        # Download and extract
        kaggle.api.dataset_download_files(
            SaragaConfig.KAGGLE_DATASET, 
            path=SaragaConfig.BASE_PATH, 
            unzip=True
        )
        print(" Download completed!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get API key from https://www.kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ folder")
        print("4. Run this script again")
        return False

def check_system_requirements():
    """
    Checks if your computer can run this project.
    Like checking if you have enough space and the right programs.
    """
    print("🔍 Checking System Requirements...")
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python Version: {python_version}")
    
    if sys.version_info.major < 3 or sys.version_info.minor < 7:
        print(" Warning: Python 3.7+ recommended")
    else:
        print("Python version OK")
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print("GPU acceleration available!")
    else:
        print("Using CPU (will be slower but works fine)")
    
    # Check disk space (rough estimate)
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    print(f"Available Disk Space: {free_space_gb:.1f} GB")
    
    if free_space_gb < 5:
        print("Warning: Low disk space. Saraga dataset needs ~3-4 GB")
    else:
        print("Sufficient disk space")
    
    return True

if __name__ == "__main__":
    print("🎵 CARNATIC MUSIC TRANSCRIPTION WITH SARAGA DATASET")
    print("=" * 60)
    
    # Check if system is ready
    check_system_requirements()
    print()
    
    # Setup project structure
    setup_project_structure()
    print()
    
    # Download dataset
    print("Ready to download Saraga dataset!")
    print("This contains:")
    print("  • 196+ Carnatic vocal recordings")
    print("  • Time-aligned melody annotations") 
    print("  • Multiple ragas including Bhairavi")
    print("  • Professional-grade annotations")
    
    download_choice = input("\nDownload now? (y/n): ").lower()
    if download_choice == 'y':
        success = download_saraga_dataset()
        if success:
            print("\nSetup Complete! Ready for Step 2.")
        else:
            print("\n Please set up Kaggle API and try again.")
    else:
        print("\n Setup project folders completed.")
        print("Run download_saraga_dataset() when ready.")
    
    print(f"\n Configuration Summary:")
    print(f"Target Raga: Bhairavi")
    print(f"Device: {SaragaConfig.DEVICE}")
    print(f"Audio Sample Rate: {SaragaConfig.TARGET_SR} Hz")
    print(f"Model: Bi-LSTM + TCN")
