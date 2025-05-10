#!/usr/bin/env python3
"""
Main script to run the AI Mental Health Journal project.
This script handles:
1. Downloading the GoEmotions dataset
2. Training the emotion detection model
3. Running the Streamlit app
"""

import os
import argparse
import subprocess
from dotenv import load_dotenv

def setup_environment():
    """Set up the project environment"""
    # Create necessary directories
    os.makedirs('models/emotion_detector', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No OpenAI API key found. Using fallback responses.")
    else:
        print("OpenAI API key found. Using enhanced responses.")

def download_dataset():
    """Download and prepare the GoEmotions dataset"""
    if os.path.exists('data/goemotions.csv'):
        print("Dataset already exists. Skipping download.")
        return
    
    print("Downloading GoEmotions dataset...")
    subprocess.run(["python", "download_dataset.py"], check=True)
    print("Dataset downloaded successfully.")

def train_model():
    """Train the emotion detection model"""
    if os.path.exists('models/emotion_detector/pytorch_model.bin'):
        print("Model already exists. Skipping training.")
        return
    
    print("Training emotion detection model...")
    subprocess.run([
        "python", "-m", "emotion_detector.train",
        "--data_path", "data/goemotions.csv",
        "--output_dir", "models/emotion_detector",
        "--epochs", "3"
    ], check=True)
    print("Model trained successfully.")

def run_app():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    subprocess.run(["streamlit", "run", "app.py"])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Mental Health Journal")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--only-app", action="store_true", help="Only run the app without setup")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.only_app:
        run_app()
    else:
        setup_environment()
        
        if not args.skip_download:
            download_dataset()
        
        if not args.skip_training:
            train_model()
        
        run_app()