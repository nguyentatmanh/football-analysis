import os
import sys

def download():
    try:
        # pyrefly: ignore [missing-import]
        import gdown
    except ImportError:
        print("gdown is not installed or out of date. Installing/upgrading dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
        # pyrefly: ignore [missing-import]
        import gdown

    # Set paths
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT, "data", "models")
    RAW_DIR = os.path.join(ROOT, "data", "raw")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Download all models from Google Drive Folder
    folder_url = "https://drive.google.com/drive/folders/1dLzC5CNbHIkgVXt4DqPMNfHvTYu7tCMP?usp=sharing"
    print("="*60)
    print(f"📥 Downloading all models from Google Drive Folder to: {MODELS_DIR}")
    print("="*60)
    try:
        gdown.download_folder(url=folder_url, output=MODELS_DIR, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Error downloading folder via URL: {e}. Trying via folder ID...")
        try:
            gdown.download_folder(id="1dLzC5CNbHIkgVXt4DqPMNfHvTYu7tCMP", output=MODELS_DIR, quiet=False)
        except Exception as e_inner:
            print(f"Failed to download folder: {e_inner}")

    # 2. Download sample video
    sample_dest = os.path.join(RAW_DIR, "sample.mp4")
    sample_id = "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
    
    print("\n" + "="*60)
    print(f"📥 Downloading default sample video to: {sample_dest}")
    print("="*60)
    if os.path.exists(sample_dest):
        print(f"Skipping already existing: {os.path.basename(sample_dest)}")
    else:
        url = f"https://drive.google.com/uc?id={sample_id}"
        gdown.download(url, sample_dest, quiet=False)
        
    print("\nAll assets downloaded successfully!")

if __name__ == "__main__":
    download()
