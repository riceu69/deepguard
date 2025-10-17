"""
Download model from Google Drive on startup
"""
import os
import gdown

def download_model():
    """Download model if not present"""
    model_dir = './models_v2'
    model_path = os.path.join(model_dir, 'best_model.pth')
    
    if os.path.exists(model_path):
        print("✅ Model already exists")
        return model_path
    
    print("📥 Downloading model from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)
    
    # YOUR GOOGLE DRIVE FILE ID HERE
    file_id = "1ztKysDs6gyuk-TiYiQMXhG6I3Ln8JAzX"  # ← REPLACE THIS
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        gdown.download(url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
        return model_path
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("⚠️  App will use pretrained backbones only")
        return None

if __name__ == "__main__":
    download_model()
