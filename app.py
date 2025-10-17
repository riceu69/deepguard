from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
from datetime import datetime
from pathlib import Path
from detector import DeepfakeDetector, FaceMatchingModel
from PIL import Image
import sys
from pathlib import Path

# Download model before loading detector
print("="*60)
print("🚀 INITIALIZING DEEPGUARD")
print("="*60)

# Import download function
from download_model import download_model

# Download model (only happens once)
model_path = download_model()

print("="*60)
print("✅ INITIALIZATION COMPLETE")
print("="*60)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="DeepGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

print("Loading models...")
deepfake_detector = DeepfakeDetector(model_path='./models_v2/best_model.pth')
face_matcher = FaceMatchingModel()
print("✅ Models ready!")

UPLOADS_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Main page - video analysis"""
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/verify")
async def verify_page():
    """Identity verification page"""
    return FileResponse(str(STATIC_DIR / "verify.html"))

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze video for deepfakes"""
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOADS_DIR / filename
        
        with open(str(filepath), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📹 Analyzing video: {filename}")
        
        result = deepfake_detector.analyze_video(str(filepath))
        result['filename'] = filename
        result['upload_time'] = timestamp
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/verify-identity")
async def verify_identity(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    """Verify identity - match ID with selfie and detect deepfakes"""
    try:
        if not id_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="ID must be an image")
        if not selfie_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Selfie must be an image")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save files
        id_filename = f"{timestamp}_id_{id_image.filename}"
        id_filepath = UPLOADS_DIR / id_filename
        with open(str(id_filepath), "wb") as buffer:
            shutil.copyfileobj(id_image.file, buffer)
        
        selfie_filename = f"{timestamp}_selfie_{selfie_image.filename}"
        selfie_filepath = UPLOADS_DIR / selfie_filename
        with open(str(selfie_filepath), "wb") as buffer:
            shutil.copyfileobj(selfie_image.file, buffer)
        
        print(f"🔍 Verifying: {id_filename} vs {selfie_filename}")
        
        # Load images
        id_img = Image.open(id_filepath).convert('RGB')
        selfie_img = Image.open(selfie_filepath).convert('RGB')
        
        # 1. Check if same person
        is_match, match_confidence = face_matcher.predict_with_confidence(id_img, selfie_img)
        
        # 2. Check if selfie is deepfake
        is_fake, fake_confidence = deepfake_detector.predict_image(selfie_img)
        
        # Overall decision
        verification_passed = is_match and not is_fake
        
        result = {
            'verification_passed': verification_passed,
            'identity_match': {
                'is_match': bool(is_match),
                'confidence': float(match_confidence),
                'status': 'MATCH' if is_match else 'NO MATCH'
            },
            'deepfake_detection': {
                'is_fake': bool(is_fake),
                'confidence': float(fake_confidence),
                'status': 'FAKE' if is_fake else 'AUTHENTIC'
            },
            'overall_status': 'VERIFIED ✅' if verification_passed else 'REJECTED ❌',
            'rejection_reasons': []
        }
        
        if not is_match:
            result['rejection_reasons'].append('Identity does not match')
        if is_fake:
            result['rejection_reasons'].append('Deepfake detected in selfie')
        
        print(f"✅ Result: {result['overall_status']}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze single image for deepfakes"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOADS_DIR / filename
        
        with open(str(filepath), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📸 Analyzing image: {filename}")
        
        # Load image
        img = Image.open(filepath).convert('RGB')
        
        # Analyze for deepfake
        is_fake, fake_confidence = deepfake_detector.predict_image(img)
        
        result = {
            'is_fake': bool(is_fake),
            'is_deepfake': bool(is_fake),  # Alias for compatibility
            'confidence': float(fake_confidence),
            'fake_confidence': float(fake_confidence),
            'faces_detected': 1,
            'filename': filename,
            'upload_time': timestamp,
            'analysis_type': 'image'
        }
        
        print(f"✅ Result: {'FAKE' if is_fake else 'REAL'} ({fake_confidence:.1f}%)")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "deepfake_detector": "loaded",
        "face_matcher": "loaded",
        "gpu": deepfake_detector.device.type
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 DeepGuard Server Starting")
    print("="*50)
    print("📹 Video Analysis: http://localhost:8000")
    print("🆔 Identity Verify: http://localhost:8000/verify")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)