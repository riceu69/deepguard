import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import os
import timm
from facenet_pytorch import InceptionResnetV1

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TwoModelEnsemble(nn.Module):
    """Dual-model ensemble: EfficientNet + FaceNet"""
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        
        # EfficientNet for visual features
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # FaceNet for facial features
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(1792, 1024),  # 1280 (EfficientNet) + 512 (FaceNet)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(512, num_classes)
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features from both models
        eff_feat = self.efficientnet(x)
        face_feat = self.facenet(x)
        
        # Concatenate features
        combined = torch.cat([eff_feat, face_feat], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Classification and confidence
        logits = self.classifier(fused)
        confidence = self.confidence_head(fused)
        
        return logits, confidence

# ============================================================================
# DEEPFAKE DETECTOR
# ============================================================================

class DeepfakeDetector:
    """Main deepfake detection class"""
    
    def __init__(self, model_path='./models_v2/best_model.pth', use_pretrained=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Download model from Google Drive if not present (for deployment)
        if not os.path.exists(model_path):
            try:
                print("📥 Downloading model from Hugging Face/Google Drive...")
                from huggingface_hub import hf_hub_download
                os.makedirs('./models_v2', exist_ok=True)
                model_path = hf_hub_download(
                    repo_id="riceu69/deepguard-model",
                    filename="best_model.pth",
                    local_dir="./models_v2"
                )
                print("✅ Model downloaded")
            except Exception as e:
                print(f"⚠️  Could not download model: {e}")
                print("    Using pretrained backbones only")
        
        # Load model architecture
        print("Loading model architecture...")
        self.model = TwoModelEnsemble(num_classes=2)
        
        # Load trained weights
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading weights: {e}")
                print("    Using pretrained backbones only")
        else:
            print(f"⚠️  Model not found at {model_path}")
            print("    Using pretrained backbones only")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        print("Loading face detector...")
        self.face_detector = MTCNN()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Detection threshold
        self.threshold = 0.35
        
        print("✅ DeepfakeDetector ready!\n")
    
    def detect_and_crop_face(self, image):
        """Detect and crop face from image"""
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        try:
            # Detect face
            result = self.face_detector.detect_faces(img_array)
            
            if result is not None and len(result) > 0:
                box = result[0]['box']
                x, y, w, h = box
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding
                
                # Crop face
                face = img_array[y:y+h, x:x+w]
                if face.size > 0:
                    return Image.fromarray(face)
        except Exception as e:
            print(f"Face detection error: {e}")
        
        # Return original if face detection fails
        return Image.fromarray(img_array) if not isinstance(image, Image.Image) else image
    
    def extract_frames(self, video_path, max_frames=30, skip_frames=5):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted = 0
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def detect_faces(self, frames):
        """Detect and extract faces from frames"""
        faces = []
        
        for frame in frames:
            result = self.face_detector.detect_faces(frame)
            
            if result is not None and len(result) > 0:
                box = result[0]['box']
                x, y, w, h = box
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding
                
                # Crop face
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    faces.append(face)
        
        print(f"Detected {len(faces)} faces")
        return faces
    
    def predict_frame(self, face):
        """Predict if a single face is real or fake"""
        # Convert to PIL if needed
        if not isinstance(face, Image.Image):
            face_pil = Image.fromarray(face)
        else:
            face_pil = face
        
        # Preprocess
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, confidence = self.model(face_tensor)
            probs = F.softmax(logits, dim=1)
            fake_prob = probs[0][1].item()
        
        return fake_prob
    
    def predict_image(self, image):
        """
        Predict if a single image is fake
        Returns: (is_fake: bool, confidence: float)
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Detect and crop face
            face = self.detect_and_crop_face(image)
            
            # Get prediction
            fake_prob = self.predict_frame(face)
            
            # Determine if fake
            is_fake = fake_prob > self.threshold
            confidence = fake_prob * 100
            
            return is_fake, confidence
        
        except Exception as e:
            print(f"Error in predict_image: {e}")
            return False, 0.0
    
    def analyze_video(self, video_path):
        """
        Main function to analyze video for deepfakes
        Returns: dict with detection results
        """
        print(f"\nAnalyzing video: {video_path}")
        
        # Liveness detection disabled for deployment
        # (Requires dlib/face_recognition which don't build on free tier)
        liveness_results = {
            'blink_detected': False,
            'head_turn_left': False,
            'head_turn_right': False,
            'smile_detected': False,
            'liveness_score': 0,
            'is_live': False
        }
        
        # Extract frames
        frames = self.extract_frames(video_path)
        if len(frames) == 0:
            return {
                'error': 'Could not extract frames from video',
                'is_deepfake': False,
                'confidence': 0.0,
                'frames_analyzed': 0,
                'faces_detected': 0,
                'liveness': liveness_results
            }
        
        # Detect faces
        faces = self.detect_faces(frames)
        if len(faces) == 0:
            return {
                'error': 'No faces detected in video',
                'is_deepfake': False,
                'confidence': 0.0,
                'frames_analyzed': len(frames),
                'faces_detected': 0,
                'liveness': liveness_results
            }
        
        # Predict on each face
        predictions = []
        for face in faces:
            fake_prob = self.predict_frame(face)
            predictions.append(fake_prob)
        
        # Calculate temporal consistency (frame-to-frame variance)
        temporal_consistency = np.std(predictions) if len(predictions) > 1 else 0
        
        # Aggregate results
        avg_fake_prob = np.mean(predictions)
        max_fake_prob = np.max(predictions)
        
        # Decision
        is_deepfake = avg_fake_prob > self.threshold
        
        # Build result dictionary
        result = {
            'is_deepfake': bool(is_deepfake),
            'confidence': float(avg_fake_prob * 100),
            'max_confidence': float(max_fake_prob * 100),
            'frames_analyzed': len(frames),
            'faces_detected': len(faces),
            'liveness': liveness_results,
            'temporal_consistency': float(temporal_consistency),
            'risk_scores': {
                'deepfake_score': float(avg_fake_prob * 100),
                'liveness_score': 0,  # Disabled
                'temporal_score': float((1 - min(temporal_consistency, 1.0)) * 100),
                'overall_risk': float(avg_fake_prob * 100)
            },
            'prediction_details': {
                'average_fake_probability': float(avg_fake_prob),
                'max_fake_probability': float(max_fake_prob),
                'threshold': self.threshold,
                'adjusted_probability': float(avg_fake_prob)
            }
        }
        
        print(f"Analysis complete: {'DEEPFAKE' if is_deepfake else 'REAL'}")
        print(f"Confidence: {avg_fake_prob * 100:.2f}%")
        
        return result

# ============================================================================
# FACE MATCHING MODEL
# ============================================================================

class FaceMatchingModel:
    """Face verification using FaceNet embeddings"""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Loading FaceNet for identity matching...")
        
        # Load FaceNet
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Face detector
        self.face_detector = MTCNN()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Similarity threshold
        self.threshold = 0.6
        
        print("✅ FaceNet loaded for face matching\n")
    
    def detect_and_crop_face(self, image):
        """Detect and crop face from PIL image"""
        img_array = np.array(image)
        
        try:
            boxes, probs = self.face_detector.detect(img_array)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Add padding
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_array.shape[1], x2 + padding)
                y2 = min(img_array.shape[0], y2 + padding)
                
                # Crop face
                face = img_array[y1:y2, x1:x2]
                return Image.fromarray(face)
        except Exception as e:
            print(f"Face detection error: {e}")
        
        # Return original if detection fails
        return image
    
    def get_embedding(self, image):
        """Get face embedding from image"""
        try:
            # Detect and crop face
            face = self.detect_and_crop_face(image)
            
            # Transform
            img_tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.facenet(img_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding
        
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def predict(self, id_image, selfie_image):
        """
        Check if two faces match
        Returns: is_match (0 or 1)
        """
        try:
            # Get embeddings
            emb1 = self.get_embedding(id_image)
            emb2 = self.get_embedding(selfie_image)
            
            if emb1 is None or emb2 is None:
                return 0
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            
            # Determine match
            is_match = 1 if similarity > self.threshold else 0
            
            return is_match
        
        except Exception as e:
            print(f"Error in face matching: {e}")
            return 0
    
    def predict_with_confidence(self, id_image, selfie_image):
        """
        Check if two faces match with confidence score
        Returns: (is_match: bool, confidence: float)
        """
        try:
            # Get embeddings
            emb1 = self.get_embedding(id_image)
            emb2 = self.get_embedding(selfie_image)
            
            if emb1 is None or emb2 is None:
                return False, 0.0
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            
            # Convert to percentage (0-100)
            confidence = similarity * 100
            
            # Determine match
            is_match = similarity > self.threshold
            
            return is_match, confidence
        
        except Exception as e:
            print(f"Error in face matching: {e}")
            return False, 0.0

# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING DEEPFAKE DETECTOR")
    print("="*70)
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Test with a video (if exists)
    test_video = "./uploads/test_video.mp4"
    if os.path.exists(test_video):
        result = detector.analyze_video(test_video)
        print("\nFinal Result:")
        print(result)
    else:
        print(f"\nTest video not found at {test_video}")
    
    # Test face matching
    print("\n" + "="*70)
    print("TESTING FACE MATCHING")
    print("="*70)
    
    matcher = FaceMatchingModel()
    print("Face matcher initialized!")