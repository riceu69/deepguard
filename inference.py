"""
Complete Inference Script for ZenTej Hackathon
Deepfake-Proof eKYC Challenge

Name: Rishabh Raj (individual)
Performs: Identity Matching + Deepfake Detection
"""

import os
import csv
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import face_recognition
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION (Leave Blank for Organizers)
# ================================
TEST_DATASET_PATH = ""
LABELS_CSV_PATH = ""

# ================================
# MODEL ARCHITECTURE (Must match training)
# ================================
import torch.nn as nn
import timm
from facenet_pytorch import InceptionResnetV1

class TwoModelEnsemble(nn.Module):
    """Dual-model ensemble: EfficientNet + FaceNet"""
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.facenet = InceptionResnetV1(pretrained=None)
        
        self.fusion = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        eff_feat = self.efficientnet(x)
        face_feat = self.facenet(x)
        combined = torch.cat([eff_feat, face_feat], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        confidence = self.confidence_head(fused)
        return logits, confidence

# ================================
# UNIFIED KYC MODEL
# ================================
class UnifiedKYCModel:
    """
    Complete KYC system:
    1. Identity Matching (Face Verification)
    2. Deepfake Detection
    """
    def __init__(self, model_path='./models_v2/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load deepfake detector
        print("Loading deepfake detector...")
        self.deepfake_model = TwoModelEnsemble(num_classes=2)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.deepfake_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Model loaded (AUC: {checkpoint.get('val_auc', 'N/A'):.4f})")
            else:
                self.deepfake_model.load_state_dict(checkpoint)
                print("✓ Model loaded")
        else:
            print(f"⚠ Warning: Model not found at {model_path}")
        
        self.deepfake_model.to(self.device)
        self.deepfake_model.eval()
        
        # Image preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Thresholds (tuned on validation set)
        self.match_threshold = 0.6
        self.fake_threshold = 0.5
        
        print("✓ KYC system ready\n")
    
    def detect_face_encoding(self, image):
        """Get face encoding for matching"""
        img_np = np.array(image.convert('RGB'))
        face_encodings = face_recognition.face_encodings(img_np)
        
        if not face_encodings:
            return None
        
        return face_encodings[0]
    
    def predict_match(self, id_image, selfie_image):
        """
        Predict if two images match (same person)
        Returns: 1 if match, 0 if no match
        """
        id_encoding = self.detect_face_encoding(id_image)
        selfie_encoding = self.detect_face_encoding(selfie_image)
        
        if id_encoding is None or selfie_encoding is None:
            return 0  # No face detected = no match
        
        distance = face_recognition.face_distance([id_encoding], selfie_encoding)[0]
        is_match = 1 if distance < self.match_threshold else 0
        
        return is_match
    
    def predict_fake(self, selfie_image):
        """
        Predict if selfie is fake/deepfake
        Returns: 1 if fake, 0 if real
        """
        try:
            # Preprocess
            img_tensor = self.transform(selfie_image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits, confidence = self.deepfake_model(img_tensor)
                probs = F.softmax(logits, dim=1)
                fake_prob = probs[0][1].item()
            
            is_fake = 1 if fake_prob > self.fake_threshold else 0
            return is_fake
        
        except Exception as e:
            print(f"Error in deepfake detection: {e}")
            return 1  # Default to fake if error

# ================================
# MODEL LOADING
# ================================
def load_model():
    """Load the unified KYC model"""
    model = UnifiedKYCModel(model_path='./models/best_model.pth')
    return model

# ================================
# INFERENCE ON ONE PAIR
# ================================
def predict_match_and_fake(id_image, selfie_image, model):
    """
    Predict both identity match and fake detection
    
    Args:
        id_image: PIL Image of ID photo
        selfie_image: PIL Image of selfie
        model: UnifiedKYCModel instance
    
    Returns:
        (is_match: 0/1, is_fake: 0/1)
    """
    try:
        is_match = model.predict_match(id_image, selfie_image)
        is_fake = model.predict_fake(selfie_image)
        return is_match, is_fake
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, 1  # Default: no match, fake

# ================================
# FULL INFERENCE LOOP
# ================================
def run_inference(test_path=None):
    """
    Run inference on entire test dataset
    
    Args:
        test_path: Override TEST_DATASET_PATH
    
    Returns:
        List of predictions: [[kyc_id, selfie_name, is_match, is_fake], ...]
    """
    dataset_path = test_path if test_path else TEST_DATASET_PATH
    
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError(f"Test dataset path not found: {dataset_path}")
    
    print(f"Loading model...")
    model = load_model()
    
    print(f"\nRunning inference on: {dataset_path}")
    print("="*60)
    
    predictions = []
    kyc_folders = sorted(os.listdir(dataset_path))
    
    for i, kyc_folder in enumerate(kyc_folders, 1):
        folder_path = os.path.join(dataset_path, kyc_folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f"[{i}/{len(kyc_folders)}] Processing {kyc_folder}...", end=' ')
        
        # Load ID image
        id_path = os.path.join(folder_path, "id.jpg")
        if not os.path.exists(id_path):
            print("⚠ ID image not found")
            continue
        
        id_image = Image.open(id_path).convert('RGB')
        
        # Process all selfies
        selfie_count = 0
        for file in sorted(os.listdir(folder_path)):
            if file.startswith("selfie_"):
                selfie_path = os.path.join(folder_path, file)
                selfie_image = Image.open(selfie_path).convert('RGB')
                
                is_match, is_fake = predict_match_and_fake(id_image, selfie_image, model)
                
                predictions.append([kyc_folder, file, is_match, is_fake])
                selfie_count += 1
        
        print(f"✓ {selfie_count} selfies")
    
    print("="*60)
    print(f"✓ Inference complete! Total predictions: {len(predictions)}\n")
    
    return predictions

# ================================
# EVALUATION
# ================================
def evaluate(predictions, labels_path=None):
    """Evaluate predictions against ground truth"""
    csv_path = labels_path if labels_path else LABELS_CSV_PATH
    
    if not csv_path or not os.path.exists(csv_path):
        print("⚠ Labels file not found. Skipping evaluation.")
        return
    
    # Load ground truth
    gt = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if present
        
        for row in reader:
            if len(row) >= 4:
                gt[(row[0], row[1])] = (int(row[2]), int(row[3]))
    
    # Collect predictions
    y_true_match, y_pred_match = [], []
    y_true_fake, y_pred_fake = [], []
    
    for (kyc, selfie, pm, pf) in predictions:
        if (kyc, selfie) in gt:
            true_match, true_fake = gt[(kyc, selfie)]
            y_true_match.append(true_match)
            y_pred_match.append(pm)
            y_true_fake.append(true_fake)
            y_pred_fake.append(pf)
    
    if not y_true_match:
        print("⚠ No matching samples found")
        return
    
    # Print results
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 IDENTITY MATCHING METRICS")
    print("-"*60)
    match_acc = accuracy_score(y_true_match, y_pred_match)
    print(f"Accuracy: {match_acc*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_true_match, y_pred_match, 
                              target_names=['No Match', 'Match'],
                              digits=4))
    
    print("\n📊 DEEPFAKE DETECTION METRICS")
    print("-"*60)
    fake_acc = accuracy_score(y_true_fake, y_pred_fake)
    print(f"Accuracy: {fake_acc*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_true_fake, y_pred_fake,
                              target_names=['Real', 'Fake'],
                              digits=4))
    
    print("\n" + "="*60)
    print(f"📈 OVERALL PERFORMANCE")
    print("="*60)
    print(f"Identity Matching Accuracy: {match_acc*100:.2f}%")
    print(f"Deepfake Detection Accuracy: {fake_acc*100:.2f}%")
    print(f"Average Accuracy: {(match_acc + fake_acc)*50:.2f}%")
    print("="*60 + "\n")

# ================================
# SAVE PREDICTIONS
# ================================
def save_predictions(predictions, output_path='predictions.csv'):
    """Save predictions to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kyc_id', 'selfie_name', 'is_match', 'is_fake'])
        writer.writerows(predictions)
    
    print(f"✓ Predictions saved to: {output_path}")

# ================================
# MAIN EXECUTION
# ================================
def main():
    """Main with argparse support"""
    parser = argparse.ArgumentParser(description='KYC Inference Script')
    parser.add_argument('--test_path', type=str, default=TEST_DATASET_PATH,
                       help='Path to test dataset')
    parser.add_argument('--labels_path', type=str, default=LABELS_CSV_PATH,
                       help='Path to labels CSV')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation')
    
    args = parser.parse_args()
    
    # Run inference
    print("\n🚀 Starting KYC Inference Pipeline\n")
    predictions = run_inference(test_path=args.test_path)
    
    # Save predictions
    save_predictions(predictions, output_path=args.output)
    
    # Evaluate if labels available
    if not args.no_eval:
        evaluate(predictions, labels_path=args.labels_path)
    
    print("✅ Pipeline complete!\n")

# ================================
# EXECUTE
# ================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        main()
    else:
        # Direct execution mode (for organizers)
        print("\n🚀 Starting KYC Inference Pipeline\n")
        
        if not TEST_DATASET_PATH:
            print("⚠ TEST_DATASET_PATH is empty!")
            print("Usage:")
            print("  python inference.py --test_path /path/to/test --labels_path /path/to/labels.csv")
            sys.exit(1)
        
        predictions = run_inference()
        save_predictions(predictions)
        evaluate(predictions)
        
        print("✅ Pipeline complete!\n")