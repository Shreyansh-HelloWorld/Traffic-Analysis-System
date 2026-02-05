import cv2
import numpy as np
import re
import os
import warnings
import torch

# Suppress harmless PyTorch torch.classes introspection warnings
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

# Patch torch.load to use weights_only=False for YOLO model loading (PyTorch 2.6+ compatibility)
# This is safe because we trust our own model files in the models/ directory
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

# Use EasyOCR - best balance of accuracy and reliability on Streamlit Cloud
import easyocr

class ANPRDetector:
    """Automatic Number Plate Recognition Detector"""
    
    # Indian state codes
    STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
        "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
        "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
        "TN", "TR", "TS", "UK", "UP", "WB"
    }
    
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize ANPR Detector
        
        Args:
            model_path: Path to YOLO weights for plate detection
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Initialize EasyOCR with English - optimized for license plates
        self.ocr = easyocr.Reader(
            ['en'],
            gpu=False,
            verbose=False
        )
    
    def detect_plates(self, image):
        """
        Detect number plates in image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detected plates with bounding boxes and confidence
        """
        results = self.model(image, conf=self.conf_threshold)[0]
        
        plates = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                plates.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        
        return plates
    
    @staticmethod
    def clean_plate(text):
        """Remove non-alphanumeric characters and convert to uppercase"""
        text = text.upper()
        return re.sub(r'[^A-Z0-9]', '', text)
    
    def smart_post_process(self, raw_text):
        """
        Post-process OCR text to extract valid Indian number plate
        Simplified approach matching successful Colab notebook
        
        Args:
            raw_text: Raw OCR output
            
        Returns:
            Cleaned and validated plate number
        """
        if not raw_text:
            return "UNREADABLE"
        
        raw = self.clean_plate(raw_text)
        
        # Find valid state code (matching Colab approach)
        for i in range(len(raw) - 1):
            if raw[i:i+2] in self.STATE_CODES:
                raw = raw[i:]
                break
        
        if len(raw) < 6 or raw[:2] not in self.STATE_CODES:
            return "INVALID"
        
        state = raw[:2]
        rest = list(raw[2:])
        
        # Simple character corrections (matching Colab approach)
        for i in range(len(rest)):
            if i < 2:
                # RTO code positions: letters often misread as digits
                rest[i] = rest[i].replace('O', '0').replace('I', '1')
            elif i >= len(rest) - 4:
                # Last 4 positions should be digits
                rest[i] = rest[i].replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8')
        
        plate = state + ''.join(rest)
        
        # Standard Indian plate format: AA00AA0000 or AA0AA0000
        if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', plate):
            return plate
        
        # Bharat Series (BH): 00BH0000AA
        if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', plate):
            return plate
        
        return "INVALID"
    
    def _run_ocr(self, image):
        """Run OCR on image using EasyOCR - simple approach like Colab"""
        try:
            # EasyOCR works with numpy arrays directly
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    # Convert grayscale to BGR for EasyOCR
                    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    img = image
            else:
                img = np.array(image)
            
            # Simple OCR call - like PaddleOCR in Colab
            # EasyOCR parameters optimized for license plates
            results = self.ocr.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Restrict to plate chars
            )
            
            if results:
                # Combine all detected text (like Colab: "".join([l[1][0] for l in ocr_res[0]]))
                texts = [text for (bbox, text, conf) in results if conf > 0.1]
                if texts:
                    combined = "".join(texts)
                    return combined.upper()
            
        except Exception as e:
            print(f"  -> OCR error: {e}")
        return None
    
    def perform_ocr(self, crop):
        """
        Perform OCR - simplified approach matching successful Colab notebook
        
        Args:
            crop: Cropped plate image
            
        Returns:
            Tuple of (raw_text, final_plate)
        """
        if crop is None or crop.size == 0:
            return "UNREADABLE", "UNREADABLE"
        
        # Ensure crop is valid size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return "UNREADABLE", "UNREADABLE"
        
        results = []
        
        # Method 1: Direct OCR on original crop (like Colab - this worked best!)
        print("  -> Trying direct OCR on original crop...")
        raw_text = self._run_ocr(crop)
        if raw_text:
            print(f"  -> OCR result: {raw_text}")
            final_plate = self.smart_post_process(raw_text)
            if final_plate not in ["INVALID", "UNREADABLE"]:
                return (raw_text, final_plate)
            results.append((raw_text, final_plate))
        
        # Method 2: Simple upscale only (helps with small plates)
        print("  -> Trying 2x upscale...")
        upscaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        raw_text = self._run_ocr(upscaled)
        if raw_text:
            print(f"  -> OCR result: {raw_text}")
            final_plate = self.smart_post_process(raw_text)
            if final_plate not in ["INVALID", "UNREADABLE"]:
                return (raw_text, final_plate)
            results.append((raw_text, final_plate))
        
        # Method 3: Grayscale + contrast (simple preprocessing)
        print("  -> Trying grayscale + contrast...")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        raw_text = self._run_ocr(enhanced)
        if raw_text:
            print(f"  -> OCR result: {raw_text}")
            final_plate = self.smart_post_process(raw_text)
            if final_plate not in ["INVALID", "UNREADABLE"]:
                return (raw_text, final_plate)
            results.append((raw_text, final_plate))
        
        # Method 4: Upscale + grayscale (combination)
        print("  -> Trying upscale + grayscale...")
        upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        raw_text = self._run_ocr(upscaled_gray)
        if raw_text:
            print(f"  -> OCR result: {raw_text}")
            final_plate = self.smart_post_process(raw_text)
            if final_plate not in ["INVALID", "UNREADABLE"]:
                return (raw_text, final_plate)
            results.append((raw_text, final_plate))
        
        # If no valid plate found, return best raw result
        if results:
            # Prefer results with more alphanumeric characters
            non_unreadable = [r for r in results if r[0] and r[0] != "UNREADABLE"]
            if non_unreadable:
                best = max(non_unreadable, key=lambda x: len(self.clean_plate(x[0])))
                return best
            return results[0]
        
        return "UNREADABLE", "UNREADABLE"
