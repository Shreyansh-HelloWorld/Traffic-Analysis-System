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
        
        Args:
            raw_text: Raw OCR output
            
        Returns:
            Cleaned and validated plate number
        """
        if not raw_text:
            return "UNREADABLE"
        
        raw = self.clean_plate(raw_text)
        
        # Common OCR misreads for state codes - try all corrections
        STATE_CODE_CORRECTIONS = {
            # O/0 confused with D
            "OL": "DL", "0L": "DL",  # Delhi
            "ON": "DN", "0N": "DN",  # Dadra & Nagar Haveli
            "OD": "OD",              # Odisha (keep as is)
            # X confused with K
            "XA": "KA",              # Karnataka
            "XL": "KL",              # Kerala
            # I/1 confused with L
            "DI": "DL", "D1": "DL",  # Delhi
            "MI": "MH", "M1": "MH",  # Maharashtra
            # Other common confusions
            "AH": "MH",              # Maharashtra
            "HH": "MH",              # Maharashtra
            "TR": "TR",              # Tripura
            "T5": "TS",              # Telangana
            "TS": "TS",              # Telangana
        }
        
        # Try state code corrections at the beginning
        for wrong, correct in STATE_CODE_CORRECTIONS.items():
            if raw.startswith(wrong):
                raw = correct + raw[2:]
                break
        
        # Find valid state code
        for i in range(len(raw) - 1):
            if raw[i:i+2] in self.STATE_CODES:
                raw = raw[i:]
                break
        
        if len(raw) < 8 or raw[:2] not in self.STATE_CODES:
            # Check for Bharat Series (BH): format YYBHXXXXAA
            if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', raw):
                return raw
            return "INVALID"
        
        state = raw[:2]
        rest = raw[2:]
        
        if len(rest) < 4:
            return "INVALID"
        
        # Last 4 digits (strict digit slot)
        number = list(rest[-4:])
        for i in range(4):
            if number[i] == 'O': number[i] = '0'
            if number[i] == 'I': number[i] = '1'
            if number[i] == 'S': number[i] = '5'
            if number[i] == 'B': number[i] = '8'
            if number[i] == 'Z': number[i] = '2'
            if number[i] == 'G': number[i] = '6'
            if number[i] == 'D': number[i] = '0'
            if number[i] == 'Q': number[i] = '0'
        
        number = ''.join(number)
        
        # Middle part = RTO digits + series letters
        middle = list(rest[:-4])
        
        # First 1-2 chars → RTO digits
        for i in range(min(2, len(middle))):
            if middle[i] == 'O': middle[i] = '0'
            if middle[i] == 'I': middle[i] = '1'
            if middle[i] == 'S': middle[i] = '5'
            if middle[i] == 'B': middle[i] = '8'
            if middle[i] == 'Z': middle[i] = '2'
            if middle[i] == 'G': middle[i] = '6'
        
        # Remaining → series letters (reverse OCR confusion)
        for i in range(2, len(middle)):
            if middle[i] == '8': middle[i] = 'B'
            if middle[i] == '0': middle[i] = 'O'
            if middle[i] == '1': middle[i] = 'I'
            if middle[i] == '5': middle[i] = 'S'
            if middle[i] == '2': middle[i] = 'Z'
            if middle[i] == '6': middle[i] = 'G'
        
        candidate = state + ''.join(middle) + number
        
        # Standard Indian plate format
        if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', candidate):
            return candidate
        
        # Bharat Series (BH)
        if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', candidate):
            return candidate
        
        return "INVALID"
    
    @staticmethod
    def preprocess_v1(crop):
        """Grayscale + CLAHE + Bilateral Filter"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return denoised
    
    @staticmethod
    def preprocess_v2(crop):
        """Grayscale + Sharpening + Adaptive Threshold"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    @staticmethod
    def preprocess_v3(crop):
        """Upscale 4x + Grayscale + Contrast Enhancement"""
        # More aggressive upscaling for small plates
        upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        alpha = 1.8  # Higher contrast
        beta = 10    # Slight brightness boost
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        return enhanced
    
    @staticmethod
    def preprocess_v4(crop):
        """Grayscale + Morphological Operations"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return morph
    
    @staticmethod
    def preprocess_v5(crop):
        """Upscale 2x + Light Denoise + Sharpen (optimized for speed)"""
        upscaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # Use faster GaussianBlur instead of slow fastNlMeansDenoising
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    
    @staticmethod
    def preprocess_v6(crop):
        """Best for Indian plates: Aggressive Upscale + CLAHE + Otsu threshold"""
        # Very aggressive upscaling for better character recognition
        upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Otsu's thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod
    def preprocess_v7(crop):
        """Add padding + Upscale + High contrast for edge plates"""
        # Add white padding around the image (helps OCR detect edge characters)
        padding = 20
        padded = cv2.copyMakeBorder(crop, padding, padding, padding, padding, 
                                     cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # Upscale aggressively
        upscaled = cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # High contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Bilateral filter to reduce noise while keeping edges
        denoised = cv2.bilateralFilter(enhanced, 11, 75, 75)
        return denoised
    
    @staticmethod
    def preprocess_v8(crop):
        """Invert colors for white-on-black plates"""
        upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # Check if plate is dark (white text on black)
        mean_val = np.mean(gray)
        if mean_val < 127:
            # Invert for white-on-black plates
            gray = cv2.bitwise_not(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod
    def normalize_ocr_text(text):
        """Normalize common OCR character confusions"""
        if not text:
            return text
        # Common lowercase to uppercase confusions
        replacements = {
            'o': 'O', 'l': 'L', 'i': 'I', 's': 'S', 'z': 'Z',
            'b': 'B', 'g': 'G', 'q': 'Q', 'd': 'D', 'a': 'A',
            'e': 'E', 'r': 'R', 't': 'T', 'n': 'N', 'm': 'M',
            'h': 'H', 'k': 'K', 'p': 'P', 'u': 'U', 'v': 'V',
            'w': 'W', 'x': 'X', 'y': 'Y', 'c': 'C', 'f': 'F', 'j': 'J',
        }
        result = text.upper()
        # Remove spaces and special chars except alphanumeric
        result = ''.join(c for c in result if c.isalnum())
        return result
    
    def _run_ocr(self, image, allowlist=None):
        """Run OCR on image using EasyOCR"""
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
            
            # Build OCR parameters
            ocr_params = {
                'detail': 1,  # Get confidence scores
                'paragraph': False,
                'min_size': 5,
                'text_threshold': 0.3,  # Lower threshold for more detections
                'low_text': 0.2,
                'link_threshold': 0.3,
                'contrast_ths': 0.05,  # Very low contrast threshold
                'adjust_contrast': 0.8,
                'width_ths': 0.5,
                'decoder': 'beamsearch',  # Better decoder
                'beamWidth': 10,
            }
            
            # Add allowlist if specified (restrict to plate characters)
            if allowlist:
                ocr_params['allowlist'] = allowlist
            
            # Run EasyOCR
            results = self.ocr.readtext(img, **ocr_params)
            
            if results:
                # Sort by confidence and get all text
                texts = []
                for (bbox, text, conf) in results:
                    if conf > 0.05:  # Very low threshold - we'll filter later
                        texts.append((text, conf))
                
                if texts:
                    # Sort by confidence descending
                    texts.sort(key=lambda x: x[1], reverse=True)
                    combined = "".join([t[0] for t in texts])
                    return self.normalize_ocr_text(combined)
            
        except Exception as e:
            print(f"  -> OCR error: {e}")
        return None
    
    def perform_ocr(self, crop):
        """
        Perform OCR with multiple preprocessing methods for best accuracy
        
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
        
        # Allowlist for Indian license plates (letters + digits)
        plate_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Try preprocessing methods for better accuracy (prioritized order)
        preprocessing_methods = [
            ("Padded", self.preprocess_v7),        # Best for edge characters
            ("Otsu4x", self.preprocess_v6),        # Best for Indian plates with 4x upscale
            ("Upscale4x", self.preprocess_v3),     # Best for small plates
            ("Invert", self.preprocess_v8),        # Best for white-on-black plates
            ("CLAHE", self.preprocess_v1),         # Best for low contrast
            ("Morphology", self.preprocess_v4),    # Best for broken characters
            ("Threshold", self.preprocess_v2),     # Best for noisy images
        ]
        
        for name, preprocess_func in preprocessing_methods:
            try:
                print(f"  -> Trying {name} preprocessing...")
                processed = preprocess_func(crop)
                
                # Try with allowlist first (restricted to plate characters)
                raw_text = self._run_ocr(processed, allowlist=plate_allowlist)
                if raw_text:
                    print(f"  -> OCR result (with allowlist): {raw_text}")
                    final_plate = self.smart_post_process(raw_text)
                    if final_plate not in ["INVALID", "UNREADABLE"]:
                        return (raw_text, final_plate)  # Early exit on valid plate
                    results.append((raw_text, final_plate))
                
                # Try without allowlist (might catch more characters)
                raw_text = self._run_ocr(processed, allowlist=None)
                if raw_text:
                    print(f"  -> OCR result (no allowlist): {raw_text}")
                    final_plate = self.smart_post_process(raw_text)
                    if final_plate not in ["INVALID", "UNREADABLE"]:
                        return (raw_text, final_plate)  # Early exit on valid plate
                    results.append((raw_text, final_plate))
                else:
                    print(f"  -> {name} OCR returned no result")
            except Exception as e:
                print(f"  -> {name} preprocessing failed: {e}")
                continue
        
        # Also try original image as last resort
        print("  -> Trying original image...")
        for allowlist in [plate_allowlist, None]:
            raw_text = self._run_ocr(crop, allowlist=allowlist)
            if raw_text:
                print(f"  -> Original OCR result: {raw_text}")
                final_plate = self.smart_post_process(raw_text)
                if final_plate not in ["INVALID", "UNREADABLE"]:
                    return (raw_text, final_plate)
                results.append((raw_text, final_plate))
        
        # If no valid plate found, return best raw result
        if results:
            # Prefer results with more alphanumeric characters
            non_unreadable = [r for r in results if r[0] and r[0] != "UNREADABLE"]
            if non_unreadable:
                return max(non_unreadable, key=lambda x: len(self.clean_plate(x[0])))
            return results[0]
        
        return "UNREADABLE", "UNREADABLE"
