import cv2
import numpy as np
import re
import os
import warnings
import torch
import platform

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

# Detect if running on Mac (for fallback to Tesseract)
IS_MAC = platform.system() == "Darwin"

# Import OCR based on platform
if IS_MAC:
    import pytesseract
    from PIL import Image
    USING_PADDLE = False
else:
    # Use PaddleOCR 3.x for Linux/Cloud deployment
    from paddleocr import PaddleOCR
    USING_PADDLE = True

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
        
        if USING_PADDLE:
            # Initialize PaddleOCR 3.x with settings optimized for license plates
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            # Tesseract config for Mac fallback
            self.tesseract_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
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
        """Upscale + Grayscale + Contrast Enhancement"""
        upscaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        alpha = 1.5  # Contrast
        beta = 0     # Brightness
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
        """Best for Indian plates: Upscale + CLAHE + Otsu threshold"""
        # Upscale for better character recognition
        upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Otsu's thresholding
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
    
    def _run_ocr(self, image):
        """Run OCR on image - uses PaddleOCR on Linux, Tesseract on Mac"""
        try:
            if USING_PADDLE:
                # PaddleOCR 3.x API
                # Save image temporarily for PaddleOCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name
                    cv2.imwrite(temp_path, image)
                
                try:
                    result = self.ocr.predict(input=temp_path)
                    # Parse PaddleOCR 3.x result format - rec_texts contains recognized text
                    texts = []
                    if result:
                        for res in result:
                            # res.rec_texts is a list of recognized texts
                            if hasattr(res, 'rec_texts') and res.rec_texts:
                                texts.extend(res.rec_texts)
                    if texts:
                        return self.normalize_ocr_text("".join(texts))
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # Tesseract (for Mac fallback)
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 2:
                        pil_image = Image.fromarray(image)
                    else:
                        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = image
                text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
                return self.normalize_ocr_text(text) if text.strip() else None
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
        
        # Try original image first
        print("  -> Running OCR on original image...")
        raw_text = self._run_ocr(crop)
        if raw_text:
            print(f"  -> OCR result: {raw_text}")
            final_plate = self.smart_post_process(raw_text)
            if final_plate not in ["INVALID", "UNREADABLE"]:
                return (raw_text, final_plate)  # Early exit on valid plate
            results.append((raw_text, final_plate))
        else:
            print("  -> OCR returned no result")
        
        # Try preprocessing methods for better accuracy
        preprocessing_methods = [
            ("Otsu", self.preprocess_v6),           # Best for Indian plates
            ("CLAHE", self.preprocess_v1),          # Best for low contrast
            ("Upscale", self.preprocess_v3),        # Best for small plates
            ("Threshold", self.preprocess_v2),      # Best for noisy images
            ("Morphology", self.preprocess_v4),     # Best for broken characters
        ]
        
        for name, preprocess_func in preprocessing_methods:
            try:
                print(f"  -> Trying {name} preprocessing...")
                processed = preprocess_func(crop)
                raw_text = self._run_ocr(processed)
                if raw_text:
                    print(f"  -> OCR result: {raw_text}")
                    final_plate = self.smart_post_process(raw_text)
                    if final_plate not in ["INVALID", "UNREADABLE"]:
                        return (raw_text, final_plate)  # Early exit on valid plate
                    results.append((raw_text, final_plate))
                else:
                    print(f"  -> {name} OCR returned no result")
            except Exception as e:
                print(f"  -> {name} preprocessing failed: {e}")
                continue
        
        # If no valid plate found, return best raw result
        if results:
            # Prefer results with more alphanumeric characters
            non_unreadable = [r for r in results if r[0] and r[0] != "UNREADABLE"]
            if non_unreadable:
                return max(non_unreadable, key=lambda x: len(self.clean_plate(x[0])))
            return results[0]
        
        return "UNREADABLE", "UNREADABLE"
