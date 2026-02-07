import cv2
import numpy as np
import re
import warnings
import torch
from ultralytics.nn.tasks import DetectionModel

warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    pass

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
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
        
        # Initialize EasyOCR
        self.ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
    
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
        Full post-processing with digit/letter corrections
        """
        if not raw_text:
            return "UNREADABLE"
        
        raw = self.clean_plate(raw_text)
        
        # Special state-code OCR corrections
        if raw.startswith("XA"):
            candidate = "KA" + raw[2:]
            if re.match(r'^KA[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', candidate):
                raw = candidate
        
        if raw.startswith("DI"):
            candidate = "DL" + raw[2:]
            if re.match(r'^DL[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', candidate):
                raw = candidate
        
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
        """Upscale 3x + Denoise + Sharpen"""
        upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    
    def _run_ocr(self, image):
        """Run EasyOCR on an image and return raw text or None"""
        try:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            results = self.ocr.readtext(
                image, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            )
            if results:
                texts = [t for (_, t, c) in results if c > 0.1]
                if texts:
                    return "".join(texts).upper()
        except Exception:
            pass
        return None

    def perform_ocr(self, crop):
        """
        Perform OCR with multiple preprocessing methods.
        Uses EasyOCR + full post-processing for best accuracy.
        """
        if crop is None or crop.size == 0:
            return "UNREADABLE", "UNREADABLE"
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return "UNREADABLE", "UNREADABLE"
        
        results = []
        
        # Try original image first
        raw = self._run_ocr(crop)
        if raw:
            plate = self.smart_post_process(raw)
            results.append((raw, plate))
            if plate not in ["INVALID", "UNREADABLE"]:
                return (raw, plate)
        
        # Try all 5 preprocessing methods
        for fn in [self.preprocess_v1, self.preprocess_v2, self.preprocess_v3,
                   self.preprocess_v4, self.preprocess_v5]:
            try:
                processed = fn(crop)
                raw = self._run_ocr(processed)
                if raw:
                    plate = self.smart_post_process(raw)
                    results.append((raw, plate))
                    if plate not in ["INVALID", "UNREADABLE"]:
                        return (raw, plate)
            except Exception:
                continue
        
        if results:
            non_unreadable = [r for r in results if r[0]]
            if non_unreadable:
                return max(non_unreadable, key=lambda x: len(self.clean_plate(x[0])))
            return results[0]
        
        return "UNREADABLE", "UNREADABLE"
