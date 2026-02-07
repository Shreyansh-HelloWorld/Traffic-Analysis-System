import cv2
import numpy as np
import re
import os
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

os.environ["FLAGS_use_cuda"] = "0"
from paddleocr import PaddleOCR

class ANPRDetector:
    """Automatic Number Plate Recognition Detector"""
    
    # ─── Current state/UT codes (37 active) ───
    STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL",
        "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
        "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
        "TN", "TR", "TG", "UK", "UP", "WB"
    }
    
    # ─── Legacy codes still valid on existing vehicles ───
    LEGACY_CODES = {
        "TS",   # Telangana (June 2014 – March 2024, replaced by TG)
        "OR",   # Orissa (until 2012, now OD)
        "DN",   # Dadra & Nagar Haveli (until 2020, merged into DD)
        "UA",   # Uttaranchal (2000–2007, now UK)
    }
    
    ALL_STATE_CODES = STATE_CODES | LEGACY_CODES   # union of both sets
    
    # ─── OCR char → letter substitution table (digit-that-looks-like-letter) ───
    DIGIT_TO_LETTER = {
        '0': 'O', '1': 'I', '2': 'Z', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B',
    }
    # ─── OCR char → digit substitution table (letter-that-looks-like-digit) ───
    LETTER_TO_DIGIT = {
        'O': '0', 'I': '1', 'S': '5', 'B': '8',
        'Z': '2', 'G': '6', 'D': '0', 'Q': '0',
        'T': '7', 'A': '4', 'L': '1', 'J': '1',
    }
    # Strict subset: only high-confidence digit look-alikes.
    # Used for positional detection (backward walk, RTO boundary)
    # to avoid swallowing real series letters like A, T, L, J.
    STRICT_DIGIT_LIKES = {
        'O': '0', 'I': '1', 'S': '5', 'B': '8',
        'Z': '2', 'G': '6', 'D': '0', 'Q': '0',
    }
    
    # ─── Common 2-char OCR misreads → correct state code ───
    # ONLY unambiguous mappings — each key can only mean ONE state code.
    # Ambiguous patterns like "00" (could be DD, DL, OD) are left
    # to the brute-force substitution in _fix_state_code().
    STATE_OCR_FIXES = {
        "XA": "KA", "X4": "KA", "K4": "KA",           # Karnataka
        "DI": "DL", "D1": "DL",                         # Delhi
        "0L": "DL", "OL": "DL",                         # Delhi (0/O misread as D)
        "0D": "OD", "O0": "OD", "01": "OD",             # Odisha
        "6A": "GA", "G4": "GA",                         # Goa
        "6J": "GJ",                                      # Gujarat
        "8R": "BR", "8K": "BR",                          # Bihar
        "N1": "NL",                                       # Nagaland
        "C6": "CG",                                       # Chhattisgarh
        "T5": "TS",                                       # Telangana (legacy)
        "7N": "TN", "7R": "TR", "76": "TG", "T6": "TG", # TN/TR/TG
        "5K": "SK",                                       # Sikkim
        "M2": "MZ",                                       # Mizoram
        "HA": "HR",                                       # Haryana
    }
    
    # ─── Standard format regex ───
    # SS  DD   X{0-3}  DDDD
    RE_STANDARD = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$')
    # Bharat Series: YY BH DDDD AA
    RE_BH = re.compile(r'^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}$')
    
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize ANPR Detector
        
        Args:
            model_path: Path to YOLO weights for plate detection
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det=False,
            rec=True,
            show_log=False,
            use_gpu=False
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
    
    # ──────────────────────────────────────────────────
    #  CORE POST-PROCESSING  (v2 — research-backed)
    # ──────────────────────────────────────────────────
    def smart_post_process(self, raw_text):
        """
        Post-process OCR text to extract valid Indian number plate.
        
        Handles:
          • 37 current + 4 legacy state codes (TS/OR/DN/UA)
          • Standard format  SS DD X{0-3} DDDD
          • Bharat Series    YY BH DDDD AA
          • Delhi single-digit RTO  (DL 1 … DL 9)
          • No-series-letter plates  SS DD DDDD  (older registrations)
          • India's O/I prohibition in series letters
          • Comprehensive OCR confusion corrections
          • State-code-level OCR recovery via substitution table
        """
        if not raw_text:
            return "UNREADABLE"
        
        raw = self.clean_plate(raw_text)
        if len(raw) < 4:
            return "UNREADABLE"
        
        # ── 1. CHECK BHARAT SERIES EARLY (digits come first) ──
        bh_candidate = self._try_bharat_series(raw)
        if bh_candidate:
            return bh_candidate
        
        # ── 2. TRY DIRECT STATE CODE MATCH ──
        raw = self._fix_state_code(raw)
        raw = self._find_state_offset(raw)
        
        # ── 3. VALIDATE WE HAVE A STATE CODE ──
        if len(raw) < 5 or raw[:2] not in self.ALL_STATE_CODES:
            return "INVALID"
        
        state = raw[:2]
        rest = raw[2:]
        
        # ── 4. PARSE THE REST: RTO + series + number ──
        candidate = self._parse_plate_body(state, rest)
        if candidate:
            return candidate
        
        return "INVALID"
    
    def _try_bharat_series(self, raw):
        """
        Detect Bharat Series: YYBHNNNNXX
        Year can be 20–29 (2020s) or later.
        """
        # Direct match
        if self.RE_BH.match(raw):
            return raw
        # BH might be buried after junk chars
        bh_pos = raw.find("BH")
        if bh_pos >= 2:
            before = raw[:bh_pos]
            after = raw[bh_pos + 2:]
            # before should be 2 digits (year), after should be digits + letters
            year_part = ''.join(c if c.isdigit() else self.LETTER_TO_DIGIT.get(c, c) for c in before[-2:])
            if year_part.isdigit():
                candidate = year_part + "BH" + after
                if self.RE_BH.match(candidate):
                    return candidate
        return None
    
    def _fix_state_code(self, raw):
        """Apply known OCR misread fixes for the first 2 characters."""
        prefix = raw[:2]
        if prefix in self.ALL_STATE_CODES:
            return raw
        # Check the explicit fix table
        if prefix in self.STATE_OCR_FIXES:
            return self.STATE_OCR_FIXES[prefix] + raw[2:]
        # Try digit→letter substitution on first 2 chars to recover a state code
        c0_options = [raw[0]]
        c1_options = [raw[1]]
        if raw[0] in self.DIGIT_TO_LETTER:
            c0_options.append(self.DIGIT_TO_LETTER[raw[0]])
        if raw[1] in self.DIGIT_TO_LETTER:
            c1_options.append(self.DIGIT_TO_LETTER[raw[1]])
        # 0 can look like both O and D — try D explicitly
        if raw[0] == '0':
            c0_options.append('D')
        if raw[1] == '0':
            c1_options.append('D')
        # Also try letter→different letter for common confusions
        extra_letter_map = {
            'X': 'K', 'N': 'H', 'H': 'N', 'W': 'M',
            'V': 'U', 'U': 'V', 'R': 'P', 'P': 'R',
        }
        if raw[0] in extra_letter_map:
            c0_options.append(extra_letter_map[raw[0]])
        if raw[1] in extra_letter_map:
            c1_options.append(extra_letter_map[raw[1]])
        
        for c0 in c0_options:
            for c1 in c1_options:
                code = c0 + c1
                if code in self.ALL_STATE_CODES and code != prefix:
                    return code + raw[2:]
        return raw
    
    def _find_state_offset(self, raw):
        """
        If the first 2 chars aren't a valid state code, scan forward
        to find one (handles leading junk/partial reads).
        """
        if raw[:2] in self.ALL_STATE_CODES:
            return raw
        for i in range(1, min(4, len(raw) - 4)):
            if raw[i:i+2] in self.ALL_STATE_CODES:
                return raw[i:]
        return raw
    
    def _parse_plate_body(self, state, rest):
        """
        Parse the body after the state code:
          rest = RTO(1-2 digits) + Series(0-3 letters) + Number(1-4 digits)
        
        Handles:
          • Delhi single-digit RTO
          • No-series older plates (SS DD DDDD)
          • O/I prohibition in series letters
          • Full OCR confusion correction for each positional slot
        """
        if len(rest) < 3:
            return None
        
        # ── Determine RTO length ──
        # Delhi can have 1-digit RTO; others 2-digit
        rto_len = 0
        for i in range(min(2, len(rest))):
            ch = rest[i]
            if ch.isdigit():
                rto_len += 1
            elif ch in self.STRICT_DIGIT_LIKES:
                # High-confidence digit look-alike in RTO position
                rto_len += 1
            else:
                break
        
        if rto_len == 0:
            return None
        
        # Allow single-digit RTO for Delhi, otherwise need at least 1
        if rto_len == 1 and state != "DL":
            # Try converting the char after RTO digit if it looks like a digit
            if len(rest) > 1 and rest[1] in self.STRICT_DIGIT_LIKES:
                rto_len = 2
        
        # ── Fix RTO digits ──
        rto_chars = list(rest[:rto_len])
        for i in range(len(rto_chars)):
            if rto_chars[i] in self.STRICT_DIGIT_LIKES:
                rto_chars[i] = self.STRICT_DIGIT_LIKES[rto_chars[i]]
        rto = ''.join(rto_chars)
        if not rto.isdigit():
            return None
        
        remaining = rest[rto_len:]
        if len(remaining) == 0:
            return None
        
        # ── Split remaining into series letters + registration number ──
        # Find where the trailing digits (registration number) start
        # Walk backward from end to find the digit block
        # Use STRICT set so series letters like A, T, L, J aren't swallowed
        num_end = len(remaining)
        num_start = num_end
        for j in range(num_end - 1, -1, -1):
            ch = remaining[j]
            if ch.isdigit() or ch in self.STRICT_DIGIT_LIKES:
                num_start = j
            else:
                break
        
        if num_start == num_end:
            return None  # No digits found
        
        series_raw = remaining[:num_start]
        number_raw = remaining[num_start:]
        
        # ── Fix registration number (must be digits) ──
        number_chars = list(number_raw)
        for i in range(len(number_chars)):
            if number_chars[i] in self.LETTER_TO_DIGIT:
                number_chars[i] = self.LETTER_TO_DIGIT[number_chars[i]]
        number = ''.join(number_chars)
        
        if not number.isdigit() or len(number) > 4 or len(number) == 0:
            return None
        
        # Pad to 4 digits if reasonable (e.g., "786" → "0786")
        if len(number) < 4 and len(number) >= 1:
            number = number.zfill(4)
        
        # ── Fix series letters ──
        series_chars = list(series_raw)
        for i in range(len(series_chars)):
            ch = series_chars[i]
            # India prohibits O and I in series letters → force to digit
            if ch == 'O':
                series_chars[i] = '0'  # Likely was a digit misread
            elif ch == 'I':
                series_chars[i] = '1'  # Likely was a digit misread
            # Digits in series position → convert to letter
            elif ch in self.DIGIT_TO_LETTER:
                series_chars[i] = self.DIGIT_TO_LETTER[ch]
        
        series = ''.join(series_chars)
        
        # If series has digits left (shouldn't), this plate is malformed
        if series and not series.isalpha():
            # One more attempt: maybe the digit is actually part of RTO or number
            # Reparse with different RTO length
            return self._reparse_flexible(state, rest)
        
        if len(series) > 3:
            return None   # Max 3 series letters
        
        candidate = state + rto + series + number
        
        # ── Validate final format ──
        if self.RE_STANDARD.match(candidate):
            return candidate
        
        return None
    
    def _reparse_flexible(self, state, rest):
        """
        Fallback: try all possible RTO/series/number splits
        and return the first valid one.
        """
        # Try RTO lengths 1 and 2
        for rto_len in (2, 1):
            if rto_len > len(rest):
                continue
            if rto_len == 1 and state != "DL":
                continue
            
            rto_raw = rest[:rto_len]
            rto_chars = [self.LETTER_TO_DIGIT.get(c, c) for c in rto_raw]
            rto = ''.join(rto_chars)
            if not rto.isdigit():
                continue
            
            tail = rest[rto_len:]
            # Try series lengths 0, 1, 2, 3
            for slen in range(0, min(4, len(tail))):
                series_raw = tail[:slen]
                num_raw = tail[slen:]
                if not num_raw:
                    continue
                
                # Fix series
                series_chars = []
                valid_series = True
                for ch in series_raw:
                    if ch in self.DIGIT_TO_LETTER:
                        series_chars.append(self.DIGIT_TO_LETTER[ch])
                    elif ch.isalpha() and ch not in ('O', 'I'):
                        series_chars.append(ch)
                    elif ch == 'O':
                        valid_series = False
                        break
                    elif ch == 'I':
                        valid_series = False
                        break
                    else:
                        valid_series = False
                        break
                if not valid_series:
                    continue
                series = ''.join(series_chars)
                
                # Fix number
                num_chars = [self.LETTER_TO_DIGIT.get(c, c) for c in num_raw]
                number = ''.join(num_chars)
                if not number.isdigit():
                    continue
                if len(number) > 4 or len(number) == 0:
                    continue
                number = number.zfill(4)
                
                candidate = state + rto + series + number
                if self.RE_STANDARD.match(candidate):
                    return candidate
        
        return None
    
    @staticmethod
    def _plate_score(plate):
        """
        Score a validated plate for ranking among candidates.
        Higher = more typical / more likely correct.
        """
        if plate in ("INVALID", "UNREADABLE"):
            return -1
        score = len(plate)  # Longer = more complete
        # Prefer 4-digit registration numbers
        m = re.match(r'^[A-Z]{2}(\d{1,2})([A-Z]{0,3})(\d{1,4})$', plate)
        if m:
            rto, series, num = m.group(1), m.group(2), m.group(3)
            if len(num) == 4:
                score += 5
            if len(rto) == 2:
                score += 2
            if 1 <= len(series) <= 2:
                score += 2   # Most common series length
        return score
    
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
    
    def perform_ocr(self, crop):
        """
        Perform OCR with multiple preprocessing methods.
        Uses PaddleOCR + full post-processing for best accuracy.
        Picks the highest-scored valid plate among all attempts.
        """
        if crop is None or crop.size == 0:
            return "UNREADABLE", "UNREADABLE"
        
        results = []
        
        # Try original image first
        try:
            ocr_res = self.ocr.ocr(crop, cls=True)
            if ocr_res and ocr_res[0]:
                raw_text = "".join([l[1][0] for l in ocr_res[0]])
                plate = self.smart_post_process(raw_text)
                results.append((raw_text, plate))
        except:
            pass
        
        # Try all 5 preprocessing methods
        for fn in [self.preprocess_v1, self.preprocess_v2, self.preprocess_v3,
                   self.preprocess_v4, self.preprocess_v5]:
            try:
                processed = fn(crop)
                ocr_res = self.ocr.ocr(processed, cls=True)
                if ocr_res and ocr_res[0]:
                    raw_text = "".join([l[1][0] for l in ocr_res[0]])
                    plate = self.smart_post_process(raw_text)
                    results.append((raw_text, plate))
            except:
                continue
        
        # ── Pick the best candidate ──
        valid = [(raw, pl) for raw, pl in results if pl not in ("INVALID", "UNREADABLE")]
        if valid:
            # Score-based: pick the most "typical" Indian plate
            best = max(valid, key=lambda x: self._plate_score(x[1]))
            return best
        elif results:
            non_empty = [r for r in results if r[0]]
            if non_empty:
                return max(non_empty, key=lambda x: len(x[0]))
            return results[0]
        
        return "UNREADABLE", "UNREADABLE"
