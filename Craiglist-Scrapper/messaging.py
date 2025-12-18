import csv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import time
import os
import io
import re
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI
import logging
from datetime import datetime
from collections import defaultdict
import json
from threading import Thread, Lock
from typing import Any, Dict, Optional

from all_rounder_2 import run_pipeline_from_payload
from database import init_database, save_message, get_conversations_by_thread, get_all_threads, get_messages_by_type

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None
    ImageEnhance = None
    ImageFilter = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    np = None
    CV2_AVAILABLE = False

# Optional: explicit path to tesseract executable (Windows often needs this),
# especially when running as a service where PATH may differ from your terminal.
# Example: TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if pytesseract is not None and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# Try to import ASGI adapter for uvicorn compatibility
try:
    from asgiref.wsgi import WsgiToAsgi
    ASGI_AVAILABLE = True
except ImportError:
    ASGI_AVAILABLE = False
    WsgiToAsgi = None

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Twilio credentials from .env file
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# OpenAI credentials from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')

# Validate required environment variables with detailed error messages
missing_vars = []
if not TWILIO_ACCOUNT_SID:
    missing_vars.append('TWILIO_ACCOUNT_SID')
if not TWILIO_AUTH_TOKEN:
    missing_vars.append('TWILIO_AUTH_TOKEN')
if not TWILIO_PHONE_NUMBER:
    missing_vars.append('TWILIO_PHONE_NUMBER')
if not OPENAI_API_KEY:
    missing_vars.append('OPENAI_API_KEY')
if not OPENAI_ASSISTANT_ID:
    missing_vars.append('OPENAI_ASSISTANT_ID')

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}\n"
    error_msg += "Please check your .env file. Common issues:\n"
    error_msg += "1. Variables must be on separate lines: VARIABLE_NAME=value\n"
    error_msg += "2. No spaces around the = sign: VARIABLE_NAME=value (not VARIABLE_NAME = value)\n"
    error_msg += "3. Multi-line values must be in quotes\n"
    error_msg += "4. If you see 'Python-dotenv could not parse statement' warnings, check those line numbers in your .env file"
    logger.error(error_msg)
    print(f"\n❌ ERROR: {error_msg}\n")
    raise ValueError("Missing required environment variables")

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_ASSISTANTS_HEADER = {"OpenAI-Beta": "assistants=v2"}

# Initialize Flask app for webhook server
app = Flask(__name__, static_folder='static', static_url_path='')

# System prompt for OpenAI - loaded from .env file, fallback to default
SYSTEM_PROMPT = os.getenv(
    'OPENAI_SYSTEM_PROMPT',
    """You are Carly. Your role is to contact individuals who have listed their vehicles for sale on Craigslist. Your primary goal is to collect the VIN number of the listed vehicle. When speaking with prospects, maintain a casual and friendly tone and act like a real human. If they ask any questions, respond politely and confidently as a real person who is interested in buying a vehicle which is listed on craigslist. If anyone shows hesitation, handle the conversation with care and reassurance. Keep your responses concise and conversational, like a text message."""
)

# Initial message to send when initiating contact - loaded from .env file, fallback to default
INITIAL_MESSAGE = os.getenv(
    'INITIAL_MESSAGE',
    "Hi, I'm interested in your car for sale at your asking price. Do you mind sending me the vin so I can run a carfax on it?"
)

# Store conversation history per phone number (in-memory storage)
# Format: {phone_number: [{"role": "user" or "assistant", "content": "message"}, ...]}
conversation_history = defaultdict(list)

# Store thread ID mapping (phone number -> thread_id)
# Format: {phone_number: thread_id}
thread_id_mapping = defaultdict(str)

# Store mapping between phone numbers and OpenAI Assistant thread IDs
assistant_thread_mapping: dict[str, str] = {}


# Track pipeline execution status for unified control plane
pipeline_status: Dict[str, object] = {
    "is_running": False,
    "stop_requested": False,
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "last_result": None,
    "last_summary": None,
    "error": None,
}
pipeline_lock = Lock()
pipeline_thread: Optional[Thread] = None


def _update_pipeline_status(**kwargs):
    """Thread-safe helper to update global pipeline status dictionary."""
    with pipeline_lock:
        pipeline_status.update(kwargs)


# Primary VIN pattern: strict 17-character match with word boundaries
VIN_PATTERN_STRICT = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
# Secondary VIN pattern: more lenient, handles OCR artifacts better
# This pattern looks for 17 alphanumeric chars (excluding I, O, Q) that may have
# spaces/dashes/underscores between them, which OCR sometimes introduces
VIN_PATTERN_LENIENT = re.compile(r"([A-HJ-NPR-Z0-9][\s\-_]?){16}[A-HJ-NPR-Z0-9]")
# Pattern to find VIN keywords for context-aware extraction
VIN_KEYWORD_PATTERN = re.compile(r'\b(?:VIN|VIN#|VIN:|V\.I\.N\.?|Vehicle\s+Identification\s+Number)\s*:?\s*', re.IGNORECASE)


def _preprocess_for_vin_opencv(image_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Enhanced preprocessing specifically optimized for VIN extraction using OpenCV.
    
    This function implements the same advanced preprocessing techniques from the OCR test script:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) for varying lighting
    - Denoising to remove image noise
    - Sharpening kernel to enhance text edges
    - Adaptive thresholding for better text extraction
    - Otsu's thresholding as alternative
    
    Args:
        image_bytes: Raw bytes of the image file
        
    Returns:
        List of tuples (variant_name, processed_image_bytes) for multiple preprocessing attempts
    """
    if not CV2_AVAILABLE or Image is None:
        return []
    
    variants = []
    
    try:
        # Convert PIL Image to numpy array for OpenCV processing
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert PIL to OpenCV format (BGR)
        if pil_image.mode == 'RGBA':
            # Remove alpha channel for OpenCV
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            rgb_image.paste(pil_image, mask=pil_image.split()[3])
            pil_image = rgb_image
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        
        # Variant 1: CLAHE + Denoising + Sharpening + Adaptive Threshold
        # This is the primary method from the test script
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Apply sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        
        # Adaptive thresholding (better for varying lighting)
        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        pil_adaptive = Image.fromarray(adaptive_thresh)
        img_bytes_adaptive = io.BytesIO()
        pil_adaptive.save(img_bytes_adaptive, format='PNG')
        variants.append(("opencv_adaptive", img_bytes_adaptive.getvalue()))
        
        # Variant 2: CLAHE + Denoising + Sharpening + Otsu's Threshold
        _, otsu_thresh = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        pil_otsu = Image.fromarray(otsu_thresh)
        img_bytes_otsu = io.BytesIO()
        pil_otsu.save(img_bytes_otsu, format='PNG')
        variants.append(("opencv_otsu", img_bytes_otsu.getvalue()))
        
        # Variant 3: Just CLAHE + Denoising (no thresholding)
        pil_denoised = Image.fromarray(denoised)
        img_bytes_denoised = io.BytesIO()
        pil_denoised.save(img_bytes_denoised, format='PNG')
        variants.append(("opencv_denoised", img_bytes_denoised.getvalue()))
        
    except Exception as e:
        logger.debug(f"OpenCV preprocessing failed: {e}")
    
    return variants


def _validate_vin_structure(vin: str) -> bool:
    """
    Validate that a VIN candidate has realistic VIN structure.
    
    VIN structure rules:
    - Must be exactly 17 characters
    - Cannot contain I, O, Q
    - Positions 1-3 (WMI) should typically be letters (manufacturer code)
    - Should have reasonable mix of letters and digits (not all same character)
    - Should not be all digits or all letters
    
    Args:
        vin: 17-character VIN candidate to validate
        
    Returns:
        True if VIN structure looks valid, False otherwise
    """
    if not vin or len(vin) != 17:
        return False
    
    # Check for invalid characters (I, O, Q)
    if any(c in vin for c in ['I', 'O', 'Q']):
        return False
    
    # Check that it's not all the same character (obviously fake)
    if len(set(vin)) == 1:
        return False
    
    # Check that positions 1-3 (WMI) are typically letters
    # Most manufacturers use letters for WMI, though some use numbers
    wmi = vin[:3]
    if not any(c.isalpha() for c in wmi):
        # If WMI has no letters, it's suspicious but not impossible
        # Log it but don't reject
        logger.debug(f"VIN candidate has no letters in WMI: {vin}")
    
    # Check for reasonable letter/digit distribution
    # VINs typically have both letters and digits
    has_letters = any(c.isalpha() for c in vin)
    has_digits = any(c.isdigit() for c in vin)
    
    # Most VINs have both letters and digits
    if not (has_letters and has_digits):
        logger.debug(f"VIN candidate lacks letter/digit mix: {vin}")
        # Don't reject, but log for debugging
    
    # Check that it's not obviously a pattern (like all 0s or all 1s)
    digit_count = sum(1 for c in vin if c.isdigit())
    if digit_count > 15:  # Too many digits, likely not a real VIN
        logger.debug(f"VIN candidate has too many digits: {vin}")
        return False
    
    return True


def _correct_vin_ocr_errors(vin_candidate: str) -> str:
    """
    Correct common OCR errors in VIN candidates.
    
    Common OCR mistakes:
    - I (letter) confused with 1 (digit)
    - O (letter) confused with 0 (digit)
    - Q (letter) confused with 0 or O
    
    Uses heuristics: if surrounded by digits, likely a digit; if surrounded by letters, likely a letter.
    
    Args:
        vin_candidate: 17-character string that might contain OCR errors
        
    Returns:
        Corrected VIN string
    """
    if len(vin_candidate) != 17:
        return vin_candidate
    
    corrected = list(vin_candidate.upper())
    
    for i, char in enumerate(corrected):
        # Check if character is potentially misread
        if char in ['I', 'O', 'Q']:
            # Check surrounding characters for context
            prev_char = corrected[i-1] if i > 0 else None
            next_char = corrected[i+1] if i < len(corrected)-1 else None
            
            # If surrounded by digits, likely a digit
            if (prev_char and prev_char.isdigit() and next_char and next_char.isdigit()):
                if char == 'I':
                    corrected[i] = '1'
                elif char == 'O' or char == 'Q':
                    corrected[i] = '0'
            # If surrounded by letters, keep as letter (but Q should be corrected to O in VIN context)
            elif (prev_char and prev_char.isalpha() and next_char and next_char.isalpha()):
                if char == 'Q':
                    # Q is not valid in VINs, likely should be O
                    corrected[i] = 'O'
            # If one side is digit and other is letter, use position-based heuristic
            # VIN positions 1-3 are typically letters (WMI), positions 4-8 can be mixed
            elif i < 3 and char in ['I', 'O']:
                # First 3 positions are usually letters, but I/O confusion with 1/0
                # Keep as letter for now
                pass
            elif i >= 9 and char in ['I', 'O']:
                # Later positions more likely to be digits
                if char == 'I':
                    corrected[i] = '1'
                elif char == 'O':
                    corrected[i] = '0'
    
    return ''.join(corrected)


def _find_vin_in_text(text: str) -> Optional[str]:
    """
    Try to locate a valid-looking VIN within text with enhanced error correction and strict validation.

    This function uses multiple strategies to find VINs, prioritizing context-aware extraction:
    1. Look for VINs near keywords (VIN, VIN#, VIN:, etc.) - HIGHEST PRIORITY
    2. Normalize text and match strict pattern with validation
    3. Try lenient pattern that handles OCR spacing artifacts with validation
    4. Extract and validate 17-character sequences with OCR error correction (only if no keyword matches)

    Notes:
        - VINs are 17 characters and exclude I, O, Q (to avoid confusion with 1/0).
        - OCR may misread I as 1, O as 0, Q as O or 0.
        - This function attempts to correct common OCR errors and validates structure.
        - VINs found near keywords are prioritized to reduce false positives.

    Args:
        text: Arbitrary text (OCR output or user message).

    Returns:
        A 17-character VIN string if found and validated; otherwise None.
    """
    if not text:
        return None

    text_upper = text.upper()
    normalized = re.sub(r"[\s\-_:]", "", text_upper)
    
    # Strategy 1: Look for VINs near keywords (HIGHEST PRIORITY - most reliable)
    # This reduces false positives by only extracting VINs in context
    keyword_matches = list(VIN_KEYWORD_PATTERN.finditer(text))
    
    for keyword_match in keyword_matches:
        # Extract text after the keyword (up to 30 characters to allow for spacing)
        start_pos = keyword_match.end()
        context_text = text_upper[start_pos:start_pos + 30]
        
        # Try strict pattern first in the context
        context_normalized = re.sub(r"[\s\-_:]", "", context_text)
        match = VIN_PATTERN_STRICT.search(context_normalized)
        if match:
            vin = match.group(1)
            if _validate_vin_structure(vin):
                logger.info(f"VIN found near keyword using strict pattern: {vin}")
                return vin
        
        # Try lenient pattern in context
        match_lenient = VIN_PATTERN_LENIENT.search(context_text)
        if match_lenient:
            matched_text = match_lenient.group(0)
            normalized_vin = re.sub(r"[\s\-_:]", "", matched_text)
            if len(normalized_vin) == 17 and VIN_PATTERN_STRICT.match(normalized_vin):
                if _validate_vin_structure(normalized_vin):
                    logger.info(f"VIN found near keyword using lenient pattern: {normalized_vin}")
                    return normalized_vin
        
        # Try with OCR error correction in context
        lenient_pattern = re.compile(r'[A-Z0-9]{17}')
        matches = lenient_pattern.findall(context_normalized)
        for candidate in matches:
            corrected = _correct_vin_ocr_errors(candidate)
            if VIN_PATTERN_STRICT.match(corrected) and _validate_vin_structure(corrected):
                logger.info(f"VIN found near keyword after correction: {candidate} -> {corrected}")
                return corrected
            if VIN_PATTERN_STRICT.match(candidate) and _validate_vin_structure(candidate):
                logger.info(f"VIN found near keyword (no correction needed): {candidate}")
                return candidate

    # Strategy 2: Normalize and use strict pattern with validation (only if no keyword matches)
    match = VIN_PATTERN_STRICT.search(normalized)
    if match:
        vin = match.group(1)
        if _validate_vin_structure(vin):
            logger.info(f"VIN found using strict pattern: {vin}")
            return vin
        else:
            logger.warning(f"VIN candidate failed structure validation: {vin}")

    # Strategy 3: Try lenient pattern on original text (handles OCR spacing) with validation
    match_lenient = VIN_PATTERN_LENIENT.search(text_upper)
    if match_lenient:
        matched_text = match_lenient.group(0)
        normalized_vin = re.sub(r"[\s\-_:]", "", matched_text)
        if len(normalized_vin) == 17 and VIN_PATTERN_STRICT.match(normalized_vin):
            if _validate_vin_structure(normalized_vin):
                logger.info(f"VIN found using lenient pattern: {normalized_vin}")
                return normalized_vin
            else:
                logger.warning(f"VIN candidate from lenient pattern failed validation: {normalized_vin}")

    # Strategy 4: Look for 17-character sequences with word boundaries and validate
    # Only use this as last resort, and only with word boundaries to avoid false matches
    lenient_pattern = re.compile(r'\b[A-Z0-9]{17}\b')
    matches = lenient_pattern.findall(normalized)
    
    for candidate in matches:
        # Attempt to correct common OCR errors
        corrected = _correct_vin_ocr_errors(candidate)
        
        # Validate corrected candidate
        if VIN_PATTERN_STRICT.match(corrected) and _validate_vin_structure(corrected):
            logger.info(f"VIN found after correction: {candidate} -> {corrected}")
            return corrected
        
        # Also check if original candidate is valid (in case correction wasn't needed)
        if VIN_PATTERN_STRICT.match(candidate) and _validate_vin_structure(candidate):
            logger.info(f"VIN found (no correction needed): {candidate}")
            return candidate

    # Strategy 5: REMOVED - Sliding window approach was too error-prone
    # It would match random 17-character sequences that aren't VINs
    # If we can't find a VIN with word boundaries or context, it's likely not there

    logger.debug(f"No valid VIN found in text. Text length: {len(text)}, Sample: {text[:200]}")
    return None


def _ocr_text_from_image_bytes(image_bytes: bytes) -> Optional[str]:
    """
    Extract text from an image using OCR (if available).

    This function implements the same advanced OCR techniques from the OCR test script:
    1. OpenCV-based preprocessing (CLAHE, denoising, sharpening, adaptive thresholding)
    2. Optimized Tesseract configuration matching test script: --oem 3 --psm 6
    3. Character whitelist to reduce OCR errors
    4. Multiple preprocessing variants for maximum accuracy
    5. Falls back to PIL preprocessing if OpenCV is unavailable

    This is best-effort and intentionally defensive: OCR dependencies may be missing
    in some environments, and we don't want inbound message processing to crash.

    Args:
        image_bytes: Raw bytes for an image file.

    Returns:
        OCR'd text if available; otherwise None.
    """
    if not image_bytes:
        return None

    if Image is None or pytesseract is None:
        return None

    # Tesseract configuration optimized for VIN recognition (matching test script)
    # --oem 3 = Use default OCR engine mode
    # --psm 6 = Assume uniform block of text
    # Whitelist: A-Z (excluding I, O, Q) and 0-9
    tesseract_configs = [
        # Primary config from test script (best for VINs)
        '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789',
        # Alternative PSM modes
        '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789',
        '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789',
        # Fallback without whitelist (in case whitelist causes issues)
        '--oem 3 --psm 6',
    ]

    try:
        all_results = []
        
        # Strategy 1: Try OpenCV preprocessing first (same as test script)
        if CV2_AVAILABLE:
            opencv_variants = _preprocess_for_vin_opencv(image_bytes)
            for variant_name, processed_bytes in opencv_variants:
                try:
                    processed_image = Image.open(io.BytesIO(processed_bytes))
                    for config in tesseract_configs:
                        try:
                            text = pytesseract.image_to_string(processed_image, config=config).strip()
                            if text:
                                all_results.append((f"opencv_{variant_name}", config, text))
                                logger.debug(f"OCR (OpenCV {variant_name}): {text[:100]}")
                        except Exception as e:
                            logger.debug(f"OCR attempt failed - OpenCV {variant_name}, Error: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"OpenCV variant processing failed: {e}")
                    continue
        
        # Strategy 2: Fallback to PIL preprocessing (if OpenCV failed or unavailable)
        if not all_results or not CV2_AVAILABLE:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image)

            original_size = image.size
            preprocessing_variants = []

            # Variant 1: High contrast grayscale with sharpening
            img1 = image.convert("L")
            img1 = ImageEnhance.Contrast(img1).enhance(2.5)
            img1 = ImageEnhance.Sharpness(img1).enhance(2.5)
            preprocessing_variants.append(("pil_high_contrast_sharp", img1))

            # Variant 2: Moderate contrast with brightness adjustment
            img2 = image.convert("L")
            img2 = ImageEnhance.Brightness(img2).enhance(1.2)
            img2 = ImageEnhance.Contrast(img2).enhance(2.0)
            preprocessing_variants.append(("pil_moderate_contrast_bright", img2))

            # Variant 3: Scaled up version (helps with small text)
            if original_size[0] < 1000 or original_size[1] < 1000:
                scale_factor = max(2.0, 1500 / max(original_size))
                new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                img3 = image.convert("L").resize(new_size, Image.Resampling.LANCZOS)
                img3 = ImageEnhance.Contrast(img3).enhance(2.0)
                img3 = ImageEnhance.Sharpness(img3).enhance(2.0)
                preprocessing_variants.append(("pil_scaled_high_contrast", img3))

            # Variant 4: Denoised version
            if ImageFilter is not None:
                img4 = image.convert("L")
                img4 = img4.filter(ImageFilter.MedianFilter(size=3))
                img4 = ImageEnhance.Contrast(img4).enhance(2.5)
                preprocessing_variants.append(("pil_denoised_contrast", img4))

            # Try all PIL preprocessing variants with Tesseract configs
            for variant_name, processed_image in preprocessing_variants:
                for config in tesseract_configs:
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config).strip()
                        if text:
                            all_results.append((variant_name, config, text))
                            logger.debug(f"OCR (PIL {variant_name}): {text[:100]}")
                    except Exception as e:
                        logger.debug(f"OCR attempt failed - PIL {variant_name}, Error: {e}")
                        continue

        # Log all OCR results for debugging
        if all_results:
            logger.info(f"OCR extracted {len(all_results)} text variants from image")
            # Log first few results for debugging
            for i, (variant, config, text) in enumerate(all_results[:3]):
                logger.info(f"OCR result {i+1} ({variant}): {text[:200]}")

        # Combine all results with newlines (Tesseract might split VIN across lines)
        combined_text = "\n".join([text for _, _, text in all_results])
        
        # Log the combined text for debugging
        if combined_text:
            logger.info(f"Combined OCR text (first 500 chars): {combined_text[:500]}")

        return combined_text if combined_text else None

    except Exception as e:
        # Most common cause on Windows: tesseract.exe not on PATH for this process.
        # On Railway/Linux: tesseract should be in PATH after Dockerfile installation
        logger.warning(f"OCR failed: {e}")
        return None


def _extract_vin_from_twilio_media_urls(image_urls: list[str]) -> Optional[str]:
    """
    Download Twilio MMS media and attempt to OCR a VIN from it.

    Why this exists:
        Twilio `MediaUrl*` links are typically protected by HTTP Basic Auth using
        (Account SID, Auth Token). External services (like OpenAI) can't fetch them
        unless you proxy/authenticate, so we perform OCR locally.

    Args:
        image_urls: List of Twilio media URLs from inbound webhook payload.

    Returns:
        VIN if a valid one is found in any image; otherwise None.
    """
    if not image_urls:
        return None

    if requests is None:
        logger.warning("OCR skipped: 'requests' is not installed.")
        return None

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("OCR skipped: Twilio credentials are missing.")
        return None

    for image_url in image_urls:
        try:
            response = requests.get(
                image_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=20,
            )
            response.raise_for_status()

            ocr_text = _ocr_text_from_image_bytes(response.content) or ""
            
            # Log raw OCR output for debugging (truncated to avoid log spam)
            if ocr_text:
                logger.info(f"Raw OCR text from image (first 300 chars): {ocr_text[:300]}")
            else:
                logger.warning("OCR returned empty text from image")
            
            vin = _find_vin_in_text(ocr_text)
            if vin:
                logger.info(f"✓ Valid VIN extracted from image via OCR: {vin}")
                logger.info(f"  VIN validation passed - structure looks correct")
                return vin
            else:
                # Log when OCR text exists but no VIN found - helps debug regex matching
                if ocr_text:
                    logger.warning(f"✗ OCR extracted text but no valid VIN found. Text length: {len(ocr_text)}, Sample: {ocr_text[:200]}")
                    # Check if there are any 17-character sequences that were rejected
                    potential_vins = re.findall(r'[A-Z0-9]{17}', re.sub(r'[\s\-_:]', '', ocr_text.upper()))
                    if potential_vins:
                        logger.warning(f"  Found {len(potential_vins)} potential 17-character sequences but none passed validation: {potential_vins[:3]}")
        except Exception as e:
            logger.warning(f"Failed to OCR media URL '{image_url}': {e}")
            continue

    return None


def _extract_vin_from_uploaded_files(uploaded_files: list) -> Optional[str]:
    """
    Extract VIN from directly uploaded image files (for testing with Postman, etc.).

    This function handles multipart/form-data file uploads where images are sent
    directly in the request body rather than as URLs.

    Args:
        uploaded_files: List of file objects from Flask request.files.

    Returns:
        VIN if a valid one is found in any image; otherwise None.
    """
    if not uploaded_files:
        return None

    for file_key, file_obj in uploaded_files.items():
        if not file_obj or not file_obj.filename:
            continue

        try:
            # Read file content into bytes
            file_obj.seek(0)  # Reset file pointer
            image_bytes = file_obj.read()
            
            if not image_bytes:
                logger.warning(f"Uploaded file '{file_key}' is empty")
                continue

            logger.info(f"Processing uploaded file '{file_key}' ({len(image_bytes)} bytes)")

            # Perform OCR on the image bytes
            ocr_text = _ocr_text_from_image_bytes(image_bytes) or ""
            
            # Log raw OCR output for debugging
            if ocr_text:
                logger.info(f"Raw OCR text from uploaded file (first 300 chars): {ocr_text[:300]}")
            else:
                logger.warning("OCR returned empty text from uploaded file")
            
            # Try to find VIN in OCR text
            vin = _find_vin_in_text(ocr_text)
            if vin:
                logger.info(f"✓ Valid VIN extracted from uploaded file via OCR: {vin}")
                logger.info(f"  VIN validation passed - structure looks correct")
                return vin
            else:
                # Log when OCR text exists but no VIN found
                if ocr_text:
                    logger.warning(f"✗ OCR extracted text but no valid VIN found. Text length: {len(ocr_text)}, Sample: {ocr_text[:200]}")
                    # Check if there are any 17-character sequences that were rejected
                    potential_vins = re.findall(r'[A-Z0-9]{17}', re.sub(r'[\s\-_:]', '', ocr_text.upper()))
                    if potential_vins:
                        logger.warning(f"  Found {len(potential_vins)} potential 17-character sequences but none passed validation: {potential_vins[:3]}")
        except Exception as e:
            logger.warning(f"Failed to OCR uploaded file '{file_key}': {e}")
            continue

    return None


def _is_stop_requested() -> bool:
    """
    Check if a stop has been requested for the running pipeline.
    
    Returns:
        True if stop was requested, False otherwise
    """
    with pipeline_lock:
        return bool(pipeline_status.get("stop_requested", False))


def _start_pipeline_thread(skip_scraping: bool, skip_salesforce: bool, skip_sms: bool,
                           headless: Optional[bool], url_override: Optional[str]):
    """Launch the lead generation pipeline in a background thread."""

    def runner():
        _update_pipeline_status(
            is_running=True,
            stop_requested=False,
            last_run_started_at=datetime.utcnow().isoformat(),
            last_run_finished_at=None,
            last_result=None,
            last_summary=None,
            error=None,
        )

        try:
            payload: Dict[str, Any] = {
                "skip_scraping": skip_scraping,
                "skip_salesforce": skip_salesforce,
                "skip_sms": skip_sms,
            }

            if headless is not None:
                payload["headless"] = headless

            if url_override:
                payload["url"] = url_override

            # Pass stop check function to pipeline
            summary = run_pipeline_from_payload(payload, stop_check=_is_stop_requested)

            # Check if pipeline was stopped
            if summary.get("status") == "stopped":
                _update_pipeline_status(
                    last_result="stopped",
                    last_summary=summary,
                    error="Pipeline stopped by user request",
                )
            else:
                _update_pipeline_status(
                    last_result="success",
                    last_summary=summary,
                )
        except KeyboardInterrupt:
            logger.info("Pipeline execution interrupted by stop request")
            _update_pipeline_status(
                last_result="stopped",
                error="Pipeline stopped by user request",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline execution failed")
            _update_pipeline_status(
                last_result="failed",
                error=str(exc),
            )
        finally:
            _update_pipeline_status(
                is_running=False,
                stop_requested=False,
                last_run_finished_at=datetime.utcnow().isoformat(),
            )

    global pipeline_thread  # noqa: PLW0603
    pipeline_thread = Thread(target=runner, daemon=True)
    pipeline_thread.start()


def _parse_bool_param(value: Optional[str], default: Optional[bool] = False) -> Optional[bool]:
    """Parse typical truthy/falsey values from query or JSON parameters."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def ensure_assistant_thread(phone_number: str) -> str:
    """Return an OpenAI Assistant thread ID for the given phone number."""
    normalized_phone = format_phone_number(phone_number)

    existing_thread = assistant_thread_mapping.get(normalized_phone)
    if existing_thread:
        return existing_thread

    try:
        thread = openai_client.beta.threads.create(extra_headers=OPENAI_ASSISTANTS_HEADER)
        assistant_thread_mapping[normalized_phone] = thread.id
        logger.info(f"Created new assistant thread for {normalized_phone}")
        return thread.id
    except Exception as exc:
        logger.error(f"Failed to create assistant thread: {str(exc)}")
        raise


def wait_for_assistant_response(thread_id: str, run_id: str, timeout_seconds: int = 45) -> Optional[str]:
    """Poll the assistant run until completion and return the assistant's reply text."""
    start_time = time.time()

    while True:
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )
        status = run.status

        if status == "completed":
            messages = openai_client.beta.threads.messages.list(
                thread_id=thread_id,
                order="desc",
                limit=10,
                extra_headers=OPENAI_ASSISTANTS_HEADER,
            )
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if getattr(content, "type", None) == "text":
                            text_value = content.text.value.strip()
                            if text_value:
                                return text_value
            logger.error("Assistant run completed but no assistant message was found.")
            return None

        if status in {"queued", "in_progress"}:
            if time.time() - start_time > timeout_seconds:
                logger.error("Timed out waiting for assistant response.")
                return None
            time.sleep(1)
            continue

        if status == "requires_action":
            logger.warning("Assistant run requires action (tool invocation not supported).")
            return None

        logger.error(f"Assistant run ended with unexpected status: {status}")
        return None

def generate_ai_response(user_message, phone_number, image_urls=None):
    """
    Generate an AI-powered response using OpenAI API based on conversation history.
    Uses the system prompt from environment variables or default prompt.
    Supports both text and image messages.
    
    Args:
        user_message: The incoming message from the user (text content, may be empty for image-only)
        phone_number: The user's phone number (for conversation context)
        image_urls: Optional list of image URLs from Twilio MediaUrl parameters
    
    Returns:
        str: AI-generated response message
    """
    try:
        # Validate OpenAI API key and assistant ID are set
        if not OPENAI_API_KEY or not OPENAI_ASSISTANT_ID:
            logger.error("OpenAI configuration is incomplete. Check OPENAI API key and assistant ID.")
            return "Thank you for your message. Could you please provide the VIN number of your vehicle?"
        
        # Ensure we have a thread for this phone number
        thread_id = ensure_assistant_thread(phone_number)

        # Build message content array for OpenAI Assistants API
        # Supports both text and images
        message_content = []
        
        # Add text content if present
        if user_message and user_message.strip():
            message_content.append({
                "type": "text",
                "text": user_message
            })
        
        # Add image content if present
        if image_urls:
            for image_url in image_urls:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
            logger.info(f"Processing {len(image_urls)} image(s) for {phone_number}")
        
        # If no content at all, use a default message
        if not message_content:
            message_content.append({
                "type": "text",
                "text": "I received your message."
            })

        # Append to local history for reference / fallback
        # Store text and image info in history
        history_content = user_message if user_message else ""
        if image_urls:
            history_content += f" [Sent {len(image_urls)} image(s)]"
        conversation_history[phone_number].append({"role": "user", "content": history_content})

        # Send message to Assistant thread with content array
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )

        # Run the assistant
        run = openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=OPENAI_ASSISTANT_ID,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )

        ai_response = wait_for_assistant_response(thread_id, run.id)

        if ai_response:
            conversation_history[phone_number].append({"role": "assistant", "content": ai_response})
            if len(conversation_history[phone_number]) > 10:
                conversation_history[phone_number] = conversation_history[phone_number][-10:]
            logger.info(f"Generated AI response for {phone_number}: {ai_response[:50]}...")
            return ai_response
        
        # Fallback if assistant response could not be retrieved
        logger.error("Assistant did not return a response; using fallback message.")
        return "Thank you for your message. Could you please provide the VIN number of your vehicle?"
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        # Fallback response if OpenAI API fails
        return "Thank you for your message. Could you please provide the VIN number of your vehicle?"

def process_incoming_message():
    """
    Process incoming message from webhook (supports both GET and POST).
    Extracts phone number, message, thread ID, and media (images/photos), then generates AI response.
    
    Returns:
        tuple: (response_xml, status_code, headers) or error response
    """
    try:
        # Get the incoming message details from Twilio (works for both GET and POST)
        # For POST: request.values.get() or request.form.get()
        # For GET: request.args.get()
        incoming_message = request.values.get('Body', request.args.get('Body', '')).strip()
        sender_phone = request.values.get('From', request.args.get('From', '')).strip()
        
        # Check for media (images/photos) in the message
        # Support both Twilio MediaUrl format and direct file uploads (for Postman testing)
        num_media = request.values.get('NumMedia', request.args.get('NumMedia', '0'))
        image_urls = []
        extracted_vin_from_image: Optional[str] = None
        
        # Check for direct file uploads (multipart/form-data) - useful for Postman testing
        uploaded_files = {}
        if request.files:
            for file_key in request.files:
                file_obj = request.files[file_key]
                if file_obj and file_obj.filename:
                    uploaded_files[file_key] = file_obj
                    logger.info(f"Found uploaded file: {file_key} ({file_obj.filename})")
        
        # Check for Twilio MediaUrl format (for production Twilio webhooks)
        try:
            num_media_int = int(num_media)
            if num_media_int > 0:
                # Extract all media URLs
                for i in range(num_media_int):
                    media_url = request.values.get(f'MediaUrl{i}', request.args.get(f'MediaUrl{i}', ''))
                    if media_url:
                        image_urls.append(media_url)
                        logger.info(f"Found media URL {i+1}/{num_media_int}: {media_url}")
        except (ValueError, TypeError):
            # If NumMedia is not a valid number, treat as 0
            num_media_int = 0
        
        # Get thread ID if provided (optional, defaults to phone number)
        thread_id_param = request.values.get('ThreadId', request.args.get('ThreadId', '')).strip()
        
        # Validate incoming data - now allow empty message if images are present
        if not incoming_message and not image_urls:
            logger.warning(f"Received empty message (no text or images) from {sender_phone}")
            response = MessagingResponse()
            response.message("I didn't receive your message. Could you please try again?")
            return str(response), 200, {'Content-Type': 'text/xml'}

        # If the user sent an image (common for VIN stickers), try to extract a VIN.
        # Important: users often include text like "VIN is in the photo" without typing the VIN.
        # We should OCR whenever images exist AND no VIN is present in the text.
        if image_urls:
            vin_in_text = _find_vin_in_text(incoming_message)
            if vin_in_text:
                logger.info(f"✓ VIN found in text message: {vin_in_text}")
                extracted_vin_from_image = vin_in_text
            if not vin_in_text:
                logger.info(f"Attempting to extract VIN from {len(image_urls)} image(s)...")
                vin_from_ocr = _extract_vin_from_twilio_media_urls(image_urls)
                if vin_from_ocr:
                    extracted_vin_from_image = vin_from_ocr
                    logger.info(f"✓ VIN extracted from image: {vin_from_ocr}")
                    
                    # Normalize phone number and get thread ID for database operations
                    normalized_sender_phone = format_phone_number(sender_phone)
                    if thread_id_param:
                        thread_id = thread_id_param
                        thread_id_mapping[normalized_sender_phone] = thread_id
                    else:
                        thread_id = thread_id_mapping.get(normalized_sender_phone, normalized_sender_phone)
                        if normalized_sender_phone not in thread_id_mapping:
                            thread_id_mapping[normalized_sender_phone] = thread_id
                    
                    # Build content description for inbound message
                    content_description = incoming_message if incoming_message else ""
                    if image_urls:
                        if content_description:
                            content_description += f" [Attached {len(image_urls)} image(s) via URL]"
                        else:
                            content_description = f"[Sent {len(image_urls)} image(s) via URL]"
                    
                    # Save inbound message to database
                    save_message(
                        thread_id=thread_id,
                        phone_number=normalized_sender_phone,
                        message_type='inbound',
                        role='user',
                        content=content_description
                    )
                    
                    # Send immediate response with extracted VIN
                    vin_response_message = f"I got your VIN number as {vin_from_ocr} is that correct?"
                    response = MessagingResponse()
                    response.message(vin_response_message)
                    
                    # Save outbound message to database
                    save_message(
                        thread_id=thread_id,
                        phone_number=normalized_sender_phone,
                        message_type='outbound',
                        role='assistant',
                        content=vin_response_message
                    )
                    
                    logger.info(f"Sending VIN confirmation message - Thread ID: {thread_id}, To: {normalized_sender_phone}, VIN: {vin_from_ocr}")
                    return str(response), 200, {'Content-Type': 'text/xml'}
                else:
                    logger.warning(f"✗ Could not extract valid VIN from image(s)")
                    response = MessagingResponse()
                    response.message(
                        "I couldn’t read the VIN from that photo. Can you please type the 17‑digit VIN, or send a clearer close-up of the VIN plate/sticker?"
                    )
                    return str(response), 200, {'Content-Type': 'text/xml'}
        
        # Handle uploaded files (for Postman testing or direct file uploads)
        if uploaded_files and not extracted_vin_from_image:
            logger.info(f"Attempting to extract VIN from {len(uploaded_files)} uploaded file(s)...")
            vin_from_uploaded = _extract_vin_from_uploaded_files(uploaded_files)
            if vin_from_uploaded:
                extracted_vin_from_image = vin_from_uploaded
                logger.info(f"✓ VIN extracted from uploaded file: {vin_from_uploaded}")
                
                # Normalize phone number and get thread ID for database operations
                normalized_sender_phone = format_phone_number(sender_phone)
                if thread_id_param:
                    thread_id = thread_id_param
                    thread_id_mapping[normalized_sender_phone] = thread_id
                else:
                    thread_id = thread_id_mapping.get(normalized_sender_phone, normalized_sender_phone)
                    if normalized_sender_phone not in thread_id_mapping:
                        thread_id_mapping[normalized_sender_phone] = thread_id
                
                # Build content description for inbound message
                content_description = incoming_message if incoming_message else ""
                if uploaded_files:
                    if content_description:
                        content_description += f" [Uploaded {len(uploaded_files)} file(s)]"
                    else:
                        content_description = f"[Uploaded {len(uploaded_files)} file(s)]"
                
                # Save inbound message to database
                save_message(
                    thread_id=thread_id,
                    phone_number=normalized_sender_phone,
                    message_type='inbound',
                    role='user',
                    content=content_description
                )
                
                # Send immediate response with extracted VIN
                vin_response_message = f"I got your VIN number as {vin_from_uploaded} is that correct?"
                response = MessagingResponse()
                response.message(vin_response_message)
                
                # Save outbound message to database
                save_message(
                    thread_id=thread_id,
                    phone_number=normalized_sender_phone,
                    message_type='outbound',
                    role='assistant',
                    content=vin_response_message
                )
                
                logger.info(f"Sending VIN confirmation message - Thread ID: {thread_id}, To: {normalized_sender_phone}, VIN: {vin_from_uploaded}")
                return str(response), 200, {'Content-Type': 'text/xml'}
        
        # Also check for VIN in text-only messages
        if not image_urls and not uploaded_files and incoming_message:
            vin_in_text = _find_vin_in_text(incoming_message)
            if vin_in_text:
                logger.info(f"✓ VIN found in text-only message: {vin_in_text}")
                extracted_vin_from_image = vin_in_text
        
        if not sender_phone:
            logger.warning("Received message without sender phone number")
            response = MessagingResponse()
            response.message("Sorry, I couldn't identify the sender.")
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        # Normalize phone number format to match format used in send_sms (E.164 format)
        # This ensures conversation history lookup works correctly
        normalized_sender_phone = format_phone_number(sender_phone)
        
        # Determine thread ID: use provided ThreadId, or phone number as thread ID
        if thread_id_param:
            thread_id = thread_id_param
            # Store mapping for future reference
            thread_id_mapping[normalized_sender_phone] = thread_id
        else:
            # Use phone number as thread ID (default behavior)
            thread_id = normalized_sender_phone
            if normalized_sender_phone not in thread_id_mapping:
                thread_id_mapping[normalized_sender_phone] = thread_id
        
        # Build content description for logging and database
        content_description = incoming_message if incoming_message else ""
        if image_urls:
            if content_description:
                content_description += f" [Attached {len(image_urls)} image(s) via URL]"
            else:
                content_description = f"[Sent {len(image_urls)} image(s) via URL]"
        if uploaded_files:
            if content_description:
                content_description += f" [Uploaded {len(uploaded_files)} file(s)]"
            else:
                content_description = f"[Uploaded {len(uploaded_files)} file(s)]"
        
        logger.info(f"Received message - Thread ID: {thread_id}, From: {sender_phone} ({normalized_sender_phone}), Content: {content_description}")
        
        # Save inbound message to database (include image URLs in content if present)
        save_message(
            thread_id=thread_id,
            phone_number=normalized_sender_phone,
            message_type='inbound',
            role='user',
            content=content_description
        )
        
        # Generate AI response using OpenAI API and system prompt from .env
        # Pass image URLs if present so the AI can see the images
        # Use normalized phone number for consistent conversation history lookup
        # Thread ID is used for logging and tracking, but phone number is used for conversation history
        # Pass None for image_urls if list is empty to avoid unnecessary processing
        if extracted_vin_from_image:
            ai_response = f'I have your VIN number as "{extracted_vin_from_image}".'
        else:
            ai_response = generate_ai_response(
                incoming_message if incoming_message else "",
                normalized_sender_phone,
                image_urls=image_urls if image_urls else None
            )
        
        # Save outbound message to database
        save_message(
            thread_id=thread_id,
            phone_number=normalized_sender_phone,
            message_type='outbound',
            role='assistant',
            content=ai_response
        )
        
        # Create TwiML response to send back to the user
        response = MessagingResponse()
        response.message(ai_response)
        
        logger.info(f"Sending AI-generated response - Thread ID: {thread_id}, To: {normalized_sender_phone}, Response: {ai_response[:50]}...")
        
        return str(response), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        logger.error(f"Error processing incoming message: {str(e)}", exc_info=True)
        # Send user-friendly error response
        response = MessagingResponse()
        response.message("Sorry, I encountered an error. Please try again later.")
        return str(response), 200, {'Content-Type': 'text/xml'}  # Return 200 to avoid Twilio retries

@app.route('/', methods=['GET'])
def root():
    """Serve the frontend dashboard."""
    return app.send_static_file('index.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint."""
    return json.dumps({
        'status': 'online',
        'service': 'Craigslist Scraper with Unified Messaging',
        'endpoints': {
            '/': 'GET - Frontend dashboard',
            '/webhook': 'POST - Twilio webhook for inbound SMS',
            '/send': 'POST/GET - Send outbound SMS messages',
            '/pipeline/run': 'POST/GET - Trigger lead generation pipeline',
            '/pipeline/stop': 'POST/GET - Stop running pipeline',
            '/api/conversations': 'GET - Get all conversation threads',
            '/api/conversations/<thread_id>': 'GET - Get messages for a thread',
            '/api/conversations/<thread_id>/inbound': 'GET - Get inbound messages',
            '/api/conversations/<thread_id>/outbound': 'GET - Get outbound messages',
            '/status': 'GET - Check server status and pipeline state'
        },
        'port': os.getenv('PORT', '5001')
    }, indent=2), 200, {'Content-Type': 'application/json'}

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """
    Twilio webhook endpoint to receive incoming SMS messages.
    Supports both GET and POST methods for flexibility.
    
    GET Parameters (for testing):
        - Body: Message content
        - From: Sender phone number (required)
        - ThreadId: Optional thread ID for conversation tracking
    
    POST Parameters (from Twilio):
        - Body: Message content
        - From: Sender phone number (required)
        - ThreadId: Optional thread ID (if using Twilio Conversations API)
    
    Returns:
        TwiML XML response
    """
    logger.info(f"Webhook called via {request.method} method")
    logger.debug(f"Request args: {dict(request.args)}")
    logger.debug(f"Request values: {dict(request.values)}")
    
    return process_incoming_message()

@app.route('/send', methods=['POST', 'GET'])
def send_message_endpoint():
    """
    API endpoint to send SMS messages programmatically.
    
    GET Parameters:
        - to: Recipient phone number (required)
        - message: Message content (required)
        - thread_id: Optional thread ID for conversation tracking
    
    POST Parameters (JSON or form data):
        - to: Recipient phone number (required)
        - message: Message content (required)
        - thread_id: Optional thread ID for conversation tracking
    
    Returns:
        JSON response with send status
    """
    try:
        # Get parameters from GET or POST
        if request.method == 'GET':
            recipient = request.args.get('to', '').strip()
            message = request.args.get('message', '').strip()
            thread_id = request.args.get('thread_id', '').strip()
        else:
            # Try JSON first, then form data
            if request.is_json:
                data = request.get_json()
                recipient = data.get('to', '').strip()
                message = data.get('message', '').strip()
                thread_id = data.get('thread_id', '').strip()
            else:
                recipient = request.values.get('to', '').strip()
                message = request.values.get('message', '').strip()
                thread_id = request.values.get('thread_id', '').strip()
        
        # Validate required parameters
        if not recipient:
            return json.dumps({
                'success': False,
                'error': 'Missing required parameter: to (recipient phone number)'
            }), 400, {'Content-Type': 'application/json'}
        
        if not message:
            return json.dumps({
                'success': False,
                'error': 'Missing required parameter: message'
            }), 400, {'Content-Type': 'application/json'}
        
        # Normalize phone number
        normalized_phone = format_phone_number(recipient)
        
        # Store thread ID if provided
        if thread_id:
            thread_id_mapping[normalized_phone] = thread_id
        
        # Send the message
        success = send_sms(normalized_phone, message, initialize_conversation=True)
        
        if success:
            logger.info(f"Message sent via API - To: {normalized_phone}, Thread ID: {thread_id or normalized_phone}")
            return json.dumps({
                'success': True,
                'message': 'Message sent successfully',
                'to': normalized_phone,
                'thread_id': thread_id or normalized_phone
            }), 200, {'Content-Type': 'application/json'}
        else:
            return json.dumps({
                'success': False,
                'error': 'Failed to send message. Check logs for details.'
            }), 500, {'Content-Type': 'application/json'}
    
    except Exception as e:
        logger.error(f"Error in send_message_endpoint: {str(e)}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/pipeline/run', methods=['POST', 'GET'])
def trigger_pipeline():
    """HTTP endpoint to kick off the end-to-end lead generation pipeline."""
    with pipeline_lock:
        if pipeline_status.get("is_running"):
            return json.dumps({
                'success': False,
                'error': 'Pipeline is already running',
                'status': pipeline_status,
            }), 409, {'Content-Type': 'application/json'}

    if request.method == 'GET':
        params = request.args
    else:
        params = request.get_json() if request.is_json else request.values

    skip_scraping = _parse_bool_param(params.get('skip_scraping') if params else None)
    skip_salesforce = _parse_bool_param(params.get('skip_salesforce') if params else None)
    skip_sms = _parse_bool_param(params.get('skip_sms') if params else None)
    headless = _parse_bool_param(params.get('headless') if params else None, default=None)
    url_override = params.get('url') if params else None

    _start_pipeline_thread(
        skip_scraping=skip_scraping,
        skip_salesforce=skip_salesforce,
        skip_sms=skip_sms,
        headless=headless,
        url_override=url_override,
    )

    return json.dumps({
        'success': True,
        'message': 'Pipeline run started',
        'status': pipeline_status,
    }), 202, {'Content-Type': 'application/json'}


@app.route('/pipeline/stop', methods=['POST', 'GET'])
def stop_pipeline():
    """
    HTTP endpoint to stop the currently running pipeline.
    
    Returns:
        JSON response indicating whether the stop request was successful
    """
    with pipeline_lock:
        if not pipeline_status.get("is_running"):
            return json.dumps({
                'success': False,
                'error': 'No pipeline is currently running',
                'status': pipeline_status,
            }), 409, {'Content-Type': 'application/json'}
        
        # Set stop flag
        pipeline_status["stop_requested"] = True
        status_snapshot = dict(pipeline_status)
    
    logger.info("Stop request received for running pipeline")
    
    return json.dumps({
        'success': True,
        'message': 'Stop request sent. Pipeline will stop after current operation completes.',
        'status': status_snapshot,
    }), 200, {'Content-Type': 'application/json'}


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """
    Get all conversation threads with summary information.
    
    Returns:
        JSON response with list of all threads
    """
    try:
        threads = get_all_threads()
        logger.info(f"API /api/conversations returning {len(threads)} threads")
        return json.dumps({
            'success': True,
            'threads': threads,
            'count': len(threads)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e),
            'threads': [],
            'count': 0
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>', methods=['GET'])
def get_conversation_by_thread(thread_id):
    """
    Get all messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with all messages in the thread
    """
    try:
        messages = get_conversations_by_thread(thread_id)
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching conversation by thread: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>/inbound', methods=['GET'])
def get_inbound_messages(thread_id):
    """
    Get all inbound messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with inbound messages
    """
    try:
        messages = get_messages_by_type(thread_id, 'inbound')
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'message_type': 'inbound',
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching inbound messages: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>/outbound', methods=['GET'])
def get_outbound_messages(thread_id):
    """
    Get all outbound messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with outbound messages
    """
    try:
        messages = get_messages_by_type(thread_id, 'outbound')
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'message_type': 'outbound',
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching outbound messages: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/debug/conversations', methods=['GET'])
def debug_conversations():
    """
    Debug endpoint to check database status and raw data.
    """
    try:
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM conversations")
        count = cursor.fetchone()
        
        # Get sample records
        cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC LIMIT 5")
        samples = cursor.fetchall()
        
        # Get thread summary
        cursor.execute("""
            SELECT thread_id, COUNT(*) as cnt 
            FROM conversations 
            GROUP BY thread_id 
            LIMIT 10
        """)
        threads = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return json.dumps({
            'success': True,
            'total_messages': count.get('total', 0) if count else 0,
            'sample_messages': samples,
            'thread_summary': threads
        }, indent=2, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/status', methods=['GET'])
def status():
    """
    Status endpoint to check webhook server health and view thread information.
    
    GET Parameters (optional):
        - phone: Phone number to get thread ID for
    
    Returns:
        JSON response with server status and thread information
    """
    try:
        phone_param = request.args.get('phone', '').strip()
        
        with pipeline_lock:
            pipeline_snapshot = dict(pipeline_status)

        status_info = {
            'status': 'online',
            'server_time': datetime.now().isoformat(),
            'total_conversations': len(conversation_history),
            'total_threads': len(thread_id_mapping),
            'pipeline': pipeline_snapshot,
        }
        
        # If phone number provided, return thread ID and conversation info
        if phone_param:
            normalized_phone = format_phone_number(phone_param)
            thread_id = thread_id_mapping.get(normalized_phone, normalized_phone)
            conversation_count = len(conversation_history.get(normalized_phone, []))
            
            status_info['phone'] = normalized_phone
            status_info['thread_id'] = thread_id
            status_info['message_count'] = conversation_count
        
        return json.dumps(status_info, indent=2), 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return json.dumps({'status': 'error', 'message': str(e)}), 500, {'Content-Type': 'application/json'}

def get_thread_id(phone_number):
    """
    Get thread ID for a given phone number.
    
    Args:
        phone_number: Phone number to get thread ID for
    
    Returns:
        str: Thread ID for the phone number
    """
    normalized_phone = format_phone_number(phone_number)
    return thread_id_mapping.get(normalized_phone, normalized_phone)

def read_phone_numbers(csv_file):
    """
    Reads phone numbers from a CSV file.
    Assumes phone numbers are in the first column or a column named 'phone' or 'phone_number'
    """
    phone_numbers = []
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        # Try to find the phone number column
        headers = csv_reader.fieldnames
        phone_column = None
        
        for header in headers:
            if header.lower() in ['phone', 'phone_number', 'phonenumber', 'number']:
                phone_column = header
                break
        
        if not phone_column:
            # If no standard column name found, use the first column
            phone_column = headers[0]
        
        print(f"Reading phone numbers from column: '{phone_column}'")
        
        for row in csv_reader:
            phone = row[phone_column].strip()
            if phone:  # Only add non-empty phone numbers
                phone_numbers.append(phone)
    
    return phone_numbers

def format_phone_number(phone):
    """
    Ensures phone number is in E.164 format (+1XXXXXXXXXX)
    """
    # Remove all non-numeric characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If it doesn't start with +, add +1 for US numbers
    if not cleaned.startswith('+'):
        if len(cleaned) == 10:
            cleaned = '+1' + cleaned
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            cleaned = '+' + cleaned
        else:
            cleaned = '+1' + cleaned
    
    return cleaned

def send_sms(to_number, message, initialize_conversation=True):
    """
    Sends an SMS message using Twilio.
    
    Args:
        to_number: Phone number to send message to
        message: Message content to send
        initialize_conversation: If True, adds the message to conversation history
    
    Returns:
        bool: True if message sent successfully, False otherwise
    """
    try:
        # Validate Twilio phone number is set and not a placeholder
        if not TWILIO_PHONE_NUMBER or TWILIO_PHONE_NUMBER in ['+1234567890', 'your_twilio_phone_number_here']:
            error_msg = "TWILIO_PHONE_NUMBER is not properly set in .env file. "
            error_msg += f"Current value: '{TWILIO_PHONE_NUMBER}'. "
            error_msg += "Please update your .env file with your actual Twilio phone number."
            logger.error(error_msg)
            print(f"✗ {error_msg}")
            return False
        
        formatted_number = format_phone_number(to_number)
        
        message_obj = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=formatted_number
        )
        
        # Initialize conversation history with the initial message sent
        if initialize_conversation:
            # Normalize phone number format for consistent lookup
            normalized_phone = formatted_number
            # Get or create thread ID
            thread_id = thread_id_mapping.get(normalized_phone, normalized_phone)
            
            # Save outbound message to database
            save_message(
                thread_id=thread_id,
                phone_number=normalized_phone,
                message_type='outbound',
                role='assistant',
                content=message
            )
            
            # Only initialize if conversation doesn't exist yet
            if normalized_phone not in conversation_history or len(conversation_history[normalized_phone]) == 0:
                conversation_history[normalized_phone].append({
                    "role": "assistant",
                    "content": message
                })
                logger.info(f"Initialized conversation history for {formatted_number}")
        
        print(f"✓ Message sent to {formatted_number} - SID: {message_obj.sid}")
        return True
    
    except Exception as e:
        print(f"✗ Failed to send message to {to_number}: {str(e)}")
        logger.error(f"Error sending SMS: {str(e)}")
        return False

def main():
    """
    Main function to read CSV and send messages
    """
    csv_file = 'extracted_phones.csv'  # Change this to your CSV file name
    
    print("Starting SMS Bot...")
    print(f"Message: {INITIAL_MESSAGE}\n")
    
    # Read phone numbers from CSV
    try:
        phone_numbers = read_phone_numbers(csv_file)
        print(f"Found {len(phone_numbers)} phone numbers\n")
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        return
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return
    
    # Send messages
    success_count = 0
    for i, phone in enumerate(phone_numbers, 1):
        print(f"[{i}/{len(phone_numbers)}] Sending to {phone}...")
        
        if send_sms(phone, INITIAL_MESSAGE):
            success_count += 1
        
        # Add delay between messages to avoid rate limiting
        if i < len(phone_numbers):
            time.sleep(1)  # Wait 1 second between messages
    
    print(f"\n=== Summary ===")
    print(f"Total numbers: {len(phone_numbers)}")
    print(f"Successfully sent: {success_count}")
    print(f"Failed: {len(phone_numbers) - success_count}")

def run_webhook_server(host='0.0.0.0', port=5001, debug=False):
    """
    Run the Flask webhook server to receive incoming messages.
    This function starts the server that listens for Twilio webhooks.
    When users reply to messages, Twilio sends POST requests to /webhook endpoint,
    which then uses OpenAI API (with prompt from .env) to generate intelligent responses.
    The same server now exposes /pipeline/run to kick off the scraping → qualification
    pipeline, keeping messaging and lead generation in one process.
    
    Args:
        host: Host address to bind to (default: 0.0.0.0 for all interfaces)
        port: Port number to listen on (default: 5001)
        debug: Enable Flask debug mode (default: False)
    
    Note:
        You need to configure your Twilio phone number's webhook URL to point to:
        https://your-domain.com/webhook (for production)
        Or use ngrok for local development: ngrok http 5001
    """
    # Initialize database on server start
    try:
        logger.info("Initializing database...")
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        logger.warning("Server will continue but database features may not work")
    
    logger.info("=" * 60)
    logger.info(f"Starting webhook server on {host}:{port}")
    logger.info("The bot is now listening for incoming messages...")
    logger.info(f"System Prompt loaded: {SYSTEM_PROMPT[:50]}...")
    logger.info("Make sure your Twilio webhook URL is configured to point to this server")
    logger.info("=" * 60)
    app.run(host=host, port=port, debug=debug)

# Wrap Flask app with ASGI adapter for uvicorn compatibility
# This allows Flask (WSGI) to work with uvicorn (ASGI server)
if ASGI_AVAILABLE:
    asgi_app = WsgiToAsgi(app)
else:
    # Create a lazy wrapper that raises helpful error when accessed
    class ASGIWrapper:
        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app
        
        async def __call__(self, scope, receive, send):
            raise ImportError(
                "asgiref is not installed. Install it with: pip install asgiref\n"
                "This is required to run Flask with uvicorn."
            )
    
    asgi_app = ASGIWrapper(app)

if __name__ == "__main__":
    import sys
    
    # Check command line arguments to determine mode
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Run webhook server mode
        # Railway sets PORT, WEBHOOK_PORT, or default to 5001
        port = int(os.getenv('PORT') or os.getenv('WEBHOOK_PORT', '5001'))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        run_webhook_server(port=port, debug=debug)
    else:
        # Run main function to send initial messages
        main()