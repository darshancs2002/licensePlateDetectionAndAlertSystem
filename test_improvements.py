#!/usr/bin/env python3
"""
Test script for license plate detection improvements
"""

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

def test_enhancement_methods():
    """Test different image enhancement methods"""
    print("Testing image enhancement methods...")
    
    # Load a sample image (you can replace this with your test image)
    test_image_path = "test_plate.jpg"  # Replace with your test image
    
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found. Please add a test image.")
        return
    
    # Load image
    img = cv2.imread(test_image_path)
    if img is None:
        print("Failed to load test image")
        return
    
    # Initialize OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        rec_batch_num=1,
        use_space_char=True,
        use_zero_copy_run=True
    )
    
    # Test different enhancement methods
    methods = [
        ("Original", img),
        ("Enhanced", enhance_plate_image(img)),
        ("Inverted", 255 - enhance_plate_image(img)),
        ("OTSU", cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    ]
    
    for method_name, enhanced_img in methods:
        print(f"\n--- Testing {method_name} ---")
        try:
            result = ocr.ocr(enhanced_img, cls=False)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        print(f"Detected: '{text}' (confidence: {confidence:.2f})")
            else:
                print("No text detected")
        except Exception as e:
            print(f"Error: {str(e)}")

def enhance_plate_image(img):
    """Enhanced image preprocessing function"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
        
        # Apply adaptive threshold
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Remove noise
        kernel_open = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Resize if needed
        height, width = cleaned.shape
        if height < 60:
            scale = 60 / height
            new_width = int(width * scale)
            cleaned = cv2.resize(cleaned, (new_width, 60), interpolation=cv2.INTER_CUBIC)
        
        # Sharpen
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)
        
        return sharpened
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        return img

def clean_plate_text(text):
    """Clean and normalize license plate text"""
    if not text:
        return ""
    
    # Remove special characters and spaces
    cleaned = ''.join(c for c in text.upper() if c.isalnum())
    
    # Common OCR corrections
    corrections = {
        'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6',
        'A': '4', 'E': '3', 'T': '7', 'L': '1', 'D': '0', 'Q': '0'
    }
    
    # Apply corrections
    result = ""
    for char in cleaned:
        if char in corrections:
            letter_count = sum(1 for c in cleaned if c.isalpha())
            number_count = sum(1 for c in cleaned if c.isdigit())
            
            if letter_count > number_count or char in ['O', 'I', 'S', 'Z', 'B', 'G']:
                result += corrections[char]
            else:
                result += char
        else:
            result += char
    
    # Remove repeated characters
    cleaned_result = ""
    prev_char = ""
    repeat_count = 0
    for char in result:
        if char == prev_char:
            repeat_count += 1
            if repeat_count <= 1:
                cleaned_result += char
        else:
            cleaned_result += char
            prev_char = char
            repeat_count = 0
    
    return cleaned_result[:10]

def is_valid_plate(text):
    """Validate license plate format"""
    if not text or len(text) < 4 or len(text) > 10:
        return False
    
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    
    if not (has_letter and has_number):
        return False
    
    # Check patterns
    import re
    
    # Pattern 1: Letters followed by numbers
    if re.match(r'^[A-Z]{1,4}[0-9]{1,4}$', text):
        return True
    
    # Pattern 2: Numbers followed by letters
    if re.match(r'^[0-9]{1,4}[A-Z]{1,4}$', text):
        return True
    
    # Pattern 3: Mixed pattern
    if re.match(r'^[A-Z0-9]{4,8}$', text):
        letter_count = sum(1 for c in text if c.isalpha())
        number_count = sum(1 for c in text if c.isdigit())
        if letter_count >= 2 and number_count >= 2:
            return True
    
    # Pattern 4: Standard format
    letter_count = sum(1 for c in text if c.isalpha())
    number_count = sum(1 for c in text if c.isdigit())
    
    if letter_count >= 2 and number_count >= 2:
        return True
    
    return False

if __name__ == "__main__":
    print("License Plate Detection Test Script")
    print("==================================")
    
    # Test enhancement methods
    test_enhancement_methods()
    
    # Test text cleaning
    print("\n--- Testing Text Cleaning ---")
    test_texts = [
        "ABC123",
        "A8C123",  # B -> 8
        "ABC12O",  # O -> 0
        "ABC12I",  # I -> 1
        "ABC12S",  # S -> 5
        "ABC12Z",  # Z -> 2
        "ABC12G",  # G -> 6
        "ABC12A",  # A -> 4
        "ABC12E",  # E -> 3
        "ABC12T",  # T -> 7
        "ABC12L",  # L -> 1
        "ABC12D",  # D -> 0
        "ABC12Q",  # Q -> 0
    ]
    
    for text in test_texts:
        cleaned = clean_plate_text(text)
        valid = is_valid_plate(cleaned)
        print(f"'{text}' -> '{cleaned}' (valid: {valid})")
    
    print("\nTest completed!") 