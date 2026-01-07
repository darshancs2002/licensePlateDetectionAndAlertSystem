#!/usr/bin/env python3
"""
Test script for license plate detection deduplication improvements
"""

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
from difflib import SequenceMatcher

def test_deduplication_logic():
    """Test the deduplication logic"""
    print("Testing deduplication logic...")
    
    # Simulate the deduplication variables
    plate_tracking = {}
    last_detection_time = {}
    detection_cooldown = 30
    similarity_threshold = 0.7
    
    def is_duplicate_detection(plate_text, frame_count, box):
        """Check if this is a duplicate detection of the same plate"""
        if not plate_text or plate_text == "N/A" or len(plate_text) < 3:
            return True
        
        # Clean the plate text (simplified)
        cleaned_text = plate_text.replace(' ', '').upper()
        if not cleaned_text:
            return True
        
        # Check if we've seen this plate recently
        if cleaned_text in last_detection_time:
            frames_since_last = frame_count - last_detection_time[cleaned_text]
            if frames_since_last < detection_cooldown:
                return True
        
        # Check for similar plates (fuzzy matching)
        for existing_plate in plate_tracking.keys():
            similarity = SequenceMatcher(None, cleaned_text, existing_plate).ratio()
            if similarity > similarity_threshold:
                # Check if the existing plate was detected recently
                if existing_plate in last_detection_time:
                    frames_since_last = frame_count - last_detection_time[existing_plate]
                    if frames_since_last < detection_cooldown:
                        return True
        
        return False
    
    def update_plate_tracking(plate_text, frame_count, box, confidence):
        """Update plate tracking information"""
        cleaned_text = plate_text.replace(' ', '').upper()
        if not cleaned_text:
            return
        
        # Update tracking information
        plate_tracking[cleaned_text] = {
            'last_frame': frame_count,
            'last_box': box,
            'confidence': confidence,
            'detection_count': plate_tracking.get(cleaned_text, {}).get('detection_count', 0) + 1
        }
        
        # Update last detection time
        last_detection_time[cleaned_text] = frame_count
    
    # Test cases
    test_cases = [
        ("R-183-JF", 10, [100, 100, 200, 150], 0.8),
        ("R-183-JF", 15, [105, 105, 205, 155], 0.7),  # Should be duplicate (within cooldown)
        ("N-894-JV", 20, [300, 200, 400, 250], 0.9),
        ("R-183-JF", 50, [110, 110, 210, 160], 0.8),  # Should not be duplicate (after cooldown)
        ("R183JF", 55, [115, 115, 215, 165], 0.8),    # Should be duplicate (similar to R-183-JF)
        ("H-644-LX", 60, [500, 300, 600, 350], 0.7),
    ]
    
    print("\nTesting deduplication with video4.mp4 expected plates:")
    expected_plates = ['R183JF', 'N894JV', 'L656XH', 'H644LX', 'K884RS']
    
    for plate_text, frame_count, box, confidence in test_cases:
        is_duplicate = is_duplicate_detection(plate_text, frame_count, box)
        
        if not is_duplicate:
            update_plate_tracking(plate_text, frame_count, box, confidence)
            print(f"Frame {frame_count}: ACCEPTED '{plate_text}' (confidence: {confidence:.2f})")
        else:
            print(f"Frame {frame_count}: REJECTED '{plate_text}' (duplicate)")
    
    print(f"\nFinal tracking results:")
    print(f"Unique plates detected: {len(plate_tracking)}")
    for plate, info in plate_tracking.items():
        print(f"  {plate}: {info['detection_count']} detections, last frame {info['last_frame']}")

def test_ocr_enhancement():
    """Test OCR enhancement methods"""
    print("\nTesting OCR enhancement methods...")
    
    # This would require actual test images
    print("OCR enhancement testing requires actual license plate images.")
    print("Run the main application with debug mode enabled to test OCR improvements.")

if __name__ == "__main__":
    test_deduplication_logic()
    test_ocr_enhancement()
    print("\nDeduplication test completed!") 