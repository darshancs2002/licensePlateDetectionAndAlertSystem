import cv2
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
from datetime import datetime
import numpy as np
import re

# Load YOLO and PaddleOCR
model = YOLO('best.pt')  # Replace with your trained model
names = model.names
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'  # Specify language for better accuracy
)

# Prepare log file
now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f"plates_{now_str}.txt"
saved_ids = set()
id_to_plate = {}
id_confidence_scores = {}  # Track confidence scores

def clean_plate_text(text):
    """Clean and validate plate text"""
    if not text:
        return ""
    
    # Remove extra spaces and clean text
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Keep only alphanumeric characters and spaces
    cleaned = re.sub(r'[^A-Za-z0-9\s]', '', cleaned)
    return cleaned.upper()

def is_valid_plate(text, min_length=4, max_length=10):
    """Basic validation for plate text"""
    if not text:
        return False
    
    cleaned = text.replace(' ', '')
    if len(cleaned) < min_length or len(cleaned) > max_length:
        return False
    
    # Should contain both letters and numbers (basic check)
    has_letter = any(c.isalpha() for c in cleaned)
    has_number = any(c.isdigit() for c in cleaned)
    
    return has_letter and has_number

def enhance_plate_image(plate_img):
    """Enhance plate image for better OCR"""
    if plate_img.size == 0:
        return plate_img
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to BGR for OCR
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Debug mouse callback
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Initialize video capture
cap = cv2.VideoCapture("vid1.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames at {fps} FPS")

frame_count = 0
processed_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
            
        frame_count += 1
        
        # Process every 3rd frame for efficiency
        if frame_count % 3 != 0:
            continue
            
        processed_count += 1
        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLO tracking
        results = model.track(frame, persist=True, conf=0.5)  # Added confidence threshold
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for track_id, box, class_id, conf in zip(ids, boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                label = names[class_id]
                
                # Draw bounding box
                color = (255, 0, 0) if label.lower() == "numberplate" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display class label and confidence
                conf_text = f'{label.upper()} {conf:.2f}'
                cvzone.putTextRect(frame, conf_text, (x1, y1 - 10), scale=1, thickness=2,
                                 colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5, border=2)
                
                if label.lower() == "numberplate":
                    # Crop the plate region with some padding
                    padding = 5
                    y1_crop = max(0, y1 - padding)
                    y2_crop = min(frame.shape[0], y2 + padding)
                    x1_crop = max(0, x1 - padding)
                    x2_crop = min(frame.shape[1], x2 + padding)
                    
                    cropped_plate = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    if cropped_plate.size == 0:
                        continue
                    
                    # Only process OCR if we haven't got a good result for this ID
                    if track_id not in id_to_plate or id_confidence_scores.get(track_id, 0) < 0.8:
                        try:
                            # Enhance the plate image
                            enhanced_plate = enhance_plate_image(cropped_plate)
                            
                            # Run OCR on enhanced image
                            result = ocr.ocr(enhanced_plate, cls=False)
                            
                            plate_text = ""
                            ocr_confidence = 0
                            
                            # Parse OCR results (correct PaddleOCR format)
                            if result and result[0]:
                                texts_with_conf = []
                                for line in result[0]:
                                    if line and len(line) >= 2:
                                        text = line[1][0]  # text
                                        confidence = line[1][1]  # confidence
                                        texts_with_conf.append((text, confidence))
                                
                                if texts_with_conf:
                                    # Combine all text and calculate average confidence
                                    plate_text = " ".join([text for text, _ in texts_with_conf])
                                    ocr_confidence = sum([conf for _, conf in texts_with_conf]) / len(texts_with_conf)
                                    
                                    print(f"OCR Result: '{plate_text}' (confidence: {ocr_confidence:.3f})")
                            
                            # Clean and validate the plate text
                            cleaned_text = clean_plate_text(plate_text)
                            
                            if cleaned_text and is_valid_plate(cleaned_text) and ocr_confidence > 0.5:
                                # Update if this is a better result
                                if track_id not in id_to_plate or ocr_confidence > id_confidence_scores.get(track_id, 0):
                                    id_to_plate[track_id] = cleaned_text
                                    id_confidence_scores[track_id] = ocr_confidence
                                    
                                    # Save to log file only once per ID
                                    if track_id not in saved_ids:
                                        with open(log_filename, "a", encoding='utf-8') as f:
                                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            f.write(f"{timestamp} | ID: {track_id} | Plate: {cleaned_text} | Confidence: {ocr_confidence:.3f}\n")
                                        saved_ids.add(track_id)
                                        print(f"Saved: ID={track_id}, Plate={cleaned_text}, Conf={ocr_confidence:.3f}")
                        
                        except Exception as e:
                            print(f"OCR Error for ID {track_id}: {e}")
                            continue
                    
                    # Display plate text on frame
                    if track_id in id_to_plate:
                        plate_text = id_to_plate[track_id]
                        confidence = id_confidence_scores.get(track_id, 0)
                        display_text = f"ID: {track_id} | {plate_text} ({confidence:.2f})"
                        cvzone.putTextRect(frame, display_text, (x1, y2 + 10), scale=0.8, thickness=2,
                                         colorT=(0, 0, 0), colorR=(255, 255, 0), offset=5, border=1)
                    else:
                        cvzone.putTextRect(frame, f"ID: {track_id}", (x1, y2 + 10), scale=0.8, thickness=2,
                                         colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5, border=1)
        
        # Display frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Processed: {processed_count} | Plates Found: {len(id_to_plate)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("RGB", frame)
        
        # Use cv2.waitKey(1) for continuous playback, ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # Press 's' to save current frame
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)
            print(f"Saved frame_{frame_count}.jpg")

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {processed_count}")
    print(f"Unique plates detected: {len(id_to_plate)}")
    print(f"Log file: {log_filename}")
    
    if id_to_plate:
        print("\nDetected plates:")
        for track_id, plate in id_to_plate.items():
            conf = id_confidence_scores.get(track_id, 0)
            print(f"  ID {track_id}: {plate} (confidence: {conf:.3f})")
