# License Plate Detection Improvements

## Overview
This document outlines the improvements made to enhance the accuracy of license plate detection in the YOLO12 Number Plate OCR system.

## Key Improvements Made

### 1. Enhanced Image Preprocessing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Improves contrast in different lighting conditions
- **Bilateral Filtering**: Reduces noise while preserving edges
- **Multiple Thresholding Methods**: Combines Gaussian and Mean adaptive thresholding
- **Morphological Operations**: Cleans up the image and removes noise
- **Image Sharpening**: Enhances text clarity for better OCR

### 2. Multiple OCR Attempts
- **Three Enhancement Methods**: 
  - Standard enhanced image
  - Inverted colors (for light text on dark background)
  - OTSU thresholding (for automatic threshold detection)
- **Best Result Selection**: Chooses the result with highest confidence
- **Lower Confidence Threshold**: Reduced from 0.5 to 0.4 for better detection

### 3. Improved Text Cleaning
- **Extended Character Corrections**: Added more common OCR misreads
  - O → 0, I → 1, S → 5, Z → 2, B → 8, G → 6
  - A → 4, E → 3, T → 7, L → 1, D → 0, Q → 0
- **Intelligent Correction Logic**: Applies corrections based on character distribution
- **Duplicate Character Removal**: Removes repeated characters (common OCR error)
- **Minimum Length Validation**: Ensures plates are at least 4 characters

### 4. Enhanced License Plate Validation
- **Multiple Pattern Recognition**:
  - Letters followed by numbers (e.g., ABC123)
  - Numbers followed by letters (e.g., 123ABC)
  - Mixed patterns (e.g., A1B2C3)
  - Standard format with minimum 2 letters and 2 numbers
- **Stricter Validation**: Ensures proper license plate format

### 5. Better YOLO Configuration
- **Lower Confidence Threshold**: Reduced from 0.5 to 0.3 for better detection
- **Improved NMS Settings**: Added IoU threshold and agnostic NMS
- **Increased Max Detections**: Set to 20 for better coverage

### 6. Enhanced PaddleOCR Configuration
- **Lower Detection Thresholds**: Better for detecting small text
- **Optimized Parameters**: 
  - `det_db_thresh=0.3`
  - `det_db_box_thresh=0.5`
  - `det_db_unclip_ratio=1.6`
- **Single Batch Processing**: For better accuracy

### 7. Debug Features
- **Debug Mode Toggle**: Enable/disable debug image saving
- **Debug Image Saving**: Saves high-confidence detections for analysis
- **Enhanced Logging**: Better error tracking and debugging

## Usage Instructions

### 1. Enable Debug Mode
- Check the "Debug Mode" checkbox in the main tab
- Debug images will be saved to `debug_images/` folder
- Only saves images with confidence > 0.6

### 2. Test the Improvements
Run the test script to verify improvements:
```bash
python test_improvements.py
```

### 3. Monitor Detection Quality
- Watch the confidence scores in the detection display
- Check debug images for visual verification
- Review logs for any errors or issues

## Expected Improvements

### Detection Accuracy
- **Better Text Recognition**: Multiple enhancement methods improve OCR accuracy
- **Reduced False Negatives**: Lower confidence thresholds catch more plates
- **Improved Character Recognition**: Enhanced corrections reduce misreads

### Robustness
- **Lighting Conditions**: CLAHE and multiple methods handle different lighting
- **Image Quality**: Better preprocessing handles low-quality images
- **Text Variations**: Multiple patterns recognize different plate formats

### Performance
- **Optimized Processing**: Better parameters balance speed and accuracy
- **Error Handling**: Improved error recovery and logging
- **Debug Capabilities**: Better troubleshooting and analysis tools

## Troubleshooting

### If Detection is Still Poor
1. **Check Debug Images**: Review saved images in `debug_images/` folder
2. **Adjust Confidence Thresholds**: Modify in the code if needed
3. **Test Different Enhancement Methods**: Use the test script
4. **Verify Model Quality**: Ensure YOLO model is properly trained

### Common Issues
- **Low Confidence Scores**: May indicate poor image quality or lighting
- **Character Misreads**: Check if corrections are working properly
- **Missing Detections**: Verify YOLO model confidence threshold

## Future Enhancements
- **Machine Learning Corrections**: Train a model for better character correction
- **Multi-frame Voting**: Use results from multiple frames for better accuracy
- **Region-specific Patterns**: Add support for different license plate formats
- **Real-time Optimization**: Further optimize for real-time processing 