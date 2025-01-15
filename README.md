# Number Plate Detection and Extraction

This project implements a robust system for detecting vehicles and extracting license plate information from video files. It uses advanced object detection models and OCR (Optical Character Recognition) to process video frames, identify vehicles, locate license plates, and recognize license numbers.

## Concept

The idea behind this project is to automate the detection and recognition of vehicle license plates in video footage. This is useful in applications such as:
- Traffic monitoring and law enforcement
- Automated toll collection systems
- Parking management
- Vehicle tracking and security

### Core Steps in the System
1. **Vehicle Detection**: Use a YOLO (You Only Look Once) model to identify and localize vehicles in video frames.
2. **License Plate Detection**: Employ a custom-trained YOLO model to detect license plates on vehicles.
3. **License Plate Recognition**: Use OCR to extract text from detected license plates.
4. **Object Tracking**: Apply the SORT algorithm to maintain consistent identification of vehicles across video frames.
5. **Result Compilation**: Save the processed data (e.g., license plate numbers, bounding boxes) to a CSV file for analysis or reporting.

## Features
- **Vehicle Detection**: Identifies vehicles such as cars, trucks, and buses in video frames using YOLOv8.
- **License Plate Detection**: Locates license plates on detected vehicles using a custom-trained YOLO model.
- **OCR**: Reads and interprets text from license plates using EasyOCR.
- **Tracking**: Tracks vehicles across frames using the SORT algorithm.
- **Results Export**: Saves processed data, including bounding boxes and license numbers, to a CSV file.

## Project Structure
```
Number Plate Detection/
├── add_missing_data.py      # Handles data preprocessing (if applicable)
├── license_plate_detector.pt # Pre-trained YOLO model for license plate detection
├── main.py                  # Main script to run the detection system
├── test.csv                 # Sample dataset
├── test_interpolated.csv    # Preprocessed dataset
├── util.py                  # Utility functions for OCR and CSV writing
├── visualize.py             # Visualization functions for detected objects
├── yolov8n.pt               # Pre-trained YOLOv8 model for vehicle detection
├── sort/                    # SORT algorithm implementation
└── sample.mp4               # Sample video file for testing (if included)
```

## Requirements
- Python 3.8+
- Required Python packages:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `easyocr`
  - `csv`

## Usage

### Step 1: Setup
Ensure the following files are in the project directory:
- `yolov8n.pt`: Pre-trained YOLOv8 model for vehicle detection.
- `license_plate_detector.pt`: Custom-trained YOLO model for license plate detection.

### Step 2: Run the Detection System
Run the `main.py` script:
```bash
python main.py
```
This script processes the video file, detects vehicles and license plates, and exports results to a CSV file.

### Step 3: View Results
The processed data is saved in `results.csv`, containing:
- Frame number
- Vehicle ID
- Bounding box coordinates for vehicles and license plates
- License plate text and confidence scores

## Key Components

### 1. `main.py`
- Loads YOLO models for vehicle and license plate detection.
- Processes video frames to detect and track vehicles.
- Extracts license plate text using OCR.

### 2. `util.py`
- Implements OCR functionality using EasyOCR.
- Provides a `write_csv` function to save results.

### 3. `visualize.py`
- Contains helper functions to visualize detection results on video frames.

### 4. `sort/`
- Implements the SORT algorithm for real-time object tracking.

## Implementation Details

1. **Vehicle Detection**:
   - YOLOv8 model is used to detect vehicles in each video frame.
   - Vehicles of interest (cars, trucks, buses) are filtered based on their class IDs.

2. **License Plate Detection**:
   - A custom YOLO model detects license plates on the detected vehicles.
   - Bounding boxes for license plates are extracted for further processing.

3. **OCR for License Plate Recognition**:
   - EasyOCR processes the license plate regions to extract text.
   - The text is cleaned and corrected using character mapping to handle common OCR errors.

4. **Tracking with SORT**:
   - SORT (Simple Online and Realtime Tracking) assigns consistent IDs to vehicles across frames.
   - Ensures that license plate data is associated with the correct vehicle.

5. **Output Results**:
   - Results are saved to a CSV file containing detailed information for each vehicle and license plate.
   - Includes frame number, bounding boxes, and extracted text with confidence scores.

## Customization
- **Training Custom Models**: Replace `license_plate_detector.pt` with a retrained YOLO model for different license plate styles or regions.
- **Dataset**: Update `test.csv` or use your own dataset for testing.

## Acknowledgments
- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [SORT Algorithm](https://github.com/abewley/sort)

## License
This project is licensed under the Apache License 2.0. See `LICENSE` for details.
