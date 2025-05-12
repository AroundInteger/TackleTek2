# TackleTek2

A computer vision-based system for analyzing rugby tackles using YOLOv8 pose detection and grid-based tracking.

## Features

- Real-time pose detection using YOLOv8
- Grid-based tracking of players
- Calibration system for arena markers
- Player identification and tracking
- Height difference analysis between players

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aroundinteger/TackleTek2.git
cd TackleTek2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the sample video:
```bash
# The video file (R1_1.mp4) should be placed in the project root directory
# You can download it from the releases section of this repository
```

## Usage

1. Ensure the rugby video (R1_1.mp4) is in the project directory
2. Run the calibration script:
```bash
python detect_mkrs_bkg_grid_blobs_1.py
```

The script will:
- Process the first 10 frames for calibration
- Detect and track player poses
- Generate a visualization image showing the grid and detected poses

## Project Structure

- `detect_mkrs_bkg_grid_blobs_1.py`: Main script for pose detection and tracking
- `requirements.txt`: Python package dependencies
- `R1_1.mp4`: Sample rugby video for testing
- `calibration_visualization.jpg`: Output visualization of grid and poses

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for pose detection
- OpenCV for computer vision processing 