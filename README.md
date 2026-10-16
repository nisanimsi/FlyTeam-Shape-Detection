# FlyTeam | Real-Time Shape Detection

**Real-Time Shape Detection and Relative Position Estimation Using OpenCV**

A Python-based computer vision application designed to detect red equilateral triangles and blue regular hexagons in real time using a webcam. The system identifies the target, calculates its centroid, and determines its horizontal and vertical displacement relative to the camera frame center.

The application uses classical image processing techniques without relying on pretrained machine learning models.

## Overview

The system processes live camera frames to identify one of two predefined targets:

- **Red equilateral triangle**
- **Blue regular hexagon**

Targets are presented individually. For each detected target, the application calculates its position relative to the camera center and visualizes the result directly on the video feed.

## Key Features

- Real-time webcam image acquisition
- HSV-based color segmentation
- Morphological noise reduction
- Contour extraction and area filtering
- Polygon approximation and geometric shape classification
- Centroid estimation using image moments
- Relative position calculation in pixels
- Live visualization of target boundaries, center points, connecting lines, and directional offsets

## Technical Approach

The detection pipeline consists of five stages.

| Stage | Method | Purpose |
|---|---|---|
| Image acquisition | OpenCV `VideoCapture` | Capture live webcam frames |
| Color segmentation | HSV thresholding | Isolate red and blue regions |
| Contour processing | Morphological operations, `findContours` | Reduce noise and identify candidate regions |
| Shape classification | `approxPolyDP`, vertex count, edge ratios | Distinguish triangles and hexagons |
| Position estimation | Image moments, coordinate differences | Calculate target center and relative displacement |

### Position Calculation

The camera center is calculated from the frame dimensions:

\[
C_{camera} = \left(\frac{W}{2}, \frac{H}{2}\right)
\]

The target centroid is estimated using contour moments:

\[
C_{target} = \left(\frac{M_{10}}{M_{00}}, \frac{M_{01}}{M_{00}}\right)
\]

Horizontal and vertical offsets are calculated as:

\[
\Delta x = x_{target} - x_{camera}
\]

\[
\Delta y = y_{camera} - y_{target}
\]

A positive horizontal offset indicates **RIGHT**, while a positive vertical offset indicates **UP**. Negative values indicate **LEFT** and **DOWN**, respectively.

## Technology Stack

| Technology | Application |
|---|---|
| Python | Application development |
| OpenCV | Image acquisition, processing, and visualization |
| NumPy | Numerical operations and geometric calculations |

## Project Structure

```text
FlyTeam-Shape-Detection/
├── main.py
├── image_processing.py
├── requirements.txt
└── README.md
```

**`main.py`** — Camera initialization, frame processing, relative position calculation, and real-time visualization.

**`image_processing.py`** — HSV color segmentation, morphological filtering, contour detection, polygon approximation, shape classification, and centroid estimation.

**`requirements.txt`** — Project dependencies.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/nisanimsi/FlyTeam-Shape-Detection.git
cd FlyTeam-Shape-Detection
```

**2. Install dependencies**

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

**3. Run the application**

```bash
python main.py
```

Present a red triangle or blue hexagon to the webcam. Move the target horizontally or vertically to observe real-time changes in the calculated position.

Press `Q` to exit.

## Example Output

For a 640 × 480 camera frame:

```text
Sekil: KIRMIZI UCGEN
Sekil merkezi: (470, 150)
Kamera merkezi: (320, 240)
Sag/Sol: 150 px SAG
Yukari/Asagi: 90 px YUKARI
```

The video output also displays the detected target contour, the target centroid, the camera center, and a line connecting both points.

## Validation

The application has been tested with both target types under different positions and orientations using a webcam.

The implementation supports rotated target detection and real-time position updates. Detection reliability remains dependent on lighting conditions, image focus, target visibility, and the quality of color segmentation.

## Contributors

**Şevval Nisa Sungur** — Contour filtering, polygon approximation, geometric shape classification, and detection testing.

**Enes Görkem Gönenmiş** — HSV color segmentation, position calculation, and visualization.
