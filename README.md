# Multi-Finger Tap Detection with OpenCV and MediaPipe

This project demonstrates a **real-time multi-finger tap detection system** using OpenCV and MediaPipe Hands. The system tracks finger movements and detects taps when fingers touch the thumb within a specified threshold distance. It also provides visual feedback with color-coded finger markers and a dynamic tap count display.

---

## Features

- **Real-Time Finger Tracking**: Detects finger positions in real-time using a webcam.
- **Multi-Finger Detection**: Tracks taps for the index, middle, ring, and pinky fingers.
- **Visual Display**: Shows connections, distances, and tap counts directly on the video feed.
- **Customizable**: Adjustable thresholds and cooldown periods for tailored detection.

---

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/multi-finger-tap-detection.git
   cd multi-finger-tap-detection

   ```

2. Install the required packages:
   ```bash
    pip install opencv-python mediapipe numpy
   ```

---

## Usage

1. Run the `multi_finger_tap_detection.py` script:

   ```bash
   python multi_finger_tap_detection.py

   ```

2. Adjust the `THRESHOLD` and `COOLDOWN` values in the script for optimal detection.

---

<img src="img.png"></img>
