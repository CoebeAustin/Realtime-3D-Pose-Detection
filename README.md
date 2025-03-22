# Exersion
This Python-based application utilizes computer vision techniques mediapipe for keypoint detection and numpy for angle calculation to automatically count repetitions for curl-ups and squats in real-time. The app analyzes body movements through a web camera to track the start and completion of each repetition.

# ⭐ Features 
* Detect and count squats and curl-ups using computer vision.
* Let users choose a prompt between squats or curl-up count.
* Track and display the count on the screen.
* Counts both the left and right arms for curl-ups

# ⚙️ Technologies used 
* Python
* OpenCV
* Mediapipe

# 💡How it functions
* Detects body keypoints using MediaPipe's Pose Model
* Monitors joint angles to identify upward and downward motion
* Counts repetitions on the full range of motion and predefined threshold

## Pre-requisites
Before you can use this app, make sure you have Python installed on your system.

### 1. Install Python
To run this application, you'll need to have Python installed. Follow the instructions below to install Python:

#### For Windows:
- Visit the official Python website: https://www.python.org/downloads/
- Download the latest version of Python (make sure to choose the version that fits your system, e.g., 64-bit).
- Run the installer and ensure you check the box to add Python to the system `PATH` during installation.

#### For macOS:
- You can install Python using the [Homebrew package manager](https://brew.sh/) (if Homebrew is installed):
  ```bash
  brew install python

#### For Linux:
- For Ubuntu/Debian-based systems, you can install Python using:
  ```bash
  sudo apt update
  sudo apt install python3
- For other distributions, use your package manager to install Python. 

# Usage 
1. Clone the repository
   ```
   git clone https://github.com/CoebeAustin/Realtime-3D-Pose-Detection.git
   ```
2. Install dependencies from requirements.txt
    ```
    pip install -r requirements.txt
    ```
3. Run the application
   ```
   python exersion.py
   ```
