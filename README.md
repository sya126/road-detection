# 🛣️ Advanced Whole Road Detection & Tracking

An advanced computer vision project developed in Python that detects the entire drivable road surface. Unlike traditional lane detection systems that focus on a single ego-lane, this algorithm identifies the full road width using **geometric constraints** and **polynomial smoothing** to handle perspective distortion effectively.

![Project Demo](output/road_detection_final.mp4) 
*(Note: You can replace this link with a GIF of your result)*

## 🌟 Key Features

* **Whole Road Awareness:** Detects the boundaries of the entire drivable area (from the leftmost yellow line to the rightmost road edge), providing a complete spatial context.
* **Dynamic Horizon Safety Lock:** Solves the common "vanishing point crossing" issue. The algorithm mathematically enforces a minimum separation distance at the horizon, preventing lane lines from merging or crossing into an "X" shape.
* **Polynomial Smoothing (Anti-Kink):** Applies a secondary polynomial fit to the processed coordinates. This removes the sharp "elbows" or artifacts created by the safety clamp, ensuring the lane curvature remains smooth and organic.
* **Robust Color Filtering:** Utilizes the **HLS color space** to dynamically isolate white and yellow lane markings, making it resilient to shadow and lighting changes.

## 🛠️ How It Works

The pipeline consists of the following steps:

1.  **Preprocessing:** The image is converted to HLS space to filter yellow and white colors.
2.  **Region of Interest (ROI):** A dynamic trapezoidal mask focuses on the road area, ignoring the sky and surroundings.
3.  **Edge Detection:** Canny edge detection combined with Hough Transform identifies linear features.
4.  **Boundary Extraction:** The algorithm classifies lines into "Left" (Yellow) and "Right" (White/Barrier) boundaries.
5.  **Geometric Stabilization:**
    * *Step A:* A hard geometric clamp prevents the right lane from drifting into the left lane at the horizon.
    * *Step B:* A polynomial regression is re-applied to these clamped points to smooth out the curve.
6.  **Rendering:** The detected road surface is overlaid as a green polygon with highlighted boundaries.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/road-detection-project.git](https://github.com/your-username/road-detection-project.git)
    cd road-detection-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

3.  **Run the script:**
    ```bash
    python main.py
    ```
    *Ensure your video file is named `test_video.mp4` or update the path in `main.py`.*

## 📂 Project Structure

* `main.py`: The core logic containing the pipeline, safety spacer, and smoothing algorithms.
* `test_video.mp4`: Input video for testing.
* `output/`: Directory where the processed video is saved.

## 🧠 Technical Highlights

### The "Smart Spacer" Algorithm
Traditional perspective transforms often cause parallel lines to merge at the vanishing point due to pixel noise. This project implements a logic that:
```python
# Force a minimum width at the horizon
clamped_right_x = np.maximum(raw_right_x, left_x + MIN_WIDTH_AT_HORIZON)