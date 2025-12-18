import cv2
import numpy as np
import os
from collections import deque

# --- CONFIGURATION / AYARLAR ---
INPUT_VIDEO_PATH = "test_video.mp4"
OUTPUT_VIDEO_PATH = "output/road_detection_final.mp4"

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# History Deques (To stabilize jitter / Titremeyi önlemek için hafıza)
CACHE_SIZE = 15
left_fit_history = deque(maxlen=CACHE_SIZE)
right_fit_history = deque(maxlen=CACHE_SIZE)

def select_white_yellow(image):
    """
    Applies color masking to isolate white and yellow lane markings.
    """
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # White Color Mask
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    
    # Yellow Color Mask
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)
    
    # Combine masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def get_mean_fit(history):
    if len(history) > 0:
        return np.mean(history, axis=0)
    return None

def find_road_boundaries(image, lines):
    """
    Analyzes Hough Lines to find the absolute left and right boundaries of the road.
    """
    height, width = image.shape[:2]
    center_x = width * 0.5
    
    left_x, left_y = [], []
    right_x, right_y = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 == x1: continue # Ignore vertical lines to prevent division by zero
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out near-horizontal lines
            if abs(slope) < 0.3: continue
            
            # Left Boundary Candidates (Negative slope, left side of screen)
            if slope < 0 and x1 < center_x:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
                
            # Right Boundary Candidates (Positive slope, right side of screen)
            elif slope > 0 and x1 > center_x * 0.2: 
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 1. Fit Polynomial for Left Boundary
    if len(left_x) > 0:
        left_fit = np.polyfit(left_y, left_x, 2)
        left_fit_history.append(left_fit)
        final_left = np.mean(left_fit_history, axis=0)
    else:
        final_left = get_mean_fit(left_fit_history)

    # 2. Fit Polynomial for Right Boundary
    if len(right_x) > 0:
        right_fit = np.polyfit(right_y, right_x, 2)
        right_fit_history.append(right_fit)
        final_right = np.mean(right_fit_history, axis=0)
    else:
        final_right = get_mean_fit(right_fit_history)

    return final_left, final_right

def draw_whole_road(image, left_fit, right_fit):
    """
    Draws the detected road surface with a dynamic 'Safety Spacer' to prevent crossing at the horizon.
    """
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)
    
    # Generate y values for plotting
    plot_y = np.linspace(int(height * 0.55), height, num=height)
    
    try:
        if left_fit is None or right_fit is None: return image
            
        # 1. Calculate Raw X values
        left_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        raw_right_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
        
        # --- GITHUB VERSION: SMOOTH SAFETY SPACER ---
        
        # Step A: Hard Clamp (Zorla Açma)
        # Prevent the right lane from crossing the left lane.
        # Increased buffer to 90px to allow room for smoothing without merging.
        MIN_WIDTH_AT_HORIZON = 90
        clamped_right_x = np.maximum(raw_right_x, left_x + MIN_WIDTH_AT_HORIZON)

        # Step B: Polynomial Smoothing (Ütüleme)
        # Re-fit a curve to the clamped points to remove the sharp "elbow/kink".
        smooth_fit = np.polyfit(plot_y, clamped_right_x, 2)
        smooth_right_x = smooth_fit[0]*plot_y**2 + smooth_fit[1]*plot_y + smooth_fit[2]
        
        # Use the smoothed line for drawing
        final_right_x = smooth_right_x
        
    except (TypeError, np.linalg.LinAlgError):
        return image

    # Create points for polygon
    pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([final_right_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw Green Road Surface
    cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))
    
    # Draw Boundary Lines (Red)
    cv2.polylines(overlay, np.int_([pts_left]), False, (0, 0, 255), 20) 
    cv2.polylines(overlay, np.int_([pts_right]), False, (0, 0, 255), 20) 

    # Blend with original image
    return cv2.addWeighted(image, 1, overlay, 0.4, 0)

def process_frame(frame):
    height, width = frame.shape[:2]
    
    # 1. Preprocessing & Filters
    filtered = select_white_yellow(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 2. Region of Interest (Wide Trapezoid)
    roi_vertices = [
        (0, height),                    
        (width * 0.4, height * 0.55),   
        (width * 0.6, height * 0.55),   
        (width, height)                 
    ]
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 3. Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, np.array([]), minLineLength=40, maxLineGap=150)
    
    # 4. Find Boundaries
    left_fit, right_fit = find_road_boundaries(frame, lines)
    
    # 5. Draw Result
    result = draw_whole_road(frame, left_fit, right_fit)
    
    return result

def main():
    # Clear history for fresh run
    left_fit_history.clear()
    right_fit_history.clear()

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Processing video... Output will be saved to {OUTPUT_VIDEO_PATH}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        try:
            processed = process_frame(frame)
            out.write(processed)
            cv2.imshow('Road Detection', processed)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        except Exception as e:
            print(f"Frame processing error: {e}")
            continue

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing Complete.")

if __name__ == "__main__":
    main()