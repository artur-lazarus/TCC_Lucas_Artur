# step_02_pnp_extraction_video.py
import cv2
import numpy as np
import time
import os
from plate_detector import PlateDetector # Import our simplified module

# --- 1. Load Calibration Data (From Step 1) ---
try:
    with np.load('camera_calibration.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    print("Camera calibration data loaded successfully.")
except FileNotFoundError:
    print("Error: 'camera_calibration.npz' not found.")
    print("Please run Step 1 (Intrinsic Camera Calibration) first.")
    exit()

# --- 2. Define the 3D License Plate Model ---
# This remains the same: the 4 corners of our "template"
PLATE_WIDTH_M = 0.5207  # 20.5 inches
PLATE_HEIGHT_M = 0.1143 # 4.5 inches

object_points_3d = np.array([
    [0.0, 0.0, 0.0],                  # Top-left
    [PLATE_WIDTH_M, 0.0, 0.0],        # Top-right
    [PLATE_WIDTH_M, PLATE_HEIGHT_M, 0.0], # Bottom-right
    [0.0, PLATE_HEIGHT_M, 0.0]        # Bottom-left
], dtype=np.float32)

print(f"Using 3D plate model: {PLATE_WIDTH_M*100:.1f}cm x {PLATE_HEIGHT_M*100:.1f}cm")

# --- 3. Initialize Plate Detector ---
detector = PlateDetector()

# --- 4. Process Video and Collect 3D Points ---
all_3d_positions = []
video_path = '../../assets/your_traffic_video.mp4' # <-- Use your video file
output_dir = 'output' 
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video
        
    frame_num += 1
    
    # Process every 5th frame to speed things up
    if frame_num % 5 != 0:
        continue
    
    print(f"--- Processing Frame {frame_num} ---")
    
    # --- USE THE MODULE ---
    detections = detector.detect(frame)
    # --------------------
    
    frame_detections = 0
    for i, det in enumerate(detections):
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        
        # --- IMPLEMENTING THE PAPER'S LOGIC ---
        # We map our 4 3D model points to the 4 corners
        # of the *detected bounding box* ('P_Boundry' in the paper)
        
        image_points_2d = np.array([
            [x1, y1], # Top-left
            [x2, y1], # Top-right
            [x2, y2], # Bottom-right
            [x1, y2]  # Bottom-left
        ], dtype=np.float32)

        # --- Solve PnP ---
        # This is the practical equivalent of the paper's SFT step
        # It finds the 3D pose from the 3D model and 2D bounding box
        success, rvec, tvec = cv2.solvePnP(object_points_3d, 
                                           image_points_2d, 
                                           mtx, 
                                           dist)
        
        if success:
            frame_detections += 1
            # tvec is the 3D position (X,Y,Z) in meters
            all_3d_positions.append(tvec.flatten())
            
    if frame_detections > 0:
        print(f"  Found and saved {frame_detections} 3D points.")

cap.release()
cv2.destroyAllWindows()

# --- 5. Save the 3D Point Cloud ---
point_cloud = np.array(all_3d_positions)
if point_cloud.shape[0] > 0:
    np.save('point_cloud.npy', point_cloud)
    print(f"\n--- SUCCESS ---")
    print(f"Collected and saved {point_cloud.shape[0]} total 3D points to 'point_cloud.npy'")
    print("This is the 3D point cloud for the Motion Plane.")
else:
    print("\n--- FAILED ---")
    print("No 3D points were collected. Check your video file.")