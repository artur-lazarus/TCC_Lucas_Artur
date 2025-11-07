import cv2
import numpy as np
import time
import os
from plate_detector import PlateDetector # <-- Import our updated class

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
PLATE_WIDTH_M = 0.5207  # 20.5 inches
PLATE_HEIGHT_M = 0.1143 # 4.5 inches

object_points_3d = np.array([
    [0.0, 0.0, 0.0],                  # Top-left
    [PLATE_WIDTH_M, 0.0, 0.0],        # Top-right
    [PLATE_WIDTH_M, PLATE_HEIGHT_M, 0.0], # Bottom-right
    [0.0, PLATE_HEIGHT_M, 0.0]        # Bottom-left
], dtype=np.float32)

print(f"Using 3D plate model: {PLATE_WIDTH_M*100:.1f}cm x {PLATE_HEIGHT_M*100:.1f}cm")

# --- 3. Initialize Plate Detector (Using our new module) ---
detector = PlateDetector()

# --- 4. Process Image and Extract 3D Points ---
all_3d_positions = []
image_path = '../../assets/transito-do-Rio.jpg'
output_dir = 'output' # Define output dir for debug images
os.makedirs(output_dir, exist_ok=True)
print(f"Debug images will be saved to '{output_dir}/' directory.")

print(f"Loading image: {image_path}")
img = cv2.imread(image_path)
img_display = img.copy() # Make a copy for drawing visualizations

if img is None:
    print(f"Error: Could not load image at {image_path}")
else:
    print("Running detection and corner finding...")
    
    # --- USE THE MODULE ---
    # save_crops=True is optional for debugging
    detections = detector.detect(img, save_crops=True, save_dir=output_dir)
    # --------------------
    
    print(f"Processed {len(detections)} total detections.")

    for i, det in enumerate(detections):
        # Get data from the result dictionary
        box = det['box']
        global_corners = det['corners']
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw the YOLO bounding box (in red)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # --- Check if corners were found ---
        if global_corners is not None:
            # Draw the 4 detected corners (in green)
            for j in range(4):
                p1 = tuple(global_corners[j].astype(int))
                p2 = tuple(global_corners[(j + 1) % 4].astype(int))
                cv2.line(img_display, p1, p2, (0, 255, 0), 2)
            
            # --- Solve PnP ---
            success, rvec, tvec = cv2.solvePnP(object_points_3d, 
                                               global_corners, 
                                               mtx, 
                                               dist)
            
            if success:
                # This is the 3D point we want!
                all_3d_positions.append(tvec.flatten())
                
                # Visualize the 3D position
                pos_str = f"Z:{tvec[2][0]:.2f}m"
                cv2.putText(img_display, pos_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"  Plate {i+1}: PnP success! Position: {tvec.flatten()}")
        else:
            print(f"  Plate {i+1}: Could not find 4 corners.")

    # --- 5. Save the 3D Point Cloud ---
    point_cloud = np.array(all_3d_positions)
    if point_cloud.shape[0] > 0:
        np.save('point_cloud.npy', point_cloud)
        print(f"\nSuccessfully extracted and saved {point_cloud.shape[0]} 3D points to 'point_cloud.npy'")
    else:
        print("\nNo 3D points were extracted.")

    # Show the final visualization
    debug_filename = os.path.join(output_dir, 'debug_detections.jpg')
    cv2.imwrite(debug_filename, img_display)
    print(f"\nSaved final debug image to '{debug_filename}'")