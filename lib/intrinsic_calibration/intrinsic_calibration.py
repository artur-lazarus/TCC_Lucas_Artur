import cv2
import numpy as np
import glob

# --- 1. Define Checkerboard Parameters ---

# Define the dimensions of the checkerboard (number of inner corners).
# A 9x6 chessboard has 9 inner corners wide and 6 inner corners high.
CHECKERBOARD = (9, 6) # (inner_corners_width, inner_corners_height)

# Set termination criteria for the corner refinement algorithm
# This tells the algorithm when to stop iterating
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 2. Prepare Object Points (3D) ---

# Prepare the "object points" in 3D. These are the real-world
# coordinates of the 9x6 = 54 inner corners.
# We assume the checkerboard is on the Z=0 plane.
# (0,0,0), (1,0,0), (2,0,0) ... (8,0,0)
# (0,1,0), (1,1,0), (2,1,0) ... (8,1,0)
# ...
# (0,5,0), (1,5,0), (2,5,0) ... (8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images.
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in the image plane
img_shape = None # To store the image shape

# --- 3. Find Corners in Your Images ---

# Get a list of all calibration images
# Assumes images are in a folder named 'calibration_images'
images = glob.glob('calibration_images/*.png')

if not images:
    print("Error: No images found in 'calibration_images/' folder.")
    print("Please add your calibration photos and re-run.")
    exit()

print(f"Finding corners in {len(images)} images...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store the image shape (needed for calibration)
    if img_shape is None:
        img_shape = gray.shape[::-1] # (width, height)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If corners are found, refine them and add to our lists
    if ret == True:
        objpoints.append(objp)

        # Refine corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # --- Optional: Draw and display the corners for verification ---
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Found Corners', img)
        cv2.waitKey(2000) # Wait 2 seconds
        # -----------------------------------------------------------------

cv2.destroyAllWindows()
print("Corner finding complete.")

# --- 4. Perform Calibration ---

if not objpoints:
    print("Error: No corners were detected in any image.")
    print("Please use clearer photos or check your CHECKERBOARD dimensions.")
    exit()

print("Calibrating camera...")
# Pass the 3D points and their 2D image locations to the calibration function
# Type ignore is needed because OpenCV's type stubs are overly strict
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None  # type: ignore[call-overload]
)

if ret:
    print("\n--- Calibration Successful! ---")
    
    # mtx is the Camera Matrix (Intrinsics)
    print("\nCamera Matrix (mtx):")
    print(mtx)
    
    # dist is the Distortion Coefficients
    print("\nDistortion Coefficients (dist):")
    print(dist)
    
    # Extract and print focal length in pixels
    focal_length_x = mtx[0, 0]
    focal_length_y = mtx[1, 1]
    print(f"\nFocal Length (fx): {focal_length_x:.2f} pixels")
    print(f"Focal Length (fy): {focal_length_y:.2f} pixels")
    
    # --- 5. Save the Calibration Data ---
    # We save this for all future steps (PnP, SFT, etc.)
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)
    print("\nCalibration data saved to 'camera_calibration.npz'")

else:
    print("\n--- Calibration Failed! ---")
