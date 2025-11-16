import cv2
import numpy as np
import os
import sys
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Add detection library to path
DETECTION_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "detection")
sys.path.insert(0, DETECTION_LIB_PATH)
from detection_api import Detection

# Add calibration library to path for homography functions
CALIBRATION_LIB_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, CALIBRATION_LIB_PATH)
from homography import (
    f_from_two_orthogonal_vps,
    get_rotation_matrix_from_vps,
    build_img_to_bird_homography,
    pixel_vp_to_cam_dir
)


# -------------------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(THIS_DIR, "..", "..", "..")

# -------------------------------------------------------------------
# CALIBRATION: LOAD VANISHING POINTS FROM .npy FILES
# -------------------------------------------------------------------
# Load vanishing points from the computed .npy files
VP_U_PATH = os.path.join(THIS_DIR, "vp_u.npy")
VP_V_PATH = os.path.join(THIS_DIR, "vp_v.npy")

VP1_2D = None  # VP in direction of traffic (vp_u - road direction)
VP2_2D = None  # VP perpendicular to traffic (vp_v - perpendicular to road)
VP3_2D = None  # VP vertical to road plane (vp_w - vertical) - COMPUTED from VP1 and VP2

try:
    vp_u = np.load(VP_U_PATH)
    VP1_2D = tuple(vp_u)
    print(f"[INFO] Loaded VP1 (road direction) from {VP_U_PATH}: {VP1_2D}")
except FileNotFoundError:
    print(f"[ERROR] Could not find {VP_U_PATH}. Please run estimate_vp_u.py first.")
except Exception as e:
    print(f"[ERROR] Failed to load VP1: {e}")

try:
    vp_v = np.load(VP_V_PATH)
    VP2_2D = tuple(vp_v)
    print(f"[INFO] Loaded VP2 (perpendicular) from {VP_V_PATH}: {VP2_2D}")
except FileNotFoundError:
    print(f"[ERROR] Could not find {VP_V_PATH}. Please run estimate_vp_v.py first.")
except Exception as e:
    print(f"[ERROR] Failed to load VP2: {e}")

# Check if required VPs (VP1 and VP2) were loaded successfully
if VP1_2D is None or VP2_2D is None:
    print(f"{'='*60}\n[FATAL ERROR] Required Vanishing Points (VP1, VP2) could not be loaded.\n"
          f"Please ensure VP files exist:\n"
          f"  - {VP_U_PATH}\n"
          f"  - {VP_V_PATH}\n"
          f"{'='*60}")
else:
    print(f"[SUCCESS] Required VPs loaded successfully:")
    print(f"  VP1 (road direction): {VP1_2D}")
    print(f"  VP2 (perpendicular): {VP2_2D}")
    print(f"  VP3 will be computed from VP1 and VP2 using homography")

# -------------------------------------------------------------------
# CAMERA INTRINSICS AND HOMOGRAPHY SETUP
# -------------------------------------------------------------------

K_MATRIX = None
H_IMG_TO_GROUND = None
GROUND_PLANE_SIZE = None

def compute_vp3_from_vp1_vp2(vp1_px, vp2_px, K):
    """
    Compute the third vanishing point (vertical) from two ground plane VPs.
    
    Since VP1 and VP2 are perpendicular and lie on the ground plane,
    their cross product gives the vertical direction (VP3).
    
    Args:
        vp1_px: (x, y) pixel coordinates of VP1 (road direction)
        vp2_px: (x, y) pixel coordinates of VP2 (perpendicular to road)
        K: 3x3 camera intrinsics matrix
        
    Returns:
        (x, y) pixel coordinates of VP3 (vertical direction)
    """
    # Convert pixel VPs to camera directions
    d1 = pixel_vp_to_cam_dir(vp1_px, K)
    d2 = pixel_vp_to_cam_dir(vp2_px, K)
    
    # VP3 direction is perpendicular to both VP1 and VP2 (cross product)
    d3 = np.cross(d1, d2)
    d3 /= np.linalg.norm(d3)
    
    # Project back to image coordinates: vp = K @ d
    vp3_h = K @ d3
    vp3_px = (vp3_h[0] / vp3_h[2], vp3_h[1] / vp3_h[2])
    
    return vp3_px


def initialize_camera_and_homography(img_shape, principal_point=None):
    """
    Initialize camera intrinsics and ground plane homography from vanishing points.
    
    Uses ONLY VP1 (road direction) and VP2 (perpendicular) to compute everything.
    VP3 (vertical) is derived mathematically from VP1 and VP2.
    
    Args:
        img_shape: (height, width) of the image
        principal_point: (cx, cy) principal point, defaults to image center
        
    Returns:
        Tuple of (K, H_img_to_ground, output_size, vp3_computed)
    """
    global K_MATRIX, H_IMG_TO_GROUND, GROUND_PLANE_SIZE, VP3_2D
    
    if VP1_2D is None or VP2_2D is None:
        raise ValueError("VP1 and VP2 must be loaded!")
    
    H_img, W_img = img_shape[:2]
    
    # Use image center as principal point if not provided
    if principal_point is None:
        cx, cy = W_img / 2.0, H_img / 2.0
    else:
        cx, cy = principal_point
    
    try:
        # Compute focal length from orthogonal VPs (VP1 and VP2 are perpendicular)
        f = f_from_two_orthogonal_vps(VP1_2D, VP2_2D, cx, cy)
        print(f"[INFO] Computed focal length from VP1 and VP2: {f:.2f} pixels")
        
        # Build camera intrinsics matrix
        K = np.array([
            [f,   0.0, cx],
            [0.0, f,   cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Compute VP3 (vertical) from VP1 and VP2 using homography
        vp3_computed = compute_vp3_from_vp1_vp2(VP1_2D, VP2_2D, K)
        print(f"[INFO] Computed VP3 (vertical) from VP1 and VP2: {vp3_computed}")
        
        # Update global VP3
        VP3_2D = vp3_computed
        
        # Get rotation matrix from vanishing points
        # Note: get_rotation_matrix_from_vps expects (vertical_vp, road_vp)
        # So we pass VP3 (computed vertical) and VP1 (road direction)
        r1, r2, r3 = get_rotation_matrix_from_vps(vp3_computed, VP1_2D, K)
        
        # Build homography to ground plane (bird's eye view)
        # This maps image pixels to metric coordinates on the ground plane
        H_img_to_ground, (W_out, H_out) = build_img_to_bird_homography(
            img_shape, K, r1, r2, scale=None, margin=0.02, roi_polygon=None, target_width_px=1280.0
        )
        
        print(f"[INFO] Ground plane output size: {W_out}x{H_out}")
        print(f"[INFO] Homography matrix initialized successfully")
        
        K_MATRIX = K
        H_IMG_TO_GROUND = H_img_to_ground
        GROUND_PLANE_SIZE = (W_out, H_out)
        
        return K, H_img_to_ground, (W_out, H_out), vp3_computed
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera and homography: {e}")
        return None, None, None, None

# -------------------------------------------------------------------
# 3D MEASUREMENT FUNCTIONS
# -------------------------------------------------------------------

def pixel_to_ground_plane(pixel_points, H_img_to_ground):
    """
    Transform 2D pixel coordinates to 3D ground plane coordinates (pseudo-units).
    
    Args:
        pixel_points: numpy array of shape (N, 2) with (x, y) pixel coordinates
        H_img_to_ground: 3x3 homography matrix from image to ground plane
        
    Returns:
        numpy array of shape (N, 2) with (X, Y) ground plane coordinates in pseudo-units
        Returns None for points that don't project validly
    """
    if H_img_to_ground is None:
        raise ValueError("Homography not initialized! Call initialize_camera_and_homography first.")
    
    # Convert to homogeneous coordinates
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    if pixel_points.ndim == 1:
        pixel_points = pixel_points.reshape(1, -1)
    
    N = pixel_points.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pixel_h = np.hstack([pixel_points, ones]).T  # 3xN
    
    # Apply homography
    ground_h = H_img_to_ground @ pixel_h  # 3xN
    
    # Normalize by homogeneous coordinate
    ground_coords = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        w = ground_h[2, i]
        if abs(w) > 1e-6:
            ground_coords[i, 0] = ground_h[0, i] / w
            ground_coords[i, 1] = ground_h[1, i] / w
        else:
            ground_coords[i] = np.nan
    
    return ground_coords


def measure_distance_3d(point_A, point_B, H_img_to_ground):
    """
    Measure Euclidean distance between two points in 3D pseudo-units.
    
    Args:
        point_A: (x, y) pixel coordinates of point A
        point_B: (x, y) pixel coordinates of point B
        H_img_to_ground: 3x3 homography matrix from image to ground plane
        
    Returns:
        Distance in pseudo-units, or None if transformation fails
    """
    try:
        # Transform both points to ground plane
        points_2d = np.array([point_A, point_B], dtype=np.float64)
        points_3d = pixel_to_ground_plane(points_2d, H_img_to_ground)
        
        # Check for invalid projections
        if np.any(np.isnan(points_3d)):
            return None
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(points_3d[1] - points_3d[0])
        return distance
        
    except Exception as e:
        print(f"[Warning] Failed to measure distance: {e}")
        return None


def measure_box_dimensions(corners, H_img_to_ground):
    """
    Measure key dimensions of the 3D bounding box in pseudo-units.
    
    GEOMETRICALLY CORRECT METHOD (Average of Edges):
    1. Project all corners to 3D ground plane
    2. Measure true 3D edge distances
    3. Average parallel edges in 3D
    
    This avoids perspective distortion from computing 2D midpoints first.
    
    - Width: Average of AB (front) and CH (back) measured in 3D
    - Length: Average of AC (left) and BH (right) measured in 3D
    
    Args:
        corners: Dictionary with corner points {'A': (x,y), 'B': (x,y), ...}
        H_img_to_ground: 3x3 homography matrix from image to ground plane
        
    Returns:
        Dictionary with measurements: {'width': distance, 'length': distance}
    """
    measurements = {}
    
    if corners is None or H_img_to_ground is None:
        return measurements
    
    # Width: Average of AB (front) and CH (back) edges
    if all(k in corners for k in ['A', 'B', 'C', 'H']):
        # Measure front width AB in 3D
        width_front = measure_distance_3d(corners['A'], corners['B'], H_img_to_ground)
        # Measure back width CH in 3D
        width_back = measure_distance_3d(corners['C'], corners['H'], H_img_to_ground)
        
        if width_front is not None and width_back is not None:
            # Average the two 3D measurements
            width = (width_front + width_back) / 2.0
            measurements['width'] = width
    
    # Length: Average of AC (left) and BH (right) edges
    if all(k in corners for k in ['A', 'C', 'B', 'H']):
        # Measure left length AC in 3D
        length_left = measure_distance_3d(corners['A'], corners['C'], H_img_to_ground)
        # Measure right length BH in 3D
        length_right = measure_distance_3d(corners['B'], corners['H'], H_img_to_ground)
        
        if length_left is not None and length_right is not None:
            # Average the two 3D measurements
            length = (length_left + length_right) / 2.0
            measurements['length'] = length
    
    return measurements


def find_kde_mode(data, grid_steps=1000):
    """
    Finds the mode (peak) of a 1D data distribution using
    Kernel Density Estimation (KDE).
    
    This is a robust replacement for np.median() to find the
    most common measurement, as shown in Dubská et al. 2014 [cite: 816-818]
    and Sochor et al. 2017 [cite: 258].
    
    Args:
        data: A 1D numpy array or list of measurements
        grid_steps: The resolution of the grid for finding the peak (default: 1000)
        
    Returns:
        The value corresponding to the peak of the KDE, or None if fails
    """
    if len(data) == 0:
        return None
    
    data = np.asarray(data)
    
    if data.size == 0:
        return None
    
    # 1. Filter extreme outliers (keep central 90%)
    q05 = np.percentile(data, 5)
    q95 = np.percentile(data, 95)
    filtered_data = data[(data >= q05) & (data <= q95)]
    
    if filtered_data.size == 0:
        # Fallback if filtering removes everything
        filtered_data = data
    
    if filtered_data.size == 1:
        # Single value, return it directly
        return float(filtered_data[0])
    
    # 2. Fit the KDE model
    try:
        kde = gaussian_kde(filtered_data)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to median if KDE fails (e.g., all identical values)
        return float(np.median(filtered_data))
    
    # 3. Discretize the space to find the peak [cite: 258]
    grid_min = np.min(filtered_data)
    grid_max = np.max(filtered_data)
    
    if grid_min == grid_max:
        # All values are the same
        return float(grid_min)
    
    data_grid = np.linspace(grid_min, grid_max, grid_steps)
    
    # 4. Find the Arg Max (The Peak)
    pdf_values = kde.evaluate(data_grid)
    peak_index = np.argmax(pdf_values)
    
    # 5. Return the Mode
    robust_mode = data_grid[peak_index]
    
    return float(robust_mode)


def compute_scale_from_measurements(all_measurements, real_car_width=1.81, real_car_length=4.49, output_dir=None):
    """
    Compute scene scale (λ) by statistically merging car measurements.
    
    Uses KDE-based mode finding (Dubská et al. 2014 [cite: 816-818]) for robust estimation.
    Combines two independent measurements:
    1. Car width (from 3D bbox)
    2. Car length (from 3D bbox)
    
    Args:
        all_measurements: List of car measurement dicts
        real_car_width: Real-world car width in meters (default: 1.81m)
        real_car_length: Real-world car length in meters (default: 4.49m)
        output_dir: Directory to save histogram visualizations
        
    Returns:
        Tuple of (lambda_final, stats_dict)
    """
    # Collect car measurements
    car_widths = []
    car_lengths = []
    
    for meas in all_measurements:
        if 'width' in meas:
            car_widths.append(meas['width'])
        if 'length' in meas:
            car_lengths.append(meas['length'])
    
    print(f"\n{'='*60}\n[SCALE CALCULATION]\n{'='*60}")
    print(f"Car measurements: {len(all_measurements)} vehicles")
    print(f"  Car width samples: {len(car_widths)}")
    print(f"  Car length samples: {len(car_lengths)}")
    
    if len(car_widths) == 0 and len(car_lengths) == 0:
        print("[ERROR] No valid measurements!")
        return None, {}
    
    stats = {
        'num_vehicles': len(all_measurements),
        'num_car_width_samples': len(car_widths),
        'num_car_length_samples': len(car_lengths),
        'real_car_width': real_car_width,
        'real_car_length': real_car_length
    }
    
    scales = []
    
    # Create figure for histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scale from car width
    mode_car_width = None
    if len(car_widths) > 0:
        mode_car_width = find_kde_mode(np.array(car_widths))
        if mode_car_width is not None:
            lambda_car_width = real_car_width / mode_car_width
            scales.append(lambda_car_width)
            stats['mode_car_width_pseudo'] = mode_car_width
            stats['lambda_car_width'] = lambda_car_width
            print(f"\nCar Width: mode={mode_car_width:.2f} units → λ={lambda_car_width:.6f} m/unit")
            
            # Plot histogram with KDE for width
            ax = axes[0]
            car_widths_array = np.array(car_widths)
            ax.hist(car_widths_array, bins=20, density=True, alpha=0.6, color='blue', edgecolor='black')
            
            # Plot KDE curve
            kde = gaussian_kde(car_widths_array)
            x_range = np.linspace(car_widths_array.min(), car_widths_array.max(), 200)
            kde_values = kde.evaluate(x_range)
            ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
            
            # Mark the mode
            ax.axvline(mode_car_width, color='green', linestyle='--', linewidth=2, label=f'Mode: {mode_car_width:.2f}')
            
            ax.set_xlabel('Car Width (pseudo-units)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Car Width Distribution\n{len(car_widths)} samples, mode={mode_car_width:.2f} units', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Scale from car length
    mode_car_length = None
    if len(car_lengths) > 0:
        mode_car_length = find_kde_mode(np.array(car_lengths))
        if mode_car_length is not None:
            lambda_car_length = real_car_length / mode_car_length
            scales.append(lambda_car_length)
            stats['mode_car_length_pseudo'] = mode_car_length
            stats['lambda_car_length'] = lambda_car_length
            print(f"Car Length: mode={mode_car_length:.2f} units → λ={lambda_car_length:.6f} m/unit")
            
            # Plot histogram with KDE for length
            ax = axes[1]
            car_lengths_array = np.array(car_lengths)
            ax.hist(car_lengths_array, bins=20, density=True, alpha=0.6, color='orange', edgecolor='black')
            
            # Plot KDE curve
            kde = gaussian_kde(car_lengths_array)
            x_range = np.linspace(car_lengths_array.min(), car_lengths_array.max(), 200)
            kde_values = kde.evaluate(x_range)
            ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
            
            # Mark the mode
            ax.axvline(mode_car_length, color='green', linestyle='--', linewidth=2, label=f'Mode: {mode_car_length:.2f}')
            
            ax.set_xlabel('Car Length (pseudo-units)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Car Length Distribution\n{len(car_lengths)} samples, mode={mode_car_length:.2f} units', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Save and display histogram
    plt.tight_layout()
    if output_dir:
        histogram_path = os.path.join(output_dir, "measurement_histograms.png")
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] Histograms saved to: {histogram_path}")
    
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to render
    
    if len(scales) > 0:
        # Calculate weighted average based on measurement confidence (inverse variance)
        # Lower variance = higher confidence = higher weight
        weights = []
        scale_values = []
        
        if 'lambda_car_width' in stats and len(car_widths) > 1:
            # Weight by inverse of variance on FILTERED data (same as used for KDE)
            car_widths_array = np.array(car_widths)
            q05 = np.percentile(car_widths_array, 5)
            q95 = np.percentile(car_widths_array, 95)
            filtered_widths = car_widths_array[(car_widths_array >= q05) & (car_widths_array <= q95)]
            
            variance = np.var(filtered_widths) if len(filtered_widths) > 1 else np.var(car_widths_array)
            weight = 1.0 / (variance + 1e-6) if variance > 0 else 1.0
            weights.append(weight)
            scale_values.append(stats['lambda_car_width'])
            stats['width_variance'] = variance
            stats['width_weight'] = weight
        
        if 'lambda_car_length' in stats and len(car_lengths) > 1:
            # Weight by inverse of variance on FILTERED data (same as used for KDE)
            car_lengths_array = np.array(car_lengths)
            q05 = np.percentile(car_lengths_array, 5)
            q95 = np.percentile(car_lengths_array, 95)
            filtered_lengths = car_lengths_array[(car_lengths_array >= q05) & (car_lengths_array <= q95)]
            
            variance = np.var(filtered_lengths) if len(filtered_lengths) > 1 else np.var(car_lengths_array)
            weight = 1.0 / (variance + 1e-6) if variance > 0 else 1.0
            weights.append(weight)
            scale_values.append(stats['lambda_car_length'])
            stats['length_variance'] = variance
            stats['length_weight'] = weight
        
        # Compute weighted average
        if len(weights) > 0 and sum(weights) > 0:
            lambda_weighted = np.average(scale_values, weights=weights)
            # Normalize weights for display
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
        else:
            lambda_weighted = np.mean(scales)
            normalized_weights = [1.0/len(scales)] * len(scales)
        
        lambda_min = np.min(scales)
        lambda_avg = np.mean(scales)
        
        stats['lambda_minimum'] = lambda_min
        stats['lambda_average'] = lambda_avg
        stats['lambda_weighted'] = lambda_weighted
        stats['lambda_final'] = lambda_weighted  # Use weighted as final
        stats['num_scales'] = len(scales)
        
        print(f"\n{'='*60}")
        print(f"SCALE SUMMARY ({len(scales)} sources):")
        print(f"  Minimum (conservative): λ = {lambda_min:.6f} m/unit")
        print(f"  Average (equal weight): λ = {lambda_avg:.6f} m/unit")
        print(f"  Weighted (by confidence): λ = {lambda_weighted:.6f} m/unit")
        print(f"{'='*60}")
        print(f"FINAL SCALE (weighted): λ = {lambda_weighted:.6f} m/unit")
        print(f"{'='*60}")
        
        # Display weighting details
        weight_info = []
        if 'width_weight' in stats:
            weight_info.append(f"Width: {normalized_weights[0]:.1%} (var={stats.get('width_variance', 0):.2f})")
        if 'length_weight' in stats:
            idx = 1 if 'width_weight' in stats else 0
            weight_info.append(f"Length: {normalized_weights[idx]:.1%} (var={stats.get('length_variance', 0):.2f})")
        print(f"Confidence weights: {', '.join(weight_info)}")
        print(f"Note: Lower variance = higher confidence = more weight")
        print(f"{'='*60}\n")
        
        return lambda_weighted, stats
    
    return None, stats


# -------------------------------------------------------------------
# GEOMETRY HELPER FUNCTIONS
# -------------------------------------------------------------------

def find_tangents_to_hull(vp, hull):
    """
    Finds the two tangent lines from a vanishing point to a convex hull.
    
    A line from VP to a hull point P is a tangent if all other hull
    points lie on the same side of the line.
    
    Args:
        vp: (x, y) tuple for the vanishing point.
        hull: A list of [x, y] points forming the convex hull.
        
    Returns:
        A tuple of (line1, line2), where each line is (vp, tangent_point).
        Returns (None, None) if hull is too small.
    """
    if len(hull) < 2:
        return None, None
        
    vp = np.array(vp)
    hull_points = np.array(hull).reshape(-1, 2)
    
    min_idx = -1
    max_idx = -1
    
    # We find the tangents by finding the min/max angles,
    # but arctan2 is a robust way to do this.
    angles = [np.arctan2(p[1] - vp[1], p[0] - vp[0]) for p in hull_points]
    
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    
    p_min = tuple(hull_points[min_idx])
    p_max = tuple(hull_points[max_idx])
    
    line1 = (tuple(vp), p_min)
    line2 = (tuple(vp), p_max)
    
    return line1, line2

def line_line_intersection(line1, line2):
    """
    Finds the intersection of two lines, each defined by two points.
    Returns (x, y) tuple or None if lines are parallel.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    # Line 1: (y1 - y2)x + (x2 - x1)y + (x1*y2 - x2*y1) = 0
    # Line 2: (y3 - y4)x + (x4 - x3)y + (x3*y4 - x4*y3) = 0
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if den == 0:
        return None  # Lines are parallel
        
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    if 0 <= t <= 1 and u >= 0: # Check if intersection is on segment 1 and ray 2
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        return (px, py)
        
    # We'll calculate the intersection point even if it's not on the segment
    # This is necessary for VPs which are "infinitely" far away
    px = int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den)
    py = int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den)
    
    return (px, py)

# -------------------------------------------------------------------
# STEP 2: 2D PROJECTED BOX CONSTRUCTION (Dubská Sec 2.2)
# -------------------------------------------------------------------

def get_projected_corners(hull, vp1, vp2, vp3):
    """
    Implements the 2D box construction from Dubská et al. 2014, Fig 3. [cite: 746-762]
    
    Args:
        hull: The convex hull of the vehicle's silhouette.
        vp1, vp2, vp3: The 2D (x, y) coordinates of the vanishing points.
        
    Returns:
        A tuple of (corners, tangent_lines)
        - corners: Dict of {'A': (x,y), 'B': (x,y), ...}
        - tangent_lines: Dict of {'red_lower': (p1, p2), ...}
        Returns (None, None) if construction fails.
    """
    
    # 1. Find all 6 tangent lines [cite: 742-744]
    try:
        t_red_l, t_red_u = find_tangents_to_hull(vp1, hull)
        t_green_l, t_green_u = find_tangents_to_hull(vp2, hull)
        t_blue_l, t_blue_r = find_tangents_to_hull(vp3, hull)
        
        tangent_lines = {
            "red_lower": t_red_l, "red_upper": t_red_u,
            "green_lower": t_green_l, "green_upper": t_green_u,
            "blue_left": t_blue_l, "blue_right": t_blue_r
        }
        
        # Check if all tangents were found
        if any(v is None for v in tangent_lines.values()):
            print("[Warning] Failed to find all tangents for hull.")
            return None, None
            
    except Exception as e:
        print(f"[Warning] Error in tangent finding: {e}")
        return None, None

    # 2. Find corner intersections
    # A: intersection between red and green closest to VP3 (vpw)
    # B: intersection between green and blue furthest from VP1 (vpu)
    # C: intersection between red and blue furthest from VP2 (vpv)
    try:
        corners = {}
        
        # Helper function to calculate distance between two points
        def dist(p1, p2):
            if p1 is None or p2 is None:
                return float('inf')
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # A: intersection of red and green closest to VP3
        a1 = line_line_intersection(t_red_l, t_green_l)
        a2 = line_line_intersection(t_red_l, t_green_u)
        a3 = line_line_intersection(t_red_u, t_green_l)
        a4 = line_line_intersection(t_red_u, t_green_u)
        a_candidates = [a1, a2, a3, a4]
        a_distances = [dist(a, vp3) for a in a_candidates]
        corners['A'] = a_candidates[np.argmin(a_distances)]
        
        # B: intersection of green and blue furthest from VP1
        b1 = line_line_intersection(t_green_l, t_blue_l)
        b2 = line_line_intersection(t_green_l, t_blue_r)
        b3 = line_line_intersection(t_green_u, t_blue_l)
        b4 = line_line_intersection(t_green_u, t_blue_r)
        b_candidates = [b1, b2, b3, b4]
        b_distances = [dist(b, vp1) for b in b_candidates]
        corners['B'] = b_candidates[np.argmax(b_distances)]
        
        # C: intersection of red and blue furthest from VP2
        c1 = line_line_intersection(t_red_l, t_blue_l)
        c2 = line_line_intersection(t_red_l, t_blue_r)
        c3 = line_line_intersection(t_red_u, t_blue_l)
        c4 = line_line_intersection(t_red_u, t_blue_r)
        c_candidates = [c1, c2, c3, c4]
        c_distances = [dist(c, vp2) for c in c_candidates]
        corners['C'] = c_candidates[np.argmax(c_distances)]
        
        # D: intersection of green and blue closest to VP1 (reuse b_candidates)
        corners['D'] = b_candidates[np.argmin(b_distances)]
        
        # F: intersection of blue and red closest to VP2 (reuse c_candidates)
        corners['F'] = c_candidates[np.argmin(c_distances)]
        
        # G: intersection of red and green furthest from VP3 (reuse a_candidates)
        corners['G'] = a_candidates[np.argmax(a_distances)]
        
        # Create all auxiliary lines for calculation and visualization
        line_vp3_A = (vp3, corners['A'])
        line_vp2_F = (vp2, corners['F'])
        line_vp1_D = (vp1, corners['D'])
        line_vp3_G = (vp3, corners['G'])
        line_vp2_C = (vp2, corners['C'])
        line_vp1_B = (vp1, corners['B'])
        
        # E: Direct intersection of VP3-A × VP2-F
        corners['E'] = line_line_intersection(line_vp3_A, line_vp2_F)
        
        # H: Direct intersection of VP3-G × VP2-C
        corners['H'] = line_line_intersection(line_vp3_G, line_vp2_C)
        
        # Add all auxiliary lines to tangent_lines for visualization
        tangent_lines['vp3_A'] = line_vp3_A
        tangent_lines['vp2_F'] = line_vp2_F
        tangent_lines['vp1_D'] = line_vp1_D
        tangent_lines['vp3_G_dashed'] = line_vp3_G
        tangent_lines['vp2_C_dashed'] = line_vp2_C
        tangent_lines['vp1_B_dashed'] = line_vp1_B
        
        # Check for failures
        if any(v is None for v in corners.values()):
            print("[Warning] Failed to find all corner intersections (parallel lines?).")
            return None, None
            
    except Exception as e:
        print(f"[Warning] Error in corner intersection: {e}")
        return None, None

    return corners, tangent_lines

# -------------------------------------------------------------------
# DETECTION PIPELINE
# -------------------------------------------------------------------

def process_frame_with_yolo_masks(yolo_masks: list, fgmask: np.ndarray, original_frame: np.ndarray, verbose: bool = True, 
                                   save_debug_crops: bool = False, output_dir: str = None, 
                                   frame_number: int = 0, roi_mask: np.ndarray = None) -> tuple:
    """
    Helper function to process YOLO segmentation masks and run 2D Bounding Box construction.
    
    Args:
        yolo_masks: List of binary masks from YOLO segmentation
        fgmask: Foreground mask (grayscale or BGR) for visualization
        original_frame: Original BGR frame for wireframe overlay
        verbose: Print detection info
        save_debug_crops: Whether to save debug crop images
        output_dir: Directory to save debug crops
        frame_number: Frame number for naming
        
    Returns:
        Tuple of (display_frame, detection_data)
        - display_frame: Dual display (geometry + wireframe on real image)
        - detection_data: List of detection information
    """
    detection_count = 0
    detection_data = []
    
    # Convert fgmask to BGR if it's grayscale
    if len(fgmask.shape) == 2:
        fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    else:
        fgmask_bgr = fgmask.copy()
    
    # Create displays
    display_frame = fgmask_bgr.copy()
    wireframe_display = original_frame.copy()
    
    # Check if VPs are set
    if VP1_2D is None:
        cv2.putText(display_frame, "ERROR: VPs NOT SET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(wireframe_display, "ERROR: VPs NOT SET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return display_frame, wireframe_display, detection_data
    
    # Find contours in masks to extract vehicle regions
    for mask in yolo_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small contours
                continue
            
            detection_count += 1
            
            # Get bounding box
            x1, y1, w, h = cv2.boundingRect(contour)
            x2, y2 = x1 + w, y1 + h
            box = np.array([x1, y1, x2, y2])
            
            # Filter out vehicles cropped by ROI mask
            if roi_mask is not None:
                # Check if vehicle bbox touches ROI mask boundary
                # Expand bbox slightly and check if it contains any zero pixels (outside ROI)
                margin = 5
                x1_check = max(0, x1 - margin)
                y1_check = max(0, y1 - margin)
                x2_check = min(roi_mask.shape[1], x2 + margin)
                y2_check = min(roi_mask.shape[0], y2 + margin)
                
                # Create a thin border around bbox to check
                border_mask = np.zeros_like(roi_mask)
                border_mask[y1_check:y2_check, x1_check:x2_check] = 255
                inner_mask = np.zeros_like(roi_mask)
                inner_y1 = min(y1 + margin, y2_check)
                inner_x1 = min(x1 + margin, x2_check)
                inner_y2 = max(y2 - margin, y1_check)
                inner_x2 = max(x2 - margin, x1_check)
                if inner_y2 > inner_y1 and inner_x2 > inner_x1:
                    inner_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 255
                border_mask = cv2.subtract(border_mask, inner_mask)
                
                # Check if border intersects with ROI edge (has zero pixels)
                border_check = cv2.bitwise_and(roi_mask, border_mask)
                border_pixels = np.count_nonzero(border_mask)
                valid_pixels = np.count_nonzero(border_check)
                
                if border_pixels > 0 and valid_pixels < border_pixels * 0.9:
                    if verbose:
                        print(f"[Skipped] Vehicle cropped by ROI mask: [{x1}, {y1}, {x2}, {y2}] - {valid_pixels}/{border_pixels} border pixels in ROI")
                    continue
            
            # Use contour as polygon
            polygon = contour.reshape(-1, 2)
            
            # Fake confidence (since we don't have it from masks)
            conf = 1.0
            
            if verbose:
                print(f"[Detected] Vehicle #{detection_count}: at [{x1}, {y1}, {x2}, {y2}] - Area: {area:.0f}")
            
            # Calculate convex hull
            hull = cv2.convexHull(polygon)
            hull_points_list = hull.reshape(-1, 2)
            hull_for_drawing = hull.astype(int)
            
            if verbose:
                print(f"  Convex hull: {len(hull)} points (from {len(polygon)} contour points)")
            
            # Get 2D Projected Corners
            corners, tangent_lines = get_projected_corners(hull_points_list, VP1_2D, VP2_2D, VP3_2D)
            
            # Measure 3D distances if homography is initialized
            measurements = {}
            plate_widths = []
            
            if corners and H_IMG_TO_GROUND is not None:
                measurements = measure_box_dimensions(corners, H_IMG_TO_GROUND)
                
                if verbose and measurements:
                    print(f"  3D Measurements (pseudo-units):")
                    for key, value in measurements.items():
                        print(f"    Distance {key}: {value:.2f} units")
            
            # Store all data including measurements
            detection_data.append((box, polygon, conf, hull_for_drawing, corners, measurements))
            
            # Draw convex hull on the display
            cv2.drawContours(display_frame, [hull_for_drawing], -1, (255, 255, 0), 2)
            
            # Draw tangent lines on the display
            # Color scheme: blue for vpw (VP3), green for vpv (VP2), red for vpu (VP1)
            if tangent_lines:
                tangent_colors = {
                    "red_lower": (0, 0, 255), "red_upper": (0, 0, 255),  # VP1 (vpu) = red
                    "green_lower": (0, 255, 0), "green_upper": (0, 255, 0),  # VP2 (vpv) = green
                    "blue_left": (255, 0, 0), "blue_right": (255, 0, 0),  # VP3 (vpw) = blue
                    "vp3_A": (255, 0, 0),  # VP3 (vpw) line = blue
                    "vp2_F": (0, 255, 0),  # VP2 (vpv) line = green
                    "vp1_D": (0, 0, 255),  # VP1 (vpu) line = red
                    "vp1_B_dashed": (0, 0, 255),  # VP1 (vpu) dashed = red
                    "vp2_C_dashed": (0, 255, 0),  # VP2 (vpv) dashed = green
                    "vp3_G_dashed": (255, 0, 0)   # VP3 (vpw) dashed = blue
                }
                
                for name, line in tangent_lines.items():
                    if line is not None:
                        p1, p2 = line
                        color = tangent_colors.get(name, (128, 128, 128))
                        vx, vy = int(p1[0]), int(p1[1])
                        px, py = int(p2[0]), int(p2[1])
                        
                        # Check if this is a dashed line
                        is_dashed = 'dashed' in name
                        
                        if is_dashed:
                            # Draw dashed line
                            def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
                                dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                                dashes = int(dist / dash_length)
                                for i in range(dashes):
                                    if i % 2 == 0:  # Draw only even segments
                                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes)
                                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)
                                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i + 1) / dashes)
                                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i + 1) / dashes)
                                        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
                            draw_dashed_line(display_frame, (vx, vy), (px, py), color)
                        else:
                            # Draw solid line with extension
                            p_ext_x = int(1.5*px - 0.5*vx)
                            p_ext_y = int(1.5*py - 0.5*vy)
                            cv2.line(display_frame, (vx, vy), (p_ext_x, p_ext_y), color, 2)
            
            # Draw corner points on the display
            if corners:
                for point_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    if point_name in corners:
                        point = corners[point_name]
                        cv2.circle(display_frame, point, 6, (0, 255, 255), -1)
                        cv2.putText(display_frame, point_name, (point[0] + 8, point[1] + 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create separate wireframe display on real image
            wireframe_display = original_frame.copy()
            
            # Helper function to draw dashed line segment
            def draw_dashed_segment(img, pt1, pt2, color, thickness=2, dash_length=10):
                if pt1 is None or pt2 is None:
                    return
                dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                dashes = max(1, int(dist / dash_length))
                for i in range(dashes):
                    if i % 2 == 0:  # Draw only even segments
                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes)
                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)
                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i + 1) / dashes)
                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i + 1) / dashes)
                        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # Draw box edges if all corners exist
            if corners:
                edge_color = (255, 255, 255)  # White for box edges
                
                # Solid edges
                solid_edges = [
                    ('A', 'B'), ('A', 'C'), ('A', 'E'),
                    ('B', 'F'), ('C', 'D'), ('D', 'E'),
                    ('D', 'G'), ('E', 'F'), ('F', 'G')
                ]
                
                for p1_name, p2_name in solid_edges:
                    if p1_name in corners and p2_name in corners:
                        pt1 = corners[p1_name]
                        pt2 = corners[p2_name]
                        cv2.line(wireframe_display, pt1, pt2, edge_color, 2)
                
                # Dashed edges
                dashed_edges = [('B', 'H'), ('C', 'H'), ('G', 'H')]
                
                for p1_name, p2_name in dashed_edges:
                    if p1_name in corners and p2_name in corners:
                        pt1 = corners[p1_name]
                        pt2 = corners[p2_name]
                        draw_dashed_segment(wireframe_display, pt1, pt2, edge_color)
                
                # Draw corner points on wireframe
                for point_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    if point_name in corners:
                        point = corners[point_name]
                        cv2.circle(wireframe_display, point, 6, (0, 255, 255), -1)
                        cv2.putText(wireframe_display, point_name, (point[0] + 8, point[1] + 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display measurements on wireframe
                if measurements:
                    y_offset = 60
                    for key, value in measurements.items():
                        text = f"{key}: {value:.2f} units"
                        cv2.putText(wireframe_display, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
            
    return display_frame, wireframe_display, detection_data


def process_frame(frame: np.ndarray, detector_model=None, verbose: bool = True, save_debug_crops: bool = False, 
                  output_dir: str = None, frame_number: int = 0, confidence_threshold: float = 0.5,
                  roi_mask: np.ndarray = None) -> tuple:
    """
    Helper function to process a single frame with vehicle segmentation
    AND run the 2D Bounding Box construction.
    
    Uses the Detection API's "goes nuts" YOLO-based background subtraction.
    
    Args:
        frame: BGR frame to process
        detector_model: Unused (kept for API compatibility)
        verbose: Print detection info
        save_debug_crops: Whether to save debug crop images
        output_dir: Directory to save debug crops
        frame_number: Frame number for naming
        confidence_threshold: Unused (kept for API compatibility)
        roi_mask: ROI mask for filtering cropped vehicles
        
    Returns:
        Tuple of (geometry_display, wireframe_display, detection_data)
        - geometry_display: FGMask + hull + tangent lines + points
        - wireframe_display: 3D box on real image
        - detection_data: List of detection information
    """
    # Run YOLOv8 segmentation using Detection API's "goes nuts" method
    d = Detection([frame], max_frames=1, color=True)
    mask_result = d.yolo_subtract(conf_threshold=0.7)
    
    if mask_result is None or len(mask_result.masks) == 0:
        # Return empty displays
        empty = np.zeros_like(frame)
        return empty, frame.copy(), []
    
    # Generate the foreground mask by combining all YOLO masks
    fgmask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in mask_result.masks:
        fgmask = cv2.bitwise_or(fgmask, mask)
    
    # Convert to BGR for visualization
    fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    
    # Process with both foreground mask and original frame, passing roi_mask through
    return process_frame_with_yolo_masks(mask_result.masks, fgmask_bgr, frame, verbose, save_debug_crops, 
                                         output_dir, frame_number, roi_mask=roi_mask)


def run_detection_pipeline(image_path: str, output_path: str):
    """
    Runs vehicle detection/segmentation pipeline on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
    """
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"\nProcessing {image_path}...")
    
    # Process the frame
    display_frame, detection_data = process_frame(frame, verbose=True)
    detection_count = len(detection_data)
                        
    # Save the result
    if detection_count > 0:
        cv2.imwrite(output_path, display_frame)
        print(f"\nSuccessfully processed {detection_count} vehicles.")
        print(f"Output saved to {output_path}")
    else:
        print("\nNo vehicles detected in the image.")


def process_video(video_path: str, output_path: str, target_fps: int = 10, display: bool = True, max_frames: int = None, mask_path: str = None, single_frame: int = None):
    """
    Process a video file with vehicle segmentation using "goes nuts" YOLO-based background subtraction.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        target_fps: Target frames per second for output
        display: Whether to display processing in real-time
        max_frames: Maximum number of frames to process
        mask_path: Path to ROI mask image
        single_frame: If set, process only this frame number
    """
    # Load ROI mask if provided
    roi_mask = None
    if mask_path:
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            print(f"Warning: Could not load ROI mask from {mask_path}. Proceeding without mask.")
        else:
            print(f"Successfully loaded ROI mask from {mask_path}")
    
    # 3. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize camera intrinsics and homography for 3D measurements
    print("\n[INFO] Initializing camera calibration and homography...")
    print("[INFO] Using VP1 (road) and VP2 (perpendicular) - VP3 (vertical) will be computed")
    try:
        K, H, size, vp3_computed = initialize_camera_and_homography((height, width))
        if K is not None:
            print("[SUCCESS] Camera calibration initialized successfully")
            print(f"[INFO] VP3 computed as: {vp3_computed}")
        else:
            print("[WARNING] Failed to initialize camera calibration - 3D measurements will be unavailable")
    except Exception as e:
        print(f"[WARNING] Could not initialize camera calibration: {e}")
        print("[WARNING] 3D measurements will be unavailable")
    
    # Resize mask to match video dimensions
    if roi_mask is not None:
        if roi_mask.shape[0] != height or roi_mask.shape[1] != width:
            print(f"Resizing mask from {roi_mask.shape} to {height}x{width}")
            roi_mask = cv2.resize(roi_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        print(f"ROI mask prepared: {height}x{width}")
    
    # ... (print video info) ...
    
    # Calculate frame skip
    frame_skip = max(1, int(original_fps / target_fps))
    effective_fps = original_fps / frame_skip
    
    # 3. Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 4. Set up video writers (one for geometry, one for wireframe)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_geometry = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
    wireframe_path = output_path.replace('.mp4', '_wireframe.mp4')
    out_wireframe = cv2.VideoWriter(wireframe_path, fourcc, effective_fps, (width, height))
    
    # --- Handle single frame mode ---
    if single_frame is not None:
        print(f"\n{'='*60}\nSINGLE FRAME MODE: Frame {single_frame}\n{'='*60}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, single_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {single_frame}")
            cap.release()
            return
        
        masked_frame = frame.copy()
        if roi_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        
        print(f"\nRunning YOLO segmentation...")
        geometry_display, wireframe_display, detection_data = process_frame(
            masked_frame, verbose=True,
            save_debug_crops=True, output_dir=output_dir, frame_number=single_frame,
            roi_mask=roi_mask
        )
        
        detection_count = len(detection_data)
        print(f"\n{'='*60}\nDetection Results:\n  Total detections: {detection_count}\n{'='*60}")
        
        geometry_output = output_path.replace('.mp4', f'_frame_{single_frame}_geometry.jpg')
        wireframe_output = output_path.replace('.mp4', f'_frame_{single_frame}_wireframe.jpg')
        cv2.imwrite(geometry_output, geometry_display)
        cv2.imwrite(wireframe_output, wireframe_display)
        print(f"\nSaved geometry to: {geometry_output}")
        print(f"Saved wireframe to: {wireframe_output}")
        
        if display:
            cv2.imshow('Geometry - FGMask + Hull + Tangents', geometry_display)
            cv2.imshow('Wireframe - 3D Box on Real Image', wireframe_display)
            print("\nPress any key to close...")
            cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # --- Video Loop ---
    print(f"\nProcessing video... (Press 'q' to quit, 'p' to pause)\n")
    if max_frames:
        print(f"Max frames limit: {max_frames} frames will be read from video\n")
    
    frame_count = 0
    processed_count = 0
    total_detections = 0
    paused = False
    
    # Accumulate all measurements for scale calculation
    all_measurements = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Check max_frames limit BEFORE processing
            if max_frames and frame_count > max_frames:
                print(f"\nReached max_frames limit: {max_frames}")
                break
            
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            masked_frame = frame.copy()
            if roi_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            geometry_display, wireframe_display, detection_data = process_frame(
                masked_frame, verbose=False,
                save_debug_crops=True, output_dir=output_dir, frame_number=frame_count,
                roi_mask=roi_mask
            )
            detection_count = len(detection_data)
            total_detections += detection_count
            
            # Collect measurements from this frame
            for detection in detection_data:
                if len(detection) >= 6:  # Has measurements
                    box, polygon, conf, hull, corners, measurements = detection
                    if measurements:
                        all_measurements.append(measurements)
            
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {detection_count}"
            cv2.putText(geometry_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(wireframe_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out_geometry.write(geometry_display)
            out_wireframe.write(wireframe_display)
            
            if display:
                cv2.imshow('Geometry - FGMask + Hull + Tangents', geometry_display)
                cv2.imshow('Wireframe - 3D Box on Real Image', wireframe_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): print("\nStopped by user"); break
                elif key == ord('p'):
                    paused = not paused
                    if paused: print("Paused - Press 'p' to resume")
                    while paused:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('p'): paused = False; print("Resumed")
                        elif key == ord('q'): paused = False; break
            
            if processed_count % 50 == 0:
                print(f"Progress: {(frame_count / total_frames) * 100:.1f}% | Processed: {processed_count} | Total Cars: {total_detections}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        out_geometry.release()
        out_wireframe.release()
        if display: cv2.destroyAllWindows()
        
        print(f"\n{'='*60}\nProcessing Complete!\n{'='*60}")
        print(f"Total frames read: {frame_count}/{total_frames}")
        print(f"Frames processed: {processed_count}")
        print(f"Total vehicles detected: {total_detections}")
        print(f"Geometry video saved to: {output_path}")
        print(f"Wireframe video saved to: {wireframe_path}")
        print(f"{'='*60}")
        
        # Compute final scene scale from accumulated measurements
        if len(all_measurements) > 0 and H_IMG_TO_GROUND is not None:
            lambda_final, scale_stats = compute_scale_from_measurements(
                all_measurements,
                real_car_width=1.81,  # Median width from 741 vehicle dataset
                real_car_length=4.49,  # Median length from 741 vehicle dataset
                output_dir=output_dir
            )
            
            if lambda_final is not None:
                # Save scale to file
                scale_output_path = os.path.join(output_dir, "scene_scale.txt")
                with open(scale_output_path, 'w') as f:
                    f.write(f"Scene Scale Factor (λ)\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(f"Final Scale: {lambda_final:.6f} meters/pseudo-unit\n")
                    f.write(f"Average Scale: {scale_stats.get('lambda_average', 0):.6f} meters/pseudo-unit\n\n")
                    f.write(f"Statistics:\n")
                    f.write(f"  Total vehicles measured: {scale_stats.get('num_vehicles', 0)}\n")
                    f.write(f"  Width samples: {scale_stats.get('num_width_samples', 0)}\n")
                    f.write(f"  Length samples: {scale_stats.get('num_length_samples', 0)}\n\n")
                    if 'mode_car_width_pseudo' in scale_stats:
                        f.write(f"  Mode car width (pseudo-units): {scale_stats['mode_car_width_pseudo']:.2f}\n")
                        f.write(f"  Scale from car width: {scale_stats['lambda_car_width']:.6f} m/unit\n")
                    if 'mode_car_length_pseudo' in scale_stats:
                        f.write(f"  Mode car length (pseudo-units): {scale_stats['mode_car_length_pseudo']:.2f}\n")
                        f.write(f"  Scale from car length: {scale_stats['lambda_car_length']:.6f} m/unit\n")
                
                print(f"\n[SUCCESS] Scale factor saved to: {scale_output_path}")
        else:
            print("\n[INFO] Scale calculation skipped (no measurements or homography not initialized)")


# --- Main execution ---
if __name__ == "__main__":
    INPUT_VIDEO = os.path.join(PROJECT_ROOT, "assets", "video.avi")
    OUTPUT_VIDEO = os.path.join(THIS_DIR, "output", "test_traffic_output.mp4")
    MASK_PATH = os.path.join(PROJECT_ROOT, "assets", "video_mask.png")
    TARGET_FPS = 30
    MAX_FRAMES = 5000
    SINGLE_FRAME = None # 1670
    
    process_video(
        INPUT_VIDEO, 
        OUTPUT_VIDEO, 
        target_fps=TARGET_FPS, 
        display=True, 
        max_frames=MAX_FRAMES,
        mask_path=MASK_PATH,
        single_frame=SINGLE_FRAME
    )
