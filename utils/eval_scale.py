import numpy as np
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def pixel_to_ground_plane(pixel_points, H):
    """
    Transform 2D pixel coordinates to ground plane coordinates using homography.
    
    Args:
        pixel_points: Array of shape (N, 2) with (x, y) pixel coordinates
        H: 3x3 homography matrix from image to bird's eye view
        
    Returns:
        Array of shape (N, 2) with (X, Y) ground plane coordinates in warped pixels
    """
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    if pixel_points.ndim == 1:
        pixel_points = pixel_points.reshape(1, -1)
    
    N = pixel_points.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pixel_h = np.hstack([pixel_points, ones]).T
    
    ground_h = H @ pixel_h
    
    ground_coords = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        w = ground_h[2, i]
        if abs(w) > 1e-6:
            ground_coords[i, 0] = ground_h[0, i] / w
            ground_coords[i, 1] = ground_h[1, i] / w
        else:
            ground_coords[i] = np.nan
            logging.warning(f"Point {i} has near-zero homogeneous coordinate (w={w})")
    
    return ground_coords


def measure_distance_warped(point_A, point_B, H):
    """
    Measure Euclidean distance between two points in warped image coordinates.
    
    Args:
        point_A: (x, y) pixel coordinates of point A in original image
        point_B: (x, y) pixel coordinates of point B in original image
        H: 3x3 homography matrix from image to bird's eye view
        
    Returns:
        Distance in warped image pixels, or None if transformation fails
    """
    points_2d = np.array([point_A, point_B], dtype=np.float64)
    points_warped = pixel_to_ground_plane(points_2d, H)
    
    if np.any(np.isnan(points_warped)):
        logging.error("Failed to transform points to warped coordinates")
        return None
    
    distance = np.linalg.norm(points_warped[1] - points_warped[0])
    return distance


def estimate_scale_from_points(H, point_A, point_B, real_distance_meters):
    """
    Estimate the scale (meters per pixel) in the warped image from two reference points.
    
    This function takes two points in the original image, transforms them to the warped
    bird's eye view using the homography matrix, measures the distance in warped pixels,
    and computes the scale as meters per pixel.
    
    Args:
        H: 3x3 homography matrix from image to bird's eye view (M_img_to_bird)
        point_A: (x, y) pixel coordinates of first point in original image
        point_B: (x, y) pixel coordinates of second point in original image
        real_distance_meters: Real-world distance between points A and B in meters
        
    Returns:
        scale_lambda: Scale in meters per pixel in warped image, or None if failed
        
    Example:
        >>> H = np.array([[...]])  # Your homography matrix
        >>> point_A = (100, 200)   # Point A in original image
        >>> point_B = (150, 250)   # Point B in original image
        >>> real_distance = 5.0    # 5 meters in real world
        >>> scale = estimate_scale_from_points(H, point_A, point_B, real_distance)
        >>> print(f"Scale: {scale:.6f} meters/pixel")
    """
    logging.info("="*80)
    logging.info("ESTIMATING SCALE FROM REFERENCE POINTS")
    logging.info("="*80)
    logging.info(f"Point A (original image): {point_A}")
    logging.info(f"Point B (original image): {point_B}")
    logging.info(f"Real-world distance: {real_distance_meters:.3f} meters")
    
    # Measure distance in warped image
    warped_distance_px = measure_distance_warped(point_A, point_B, H)
    
    if warped_distance_px is None:
        logging.error("Failed to measure distance in warped image")
        return None
    
    if warped_distance_px <= 0:
        logging.error(f"Invalid warped distance: {warped_distance_px}")
        return None
    
    # Compute scale
    scale_lambda = real_distance_meters / warped_distance_px
    
    logging.info(f"\nWarped image distance: {warped_distance_px:.2f} pixels")
    logging.info(f"Computed scale: {scale_lambda:.6f} meters/pixel")
    logging.info("="*80)
    
    return scale_lambda


def estimate_scale_from_multiple_measurements(H, point_pairs, real_distances):
    """
    Estimate scale from multiple reference point pairs using robust statistics.
    
    Uses median for robust estimation when multiple measurements are available.
    
    Args:
        H: 3x3 homography matrix from image to bird's eye view
        point_pairs: List of tuples [(point_A1, point_B1), (point_A2, point_B2), ...]
                     where each point is (x, y) in original image
        real_distances: List of real-world distances in meters [dist1, dist2, ...]
        
    Returns:
        scale_lambda: Median scale in meters per pixel, or None if failed
        all_scales: List of all computed scales for analysis
        
    Example:
        >>> H = np.array([[...]])
        >>> point_pairs = [
        ...     ((100, 200), (150, 250)),  # First measurement
        ...     ((300, 400), (350, 450))   # Second measurement
        ... ]
        >>> real_distances = [5.0, 5.0]  # Both are 5 meters
        >>> scale, all_scales = estimate_scale_from_multiple_measurements(
        ...     H, point_pairs, real_distances
        ... )
    """
    if len(point_pairs) != len(real_distances):
        logging.error("Number of point pairs must match number of real distances")
        return None, None
    
    logging.info("="*80)
    logging.info(f"ESTIMATING SCALE FROM {len(point_pairs)} MEASUREMENTS")
    logging.info("="*80)
    
    all_scales = []
    
    for i, ((point_A, point_B), real_dist) in enumerate(zip(point_pairs, real_distances)):
        logging.info(f"\nMeasurement {i+1}:")
        scale = estimate_scale_from_points(H, point_A, point_B, real_dist)
        
        if scale is not None:
            all_scales.append(scale)
        else:
            logging.warning(f"  Measurement {i+1} failed")
    
    if len(all_scales) == 0:
        logging.error("All measurements failed")
        return None, None
    
    # Use median for robust estimation
    median_scale = float(np.median(all_scales))
    
    logging.info("\n" + "="*80)
    logging.info("SCALE ESTIMATION SUMMARY")
    logging.info("="*80)
    logging.info(f"Valid measurements: {len(all_scales)}/{len(point_pairs)}")
    logging.info(f"Scale range: [{np.min(all_scales):.6f}, {np.max(all_scales):.6f}] m/px")
    logging.info(f"Scale mean: {np.mean(all_scales):.6f} m/px")
    logging.info(f"Scale median: {median_scale:.6f} m/px")
    logging.info(f"Scale std dev: {np.std(all_scales):.6f} m/px")
    logging.info("="*80)
    
    return median_scale, all_scales


# =============================================================================
# Visualization Helper
# =============================================================================

def visualize_scale_measurement(image, H, point_A, point_B, real_distance, 
                                W_out, H_out, scale_lambda=None):
    """
    Visualize the scale measurement by showing points in both original and warped images.
    
    Args:
        image: Original image (grayscale or BGR)
        H: 3x3 homography matrix
        point_A: (x, y) coordinates of point A in original image
        point_B: (x, y) coordinates of point B in original image
        real_distance: Real-world distance in meters
        W_out: Width of warped image
        H_out: Height of warped image
        scale_lambda: Optional pre-computed scale to display
        
    Returns:
        vis_combined: Combined visualization image
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis_original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_original = image.copy()
    
    # Draw points on original image
    cv2.circle(vis_original, tuple(map(int, point_A)), 8, (0, 255, 0), -1)
    cv2.circle(vis_original, tuple(map(int, point_B)), 8, (0, 0, 255), -1)
    cv2.line(vis_original, tuple(map(int, point_A)), tuple(map(int, point_B)), 
             (255, 255, 0), 2)
    
    # Add labels
    cv2.putText(vis_original, "A", (int(point_A[0])+10, int(point_A[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_original, "B", (int(point_B[0])+10, int(point_B[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis_original, f"{real_distance:.2f}m", 
                (int((point_A[0]+point_B[0])/2), int((point_A[1]+point_B[1])/2)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Warp the image
    warped = cv2.warpPerspective(image, H, (W_out, H_out))
    if len(warped.shape) == 2:
        vis_warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    else:
        vis_warped = warped.copy()
    
    # Transform points to warped image
    points_warped = pixel_to_ground_plane(np.array([point_A, point_B]), H)
    
    if not np.any(np.isnan(points_warped)):
        point_A_warped = tuple(map(int, points_warped[0]))
        point_B_warped = tuple(map(int, points_warped[1]))
        
        cv2.circle(vis_warped, point_A_warped, 8, (0, 255, 0), -1)
        cv2.circle(vis_warped, point_B_warped, 8, (0, 0, 255), -1)
        cv2.line(vis_warped, point_A_warped, point_B_warped, (255, 255, 0), 2)
        
        warped_dist = np.linalg.norm(points_warped[1] - points_warped[0])
        
        cv2.putText(vis_warped, "A", (point_A_warped[0]+10, point_A_warped[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_warped, "B", (point_B_warped[0]+10, point_B_warped[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis_warped, f"{warped_dist:.1f}px", 
                    (int((point_A_warped[0]+point_B_warped[0])/2), 
                     int((point_A_warped[1]+point_B_warped[1])/2)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if scale_lambda is not None:
            cv2.putText(vis_warped, f"Scale: {scale_lambda:.6f} m/px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine images side by side
    h1, w1 = vis_original.shape[:2]
    h2, w2 = vis_warped.shape[:2]
    max_h = max(h1, h2)
    
    # Resize to same height
    if h1 < max_h:
        scale_factor = max_h / h1
        vis_original = cv2.resize(vis_original, (int(w1*scale_factor), max_h))
    if h2 < max_h:
        scale_factor = max_h / h2
        vis_warped = cv2.resize(vis_warped, (int(w2*scale_factor), max_h))
    
    vis_combined = np.hstack([vis_original, vis_warped])
    
    return vis_combined


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Load a frame and homography matrix from your calibration
    # This would typically come from your calibration process
    
    # Mock example - replace with actual data
    import sys
    sys.path.append('..')
    
    # You would normally get these from your calibration:
    # from video_stream import video
    # from calibration import calibrate
    # H_matrix, _, _, _, _, _, _ = calibrate()
    
    # For demonstration, using a simple example
    H = np.array([
        [
            0.06613487736446343,
            -0.1809053294458529,
            186.79772104678847
        ],
        [
            0.18875649817134005,
            0.25486308535283875,
            -242.700916651341
        ],
        [
            6.445797774936904e-05,
            0.0004935350023170174,
            0.12946206457460752
        ]
    ], dtype=np.float64)
    
    # Example points (in original image coordinates)
    point_A = (239.69258725761773, 796.7403139427515)
    point_B = (1177.0525429362879, 100.32387072945521)
    real_distance = 27.995  # 5 meters
    
    # Single measurement
    scale = estimate_scale_from_points(H, point_A, point_B, real_distance)
    
    if scale is not None:
        print(f"\nEstimated scale: {scale:.6f} meters/pixel")
    
    # Multiple measurements example
    # point_pairs = [
    #     ((100, 200), (150, 250)),
    #     ((300, 400), (350, 450)),
    #     ((500, 600), (550, 650))
    # ]
    # real_distances = [5.0, 5.0, 5.0]
    
    # median_scale, all_scales = estimate_scale_from_multiple_measurements(
    #     H, point_pairs, real_distances
    # )
    
    # if median_scale is not None:
    #     print(f"\nRobust scale estimate: {median_scale:.6f} meters/pixel")
        
