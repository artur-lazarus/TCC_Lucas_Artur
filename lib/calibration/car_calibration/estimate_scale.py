import cv2
import numpy as np
import os
import sys

# Add detection library to path
DETECTION_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "detection")
sys.path.insert(0, DETECTION_LIB_PATH)
from detection_api import Detection

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
VP_W_PATH = os.path.join(THIS_DIR, "vp_w.npy")

VP1_2D = None  # VP in direction of traffic (vp_u - road direction)
VP2_2D = None  # VP perpendicular to traffic (vp_v - perpendicular to road)
VP3_2D = None  # VP vertical to road plane (vp_w - vertical)

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

try:
    vp_w = np.load(VP_W_PATH)
    VP3_2D = tuple(vp_w)
    print(f"[INFO] Loaded VP3 (vertical) from {VP_W_PATH}: {VP3_2D}")
except FileNotFoundError:
    print(f"[ERROR] Could not find {VP_W_PATH}. Please run compute_vp_w_from_homography.py first.")
except Exception as e:
    print(f"[ERROR] Failed to load VP3: {e}")

# Check if all VPs were loaded successfully
if VP1_2D is None or VP2_2D is None or VP3_2D is None:
    print(f"{'='*60}\n[FATAL ERROR] One or more Vanishing Points could not be loaded.\n"
          f"Please ensure all VP files exist:\n"
          f"  - {VP_U_PATH}\n"
          f"  - {VP_V_PATH}\n"
          f"  - {VP_W_PATH}\n"
          f"{'='*60}")
else:
    print(f"[SUCCESS] All VPs loaded successfully:")
    print(f"  VP1 (road direction): {VP1_2D}")
    print(f"  VP2 (perpendicular): {VP2_2D}")
    print(f"  VP3 (vertical): {VP3_2D}")

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
        
        # Create solid auxiliary lines: VP3-A, VP2-F, VP1-D
        line_vp3_A = (vp3, corners['A'])
        line_vp2_F = (vp2, corners['F'])
        line_vp1_D = (vp1, corners['D'])
        
        # Create dashed auxiliary lines: VP1-B, VP2-C, VP3-G
        line_vp1_B = (vp1, corners['B'])
        line_vp2_C = (vp2, corners['C'])
        line_vp3_G = (vp3, corners['G'])
        
        # Calculate auxiliary line intersections (I, J, K) from solid lines
        # I: VP1-D × VP2-F
        corners['I'] = line_line_intersection(line_vp1_D, line_vp2_F)
        # J: VP1-D × VP3-A
        corners['J'] = line_line_intersection(line_vp1_D, line_vp3_A)
        # K: VP2-F × VP3-A
        corners['K'] = line_line_intersection(line_vp2_F, line_vp3_A)
        
        # E: centroid of triangle I-J-K
        if corners['I'] and corners['J'] and corners['K']:
            centroid_x = (corners['I'][0] + corners['J'][0] + corners['K'][0]) / 3.0
            centroid_y = (corners['I'][1] + corners['J'][1] + corners['K'][1]) / 3.0
            corners['E'] = (int(centroid_x), int(centroid_y))
        
        # Calculate auxiliary line intersections (L, M, N) from dashed lines
        # L: VP1-B × VP2-C
        corners['L'] = line_line_intersection(line_vp1_B, line_vp2_C)
        # M: VP1-B × VP3-G
        corners['M'] = line_line_intersection(line_vp1_B, line_vp3_G)
        # N: VP2-C × VP3-G
        corners['N'] = line_line_intersection(line_vp2_C, line_vp3_G)
        
        # H: centroid of triangle L-M-N
        if corners['L'] and corners['M'] and corners['N']:
            centroid_x = (corners['L'][0] + corners['M'][0] + corners['N'][0]) / 3.0
            centroid_y = (corners['L'][1] + corners['M'][1] + corners['N'][1]) / 3.0
            corners['H'] = (int(centroid_x), int(centroid_y))
        
        # Add solid lines to tangent_lines for visualization
        tangent_lines['vp3_A'] = line_vp3_A
        tangent_lines['vp2_F'] = line_vp2_F
        tangent_lines['vp1_D'] = line_vp1_D
        
        # Add dashed lines to tangent_lines for visualization
        tangent_lines['vp1_B_dashed'] = line_vp1_B
        tangent_lines['vp2_C_dashed'] = line_vp2_C
        tangent_lines['vp3_G_dashed'] = line_vp3_G
        
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
                                   frame_number: int = 0) -> tuple:
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
            
            # Store all data
            detection_data.append((box, polygon, conf, hull_for_drawing, corners))
            
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
            
    return display_frame, wireframe_display, detection_data


def process_frame(frame: np.ndarray, detector_model=None, verbose: bool = True, save_debug_crops: bool = False, 
                  output_dir: str = None, frame_number: int = 0, confidence_threshold: float = 0.5) -> tuple:
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
        
    Returns:
        Tuple of (geometry_display, wireframe_display, detection_data)
        - geometry_display: FGMask + hull + tangent lines + points
        - wireframe_display: 3D box on real image
        - detection_data: List of detection information
    """
    # Run YOLOv8 segmentation using Detection API's "goes nuts" method
    d = Detection([frame], max_frames=1, color=True)
    mask_result = d.yolo_subtract(conf_threshold=0.8)
    
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
    
    # Process with both foreground mask and original frame
    return process_frame_with_yolo_masks(mask_result.masks, fgmask_bgr, frame, verbose, save_debug_crops, 
                                         output_dir, frame_number)


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
            save_debug_crops=True, output_dir=output_dir, frame_number=single_frame
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
                save_debug_crops=True, output_dir=output_dir, frame_number=frame_count
            )
            detection_count = len(detection_data)
            total_detections += detection_count
            
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


# --- Main execution ---
if __name__ == "__main__":
    INPUT_VIDEO = os.path.join(PROJECT_ROOT, "assets", "video.avi")
    OUTPUT_VIDEO = os.path.join(THIS_DIR, "output", "test_traffic_output.mp4")
    MASK_PATH = os.path.join(PROJECT_ROOT, "assets", "video_mask.png")
    TARGET_FPS = 10
    MAX_FRAMES = 700
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
