#!/usr/bin/env python

"""
estimate_vp_v.py

Restructured two-phase approach for estimating the second vanishing point (v):
- Phase 1: Plate detection only (collect 10 good plates)
- Phase 2: Line detection with dual filtering (VP-u + plate angles)
"""

import cv2
import numpy as np
import logging
import time
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='yolov5')

# --- Add detection_api to path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "detection"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

try:
    from detection_api import Detection
except ImportError:
    logging.error(f"Could not import Detection API from: {DETECTION_DIR}")
    sys.exit(1)

# --- Import Plate Detector ---
DETECT_PLATE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "detect_plate"))
if DETECT_PLATE_DIR not in sys.path:
    sys.path.insert(0, DETECT_PLATE_DIR)

from detect_plate_slow import PlateDetector

# --- Import DiamondSpace and Matplotlib ---
try:
    from diamond_space import DiamondSpace
except ImportError:
    logging.error("Could not import 'diamond_space'.")
    logging.error("Please install it: pip install diamond-space")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.error("Could not import 'matplotlib'.")
    logging.error("Please install it: pip install matplotlib")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def find_lines_with_hough(gray_frame, final_mask, gradient_threshold=50):
    """
    Finds straight line segments using the masked gradient.
    """
    
    # 1. Get gradient magnitude (the "car's edges")
    grad_x = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 2. Normalize to 0-255 for thresholding
    grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 3. Combine: Get gradient *only* where the mask is
    masked_grad = cv2.bitwise_and(grad_mag_norm, grad_mag_norm, mask=final_mask)
    
    # 4. Threshold to create clean wireframe
    _, wireframe = cv2.threshold(masked_grad, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    # 5. Find line segments in this clean "wireframe"
    lines = cv2.HoughLinesP(
        wireframe,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=20,
        maxLineGap=2
    )
    
    raw_lines = []
    if lines is not None:
        for line in lines:
            raw_lines.append(line[0]) # (x1, y1, x2, y2)
            
    return wireframe, raw_lines


def filter_lines_by_vp(lines, vp_u, angle_threshold_deg=60):
    """
    Removes lines that are pointing towards the first vanishing point (u).
    """
    filtered = []
    threshold_rad = np.deg2rad(angle_threshold_deg)
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: continue
        orientation = np.array([dx, dy]) / np.linalg.norm([dx, dy])
        if orientation[0] < 0:
            orientation = -orientation

        mid_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        vec_to_vp = vp_u - mid_point
        vec_to_vp_norm = np.linalg.norm(vec_to_vp)
        if vec_to_vp_norm == 0: continue
        
        vec_to_vp = vec_to_vp / vec_to_vp_norm
        
        dot_product = np.abs(np.dot(orientation, vec_to_vp))
        
        if dot_product < np.cos(threshold_rad): 
            filtered.append(line)
            
    return filtered


def estimate_plate_angle_from_aspect_ratio(plate_box, known_ratio=5.0):
    """
    Estimates the plate's true orientation using its known aspect ratio (520mm x 110mm ≈ 5:1).
    
    Returns tuple: (angles_list, bbox_aspect) or (None, None) if invalid.
    """
    x1, y1, x2, y2 = plate_box
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return None, None
    
    bbox_aspect = bbox_width / bbox_height
    
    if bbox_aspect >= known_ratio:
        return [0.0], bbox_aspect
    
    theta = np.arcsin(bbox_aspect / known_ratio)
    
    angle1 = theta
    angle2 = np.pi - theta
    
    return [angle1, angle2], bbox_aspect


def get_plate_angle(plate_boxes, gray_frame=None):
    """
    Calculates possible angles from detected plate boxes using aspect ratio geometry.
    Returns tuple: (angles_list, bbox_aspect, plate_box) or (None, None, None) if invalid.
    """
    for box in plate_boxes:
        angles, bbox_aspect = estimate_plate_angle_from_aspect_ratio(box, known_ratio=5.0)
        if angles is not None and len(angles) > 0:
            return angles, bbox_aspect, box
    
    return None, None, None


def filter_lines_by_plate_angles(lines, target_angles, angle_tolerance_deg=15):
    """
    Keeps only lines that are close to any of the target_angles (from plate detection).
    Works with line segments (x1, y1, x2, y2).
    """
    filtered = []
    tolerance_rad = np.deg2rad(angle_tolerance_deg)
    
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0: continue
        
        line_angle = np.arctan2(dy, dx) % np.pi
        
        for target_angle in target_angles:
            angle_diff = line_angle - target_angle
            angle_diff = (angle_diff + np.pi/2) % np.pi - np.pi/2
            
            if np.abs(angle_diff) <= tolerance_rad:
                filtered.append(line)
                break
            
    return filtered


def find_vp_from_lines_diamond_space(lines, frame_shape):
    """
    Finds the vanishing point (v) from Hough lines using the DiamondSpace accumulator.
    """
    img_h, img_w = frame_shape
    
    # --- 1. Convert lines to (A, B, C) format ---
    line_params = []
    for (x1, y1, x2, y2) in lines:
        # Line: a*x + b*y + c = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        line_params.append([A, B, C])
        
    if not line_params:
        return None
        
    lines_np = np.array(line_params, dtype=np.float32)
    
    # --- 2. Initialize and run DiamondSpace ---
    logging.info(f"Inserting {len(lines_np)} Hough lines into DiamondSpace...")
    d_val = int(1.0 * max(img_w, img_h))
    space_size = 128
    
    DS = DiamondSpace(d_val, space_size)
    DS.insert(lines_np)
    
    # --- 3. Find Peaks ---
    p, w, p_ds = DS.find_peaks(min_dist=8, prominence=0.9, t=0.35)
    
    if p is None or len(p) == 0:
        logging.warning("DiamondSpace found no peaks.")
        return None
        
    # --- 4. Visualize the Accumulator ---
    logging.info("Displaying DiamondSpace accumulator. Close plot to continue.")
    A_img = DS.attach_spaces()
    extent = ((-DS.size + 0.5) / DS.scale, (DS.size - 0.5) / DS.scale,
              (DS.size - 0.5) / DS.scale, (-DS.size + 0.5) / DS.scale)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(A_img, cmap="Greys", extent=extent)
    if p_ds is not None and len(p_ds):
        ax.plot(p_ds[:, 0] / DS.scale, p_ds[:, 1] / DS.scale, "r+", alpha=0.8, markersize=10)
    ax.set(title="Diamond Space Accumulator (Red+ = Peaks)")
    ax.invert_yaxis()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "02_diamond_space_accumulator.jpg")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved: 02_diamond_space_accumulator.jpg")
    
    plt.show()
        
    # --- 5. Return the best peak ---
    best_peak_xy = p[0][:2].astype(np.float32)
    best_weight = w[0]
    logging.info(f"DiamondSpace found VP-v: {best_peak_xy} with weight {best_weight:.2f}")
    
    return best_peak_xy


def find_supporting_lines(lines, vp, support_max_dist_px=5.0):
    """
    Filters a list of Hough lines, returning only those that "support"
    the final vanishing point.
    """
    logging.info(f"Finding supporting lines for VP {vp} (distance < {support_max_dist_px}px)...")
    supporting = []
    if vp is None:
        return supporting
    
    xv, yv = vp
    
    for line in lines:
        x1, y1, x2, y2 = line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # Calculate distance from point (xv, yv) to line (a, b, c)
        distance = abs(A * xv + B * yv + C) / np.sqrt(A**2 + B**2)
        
        if distance <= support_max_dist_px:
            supporting.append(line)
            
    logging.info(f"Found {len(supporting)} supporting lines out of {len(lines)}.")
    return supporting


def load_masks_from_video(mask_video_path, max_frames=None):
    """
    Loads foreground masks from a saved video file.
    Returns a list of grayscale mask frames.
    """
    logging.info(f"Loading masks from video: {mask_video_path}")
    cap = cv2.VideoCapture(mask_video_path)
    
    if not cap.isOpened():
        logging.error(f"Could not open mask video: {mask_video_path}")
        return None
    
    masks = []
    frame_count = 0
    
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            mask = frame
            
        masks.append(mask)
        frame_count += 1
    
    cap.release()
    logging.info(f"Loaded {len(masks)} mask frames from video")
    
    # Create a simple object to mimic the fg_masks structure
    class MaskContainer:
        def __init__(self, mask_list):
            self.masks = mask_list
    
    return MaskContainer(masks)

# -----------------------------------------------------------------------------
# Main Two-Phase Algorithm
# -----------------------------------------------------------------------------

def estimate_vanishing_point_v(detection_obj, fg_masks, vp_u, mask_path=None, 
                               frame_limit=1000, skip_frames=0, show_video=True, 
                               show_vanishing_points=False, single_frame_analysis=None, 
                               plate_detector=None):
    """
    Two-phase vanishing point estimation:
    
    Phase 1: Plate Detection Only
    - Loop through frames detecting plates only
    - No line detection during this phase
    - Collect angles from 10 good plates
    - Stop plate detection after 10 plates found
    
    Phase 2: Line Detection with Filtering
    - Start detecting lines with Hough transform
    - Apply BOTH filters per-frame:
      1. VP-u angle filter (against motion VP)
      2. Plate angle filter (using collected angles)
    - Accumulate filtered lines
    - Calculate VP-v at end
    """
            
    # --- Load and prepare static ROI mask ---
    roi_mask = None
    if mask_path:
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            logging.warning(f"Could not load ROI mask from {mask_path}.")
        else:
            logging.info(f"Successfully loaded static ROI mask from {mask_path}")
            
    # --- Phase tracking variables ---
    phase = 1  # Start with Phase 1: Plate Detection
    collected_plate_angles = []
    required_good_plates = 10
    good_plate_count = 0
    
    # --- Phase 2 variables ---
    accumulated_filtered_lines = []
    
    # --- Create a persistent canvas for visualization ---
    if single_frame_analysis is not None:
        canvas_frame = detection_obj.frames[single_frame_analysis]
        logging.info(f"Using frame {single_frame_analysis} as canvas for single frame analysis.")
    else:
        canvas_frame = detection_obj.frames[0]
    
    if canvas_frame is None:
        logging.error("Could not get frame for canvas.")
        return None, None
        
    persistent_canvas = cv2.cvtColor(canvas_frame, cv2.COLOR_GRAY2BGR)
    logging.info("Created persistent canvas for line visualization.")
    
    # --- Setup video writers ---
    output_dir = os.path.join(THIS_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    video_writers = None
    if show_video and single_frame_analysis is None:
        h, w = canvas_frame.shape[:2]
        fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        video_writers = {
            'persistent': cv2.VideoWriter(
                os.path.join(output_dir, "video_01_persistent_lines.mp4"),
                fourcc, fps, (w, h)
            ),
            'wireframe': cv2.VideoWriter(
                os.path.join(output_dir, "video_02_current_frame_wireframe.mp4"),
                fourcc, fps, (w, h)
            ),
            'mask': cv2.VideoWriter(
                os.path.join(output_dir, "video_03_foreground_mask.mp4"),
                fourcc, fps, (w, h)
            )
        }
        logging.info(f"Initialized video writers. Videos will be saved to {output_dir}")

    # --- Single frame analysis mode ---
    if single_frame_analysis is not None:
        logging.info(f"Single frame analysis mode: analyzing only frame {single_frame_analysis}")
        if single_frame_analysis >= len(detection_obj.frames) or single_frame_analysis >= len(fg_masks.masks):
            logging.error(f"Frame {single_frame_analysis} is out of range (max: {min(len(detection_obj.frames), len(fg_masks.masks))-1})")
            return None, None
    
    # =========================================================================
    # PHASE 1: PLATE DETECTION ONLY
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("PHASE 1: PLATE DETECTION ONLY")
    logging.info("="*80)
    logging.info(f"Goal: Collect angles from {required_good_plates} good plates")
    logging.info("No line detection during this phase\n")
    
    for frame_count, (frame, fg_mask) in enumerate(zip(detection_obj.frames, fg_masks.masks)):
        
        # Skip frames if single_frame_analysis is set
        if single_frame_analysis is not None and frame_count != single_frame_analysis:
            continue
        
        if frame_count > frame_limit:
            logging.info(f"Reached {frame_limit} frame processing limit.")
            break
            
        if frame is None or fg_mask is None:
            logging.warning(f"Skipping frame {frame_count}, data is missing.")
            continue
            
        # --- Resize static ROI mask on first frame ---
        if frame_count == 0 and roi_mask is not None:
            h, w = frame.shape[:2]
            if roi_mask.shape[0] != h or roi_mask.shape[1] != w:
                logging.warning(f"Resizing static mask...")
                roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
            
        gray_frame = frame  # Frame is already grayscale
        
        # --- Combine masks ---
        if roi_mask is not None:
            final_mask = cv2.bitwise_and(fg_mask, roi_mask)
        else:
            final_mask = fg_mask
        
        # --- PHASE 1: Detect plates every frame until we have 10 ---
        if phase == 1 and plate_detector is not None:
            masked_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=final_mask)
            frame_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
            plate_boxes = plate_detector.detect(frame_bgr, size=640, save_crops=False)
            
            if len(plate_boxes) > 0:
                plate_angles, bbox_aspect, plate_box = get_plate_angle(plate_boxes, gray_frame=gray_frame)
                
                if plate_angles is not None:
                    # Collect the angles
                    collected_plate_angles.extend(plate_angles)
                    good_plate_count += 1
                    
                    angle_str = ", ".join([f"{np.rad2deg(a):.1f}°" for a in plate_angles])
                    logging.info(f"Frame {frame_count}: Good plate #{good_plate_count}/{required_good_plates} - angles: [{angle_str}], aspect: {bbox_aspect:.2f}")
                    
                    # Save debug image
                    debug_plate_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    x1, y1, x2, y2 = map(int, plate_box)
                    cv2.rectangle(debug_plate_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(debug_plate_img, f"Aspect: {bbox_aspect:.2f}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    for i, angle in enumerate(plate_angles):
                        line_len = 150
                        dx_line = int(line_len * np.cos(angle))
                        dy_line = int(line_len * np.sin(angle))
                        color = (0, 255, 0) if i == 0 else (255, 255, 0)
                        cv2.line(debug_plate_img, (cx, cy), (cx + dx_line, cy + dy_line), color, 3)
                        cv2.putText(debug_plate_img, f"{np.rad2deg(angle):.1f}deg", 
                                  (cx + dx_line + 10, cy + dy_line), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    debug_path = os.path.join(output_dir, f"debug_plate_{frame_count}_count_{good_plate_count}.jpg")
                    cv2.imwrite(debug_path, debug_plate_img)
                    logging.info(f"Saved: debug_plate_{frame_count}_count_{good_plate_count}.jpg")
                    
                    # Check if we've collected enough plates
                    if good_plate_count >= required_good_plates:
                        logging.info(f"\n{'='*80}")
                        logging.info(f"PHASE 1 COMPLETE: Collected {good_plate_count} good plates")
                        logging.info(f"Collected angles: {[f'{np.rad2deg(a):.1f}°' for a in collected_plate_angles]}")
                        logging.info(f"{'='*80}\n")
                        phase = 2  # Switch to Phase 2
                        break  # Exit Phase 1 loop
        
        # Show visualization during Phase 1
        if show_video and phase == 1:
            vis_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis_frame, f"PHASE 1: Plate Detection ({good_plate_count}/{required_good_plates})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Phase 1: Plate Detection", vis_frame)
            cv2.imshow("Foreground Mask", final_mask)
            
            key_press = cv2.waitKey(10) & 0xFF
            if key_press == ord('q'):
                logging.info("User pressed 'q', stopping processing.")
                return None, None
    
    # Check if we collected enough plates
    if good_plate_count < required_good_plates:
        logging.error(f"Phase 1 incomplete: Only collected {good_plate_count}/{required_good_plates} plates")
        return None, None
    
    # =========================================================================
    # PHASE 2: LINE DETECTION WITH DUAL FILTERING
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("PHASE 2: LINE DETECTION WITH DUAL FILTERING")
    logging.info("="*80)
    logging.info(f"Applying both VP-u filter AND plate angle filter per-frame")
    logging.info(f"Plate angles to use: {[f'{np.rad2deg(a):.1f}°' for a in collected_plate_angles]}\n")
    
    for frame_count, (frame, fg_mask) in enumerate(zip(detection_obj.frames, fg_masks.masks)):
        
        # Skip frames if single_frame_analysis is set
        if single_frame_analysis is not None and frame_count != single_frame_analysis:
            continue
        
        if frame_count > frame_limit:
            logging.info(f"Reached {frame_limit} frame processing limit.")
            break
            
        if frame is None or fg_mask is None:
            continue
            
        gray_frame = frame
        
        # --- Combine masks ---
        if roi_mask is not None:
            final_mask = cv2.bitwise_and(fg_mask, roi_mask)
        else:
            final_mask = fg_mask
        
        # --- Detect lines with Hough ---
        wireframe, raw_lines = find_lines_with_hough(gray_frame, final_mask, gradient_threshold=50)
        
        # --- Apply Filter 1: VP-u angle filter ---
        vp_filtered_lines = filter_lines_by_vp(raw_lines, vp_u, angle_threshold_deg=45)
        
        # --- Apply Filter 2: Plate angle filter ---
        dual_filtered_lines = filter_lines_by_plate_angles(vp_filtered_lines, collected_plate_angles, angle_tolerance_deg=15)
        
        # --- Accumulate the dual-filtered lines ---
        accumulated_filtered_lines.extend(dual_filtered_lines)
        
        # Log progress
        if frame_count % 10 == 0 or single_frame_analysis is not None:
            logging.info(f"Frame {frame_count}: {len(raw_lines)} raw → {len(vp_filtered_lines)} VP-filtered → {len(dual_filtered_lines)} dual-filtered. Total: {len(accumulated_filtered_lines)}")
        
        # --- Visualization ---
        if show_video:
            # Draw accumulated lines on persistent canvas
            for (x1, y1, x2, y2) in dual_filtered_lines:
                cv2.line(persistent_canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            cv2.imshow("Persistent Dual-Filtered Lines (Green)", persistent_canvas)
            
            # Show current frame wireframe
            vis_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            vis_frame[wireframe > 0] = [0, 255, 255]
            cv2.putText(vis_frame, f"PHASE 2: Line Detection (Frame {frame_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Current Frame + Wireframe (Yellow)", vis_frame)
            
            cv2.imshow("Foreground Mask", final_mask)
            
            # Write to video
            if video_writers:
                video_writers['persistent'].write(persistent_canvas)
                video_writers['wireframe'].write(vis_frame)
                video_writers['mask'].write(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
            
            # Save single frame analysis images
            if single_frame_analysis is not None:
                single_frame_persistent_path = os.path.join(output_dir, f"04_single_frame_{single_frame_analysis}_persistent_lines.jpg")
                cv2.imwrite(single_frame_persistent_path, persistent_canvas)
                logging.info(f"Saved: 04_single_frame_{single_frame_analysis}_persistent_lines.jpg")
                
                single_frame_wireframe_path = os.path.join(output_dir, f"05_single_frame_{single_frame_analysis}_wireframe.jpg")
                cv2.imwrite(single_frame_wireframe_path, vis_frame)
                logging.info(f"Saved: 05_single_frame_{single_frame_analysis}_wireframe.jpg")
                
                single_frame_mask_path = os.path.join(output_dir, f"06_single_frame_{single_frame_analysis}_mask.jpg")
                cv2.imwrite(single_frame_mask_path, final_mask)
                logging.info(f"Saved: 06_single_frame_{single_frame_analysis}_mask.jpg")
            
            # Wait key
            key_press = cv2.waitKey(10) & 0xFF
            if key_press == ord('q'):
                logging.info("User pressed 'q', stopping video processing.")
                break
            if single_frame_analysis is not None:
                logging.info("Displaying single frame analysis. Press any key to continue...")
                cv2.waitKey(0)
                break
    
    # --- Save final persistent canvas ---
    persistent_canvas_path = os.path.join(output_dir, "01_persistent_lines_final.jpg")
    cv2.imwrite(persistent_canvas_path, persistent_canvas)
    logging.info(f"Saved: 01_persistent_lines_final.jpg")
    
    # --- Release video writers ---
    if video_writers:
        for name, writer in video_writers.items():
            writer.release()
        logging.info("Released all video writers.")
        
    # --- Keep windows open after loop ---
    if show_video and single_frame_analysis is None:
        logging.info("Processing finished. Press any key to close visualization...")
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    
    # --- Summary ---
    logging.info(f"\n=== FILTERING SUMMARY ===")
    logging.info(f"Phase 1: Collected {good_plate_count} plates with angles: {[f'{np.rad2deg(a):.1f}°' for a in collected_plate_angles]}")
    logging.info(f"Phase 2: Total dual-filtered lines accumulated: {len(accumulated_filtered_lines)}")
    
    if not accumulated_filtered_lines:
        logging.error("No valid lines after dual filtering. Cannot calculate VP-v.")
        return None, None
        
    # --- Calculate VP-v from accumulated lines ---
    vp_v = find_vp_from_lines_diamond_space(accumulated_filtered_lines, canvas_frame.shape[:2])
    
    return vp_v, accumulated_filtered_lines
    
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Setup Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    # --- Set your variables here ---
    video_file_path = "/Users/lucmarts/Documents/Pessoal/TCC_Lucas_Artur/assets/video.avi" 
    mask_file_path = "/Users/lucmarts/Documents/Pessoal/TCC_Lucas_Artur/assets/video_mask.png"
    vp_u_path = os.path.join(THIS_DIR, "vp_u.npy") 
    
    # Frame range parameters
    skip_frames = 5000  # Number of frames to skip at the beginning
    frame_limit = 5000  # Number of frames to process after skipping
    
    show_video_realtime = True
    # Set to a frame number to analyze *only* that frame (relative to skipped frames)
    single_frame_analysis = None  # Example: 280 would analyze frame 5280 if skip_frames=5000
    
    # Visualization mode: Set to True to show vanishing points + supporting lines
    # Set to False to show only supporting lines
    show_vanishing_points = False
    
    # Background loading: Set to path to load existing background, or None to generate from frames
    load_background_path = os.path.join(THIS_DIR, "background_frames_5000_to_6000.jpg")
    
    # Foreground mask loading: Set to path of saved mask video to load from file
    # Set to None to generate masks from scratch (requires background)
    load_masks_from_video_path = None
    
    # ---------------------------------
    
    # 1. Load VP-u from the first script
    try:
        vp_u = np.load(vp_u_path)
        logging.info(f"Successfully loaded vp_u: {vp_u} from {vp_u_path}")
    except FileNotFoundError:
        logging.error(f"Could not find '{vp_u_path}'.")
        logging.error("Please run the first script (estimate_vp_u.py) and save its output.")
        exit()
    except Exception as e:
        logging.error(f"Error loading '{vp_u_path}': {e}")
        exit()

    
    total_start_time = time.time()
    
    # --- Load Detection object with frame skipping ---
    logging.info(f"Skipping first {skip_frames} frames, then loading {frame_limit} frames...")
    d = Detection(video_file_path, max_frames=frame_limit, color=False, frame_interval=1, start_frame=skip_frames)
    
    # --- Step 1: Load or generate background ---
    if load_background_path is not None and os.path.exists(load_background_path):
        logging.info(f"Loading background from: {load_background_path}")
        d._background = cv2.imread(load_background_path, cv2.IMREAD_GRAYSCALE)
        if d._background is None:
            logging.error(f"Failed to load background from {load_background_path}")
            logging.info("Falling back to generating background from frames...")
            d.init_background(method='median')
            background_path = os.path.join(THIS_DIR, f"background_frames_{skip_frames}_to_{skip_frames+frame_limit}.jpg")
            cv2.imwrite(background_path, d._background)
            logging.info(f"Saved generated background to: {background_path}")
    else:
        if load_background_path is not None:
            logging.warning(f"Background file not found: {load_background_path}")
        logging.info("Generating median background from loaded frames...")
        d.init_background(method='median')
        background_path = os.path.join(THIS_DIR, f"background_frames_{skip_frames}_to_{skip_frames+frame_limit}.jpg")
        cv2.imwrite(background_path, d._background)
        logging.info(f"Saved generated background to: {background_path}")
    
    # --- Step 2: Load or generate foreground masks ---
    if load_masks_from_video_path is not None and os.path.exists(load_masks_from_video_path):
        logging.info(f"Loading foreground masks from video: {load_masks_from_video_path}")
        fg_masks = load_masks_from_video(load_masks_from_video_path, max_frames=frame_limit)
        if fg_masks is None:
            logging.error("Failed to load masks from video. Generating from scratch...")
            logging.info("Calculating foreground masks...")
            fg_masks = d.median_subtract_normalized(threshold_value=16).morphology.fill_holes()
    else:
        if load_masks_from_video_path is not None:
            logging.warning(f"Mask video not found: {load_masks_from_video_path}")
        logging.info("Generating foreground masks from scratch...")
        fg_masks = d.median_subtract_normalized(threshold_value=16).morphology.fill_holes()
    
    # --- Initialize plate detector ---
    plate_detector = None
    try:
        plate_detector = PlateDetector(conf_threshold=0.5)
    except Exception as e:
        logging.error(f"Failed to initialize PlateDetector: {e}")
        logging.error("Cannot proceed without plate detector for two-phase algorithm.")
        exit()
    
    logging.info("Starting two-phase VP-v estimation...")
    vp_v, all_lines = estimate_vanishing_point_v(
        detection_obj=d,
        fg_masks=fg_masks,
        vp_u=vp_u, 
        mask_path=mask_file_path,
        frame_limit=frame_limit,
        skip_frames=skip_frames,
        show_video=show_video_realtime,
        show_vanishing_points=show_vanishing_points,
        single_frame_analysis=single_frame_analysis,
        plate_detector=plate_detector
    )
    
    total_end_time = time.time()
    
    if vp_v is not None:
        logging.info(f"\n--- \nFinal Estimated Vanishing Point (v): {vp_v}")
        logging.info(f"Total estimation time: {total_end_time - total_start_time:.2f} seconds")
        vp_v_path = os.path.join(THIS_DIR, "vp_v.npy")
        np.save(vp_v_path, vp_v)
        logging.info(f"Saved result to '{vp_v_path}'")

        # --- Visualization of supporting lines ---
        logging.info("Displaying final visualization. Press any key to close.")
        cap = cv2.VideoCapture(video_file_path)
        # Set to the frame we analyzed for a consistent final image
        if single_frame_analysis is not None:
             cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames + single_frame_analysis)
        else:
             cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        ret, frame = cap.read()
        if ret:
            frame_h, frame_w = frame.shape[:2]
            vp_x_u, vp_y_u = int(vp_u[0]), int(vp_u[1])
            vp_x_v, vp_y_v = int(vp_v[0]), int(vp_v[1])
            
            # Calculate distance threshold as 3% of VP-v distance from image center
            center_x, center_y = frame_w / 2, frame_h / 2
            vp_v_distance_from_center = np.sqrt((vp_v[0] - center_x)**2 + (vp_v[1] - center_y)**2)
            support_threshold = 0.03 * vp_v_distance_from_center
            logging.info(f"Using support threshold: {support_threshold:.2f}px (3% of VP-v distance from center: {vp_v_distance_from_center:.2f}px)")
            
            # Find supporting lines
            supporting_lines = find_supporting_lines(all_lines, vp_v, support_max_dist_px=support_threshold)
            
            # --- Mode 1: Show VPs + Supporting Lines ---
            if show_vanishing_points:
                # Find boundaries to show both VPs and the frame
                min_x = min([0, vp_x_u - 50, vp_x_v - 50])
                min_y = min([0, vp_y_u - 50, vp_y_v - 50])
                max_x = max([frame_w, vp_x_u + 50, vp_x_v + 50])
                max_y = max([frame_h, vp_y_u + 50, vp_y_v + 50])
                
                new_w = max_x - min_x
                new_h = max_y - min_y
                offset_x = -min_x
                offset_y = -min_y
                
                new_canvas = np.full((new_h, new_w, 3), [100, 100, 100], dtype=np.uint8)
                new_canvas[offset_y : offset_y + frame_h, offset_x : offset_x + frame_w] = frame
                           
                # Draw VP-u (Red)
                vp_u_draw = (vp_x_u + offset_x, vp_y_u + offset_y)
                cv2.circle(new_canvas, vp_u_draw, 10, (0, 0, 255), -1) 
                cv2.putText(new_canvas, "u (motion)", (vp_u_draw[0]+15, vp_u_draw[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw VP-v (Blue)
                vp_v_draw = (vp_x_v + offset_x, vp_y_v + offset_y)
                cv2.circle(new_canvas, vp_v_draw, 10, (255, 0, 0), -1) 
                cv2.putText(new_canvas, "v (edgelet)", (vp_v_draw[0]+15, vp_v_draw[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw supporting lines (White)
                for (x1, y1, x2, y2) in supporting_lines:
                    x1_off = x1 + offset_x
                    y1_off = y1 + offset_y
                    x2_off = x2 + offset_x
                    y2_off = y2 + offset_y
                    cv2.line(new_canvas, (x1_off, y1_off), (x2_off, y2_off), (255, 255, 255), 2)
                
                # Save the final visualization
                output_dir = os.path.join(THIS_DIR, "output")
                final_viz_path = os.path.join(output_dir, "03_final_vps_and_supporting_lines.jpg")
                cv2.imwrite(final_viz_path, new_canvas)
                logging.info(f"Saved: 03_final_vps_and_supporting_lines.jpg")
                
                cv2.imshow("Final Vanishing Points (u and v) + Supporting Lines (White)", new_canvas)
                
            # --- Mode 2: Show only Supporting Lines ---
            else:
                # Create canvas with the original frame
                new_canvas = frame.copy()
                
                # Draw supporting lines (Blue) - extended towards VP
                for (x1, y1, x2, y2) in supporting_lines:
                    # Calculate direction vector
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length == 0:
                        continue
                    
                    # Normalize direction
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Extend line by a factor
                    extension_factor = 1.0
                    
                    # Calculate extended endpoints
                    x1_ext = int(x1 - dx_norm * length * extension_factor)
                    y1_ext = int(y1 - dy_norm * length * extension_factor)
                    x2_ext = int(x2 + dx_norm * length * extension_factor)
                    y2_ext = int(y2 + dy_norm * length * extension_factor)
                    
                    # Draw extended line in blue
                    cv2.line(new_canvas, (x1_ext, y1_ext), (x2_ext, y2_ext), (255, 0, 0), 2)
                
                # Save the final visualization
                output_dir = os.path.join(THIS_DIR, "output")
                final_viz_path = os.path.join(output_dir, "03_final_supporting_lines.jpg")
                cv2.imwrite(final_viz_path, new_canvas)
                logging.info(f"Saved: 03_final_supporting_lines.jpg")
                
                cv2.imshow("Supporting Lines (Blue)", new_canvas)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cap.release()

    else:
        logging.error("Could not estimate vanishing point v.")
