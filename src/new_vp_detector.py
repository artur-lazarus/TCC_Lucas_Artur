import cv2
import numpy as np
import logging
import time
import os
import sys
import warnings
from detection import Detection
from detect_plate import PlateDetector
from diamond_space import DiamondSpace
from video_stream import video

warnings.filterwarnings('ignore', category=FutureWarning, module='yolov5')

def _find_vp_from_lines_diamond_space(lines, frame_shape, line_format):
    """
    Finds vanishing point using DiamondSpace accumulator.
    
    Args:
        lines: List of lines in one of two formats:
               - 'segment': [(x1, y1, x2, y2), ...] from HoughLinesP
               - 'params': [(a, b, c), ...] from _fit_lines_to_tracks where ax + by + c = 0
        frame_shape: (height, width) of the frame
        line_format: 'segment' or 'params' to specify input format
        
    Returns:
        Vanishing point as (x, y) tuple or None if failed
    """
    img_h, img_w = frame_shape
    
    # Convert lines to (A, B, C) format if needed
    line_params = []
    
    if line_format == 'segment':
        # Convert from (x1, y1, x2, y2) to (A, B, C)
        for (x1, y1, x2, y2) in lines:
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            line_params.append([A, B, C])
    elif line_format == 'params':
        # Lines are already in (a, b, c) format
        line_params = [[a, b, c] for (a, b, c) in lines]
    else:
        logging.error(f"Unknown line_format: {line_format}")
        return None
    
    if not line_params:
        return None
    
    lines_np = np.array(line_params, dtype=np.float32)
    
    d_val = int(1.0 * max(img_w, img_h))
    space_size = 128
    
    DS = DiamondSpace(d_val, space_size)
    DS.insert(lines_np)
    
    p, w, p_ds = DS.find_peaks(min_dist=8, prominence=0.9, t=0.35)
    
    if p is None or len(p) == 0:
        logging.warning("DiamondSpace found no peaks")
        return None
    
    best_peak_xy = p[0][:2].astype(np.float32)
    logging.info(f"DiamondSpace found VP: {best_peak_xy} with weight {w[0]:.2f}")
    
    return best_peak_xy

# =============================================================================
# VP-U (Road Direction) Functions - KLT Tracking Based
# =============================================================================

def _fit_lines_to_tracks(tracks, min_track_len=10, min_track_displacement=50):
    """
    Fits lines (ax + by + c = 0) to tracks that pass length and displacement filters.
    """
    lines = []
    valid_tracks_count = 0
    
    for track in tracks:
        if len(track) < min_track_len:
            continue
        
        start_point = track[0]
        end_point = track[-1]
        displacement = np.linalg.norm(start_point - end_point)
        
        if displacement < min_track_displacement:
            continue
        
        valid_tracks_count += 1
        points = np.array(track).reshape(-1, 2)
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        vx, vy, x0, y0 = line_params.flatten()
        a = vy
        b = -vx
        c = vx * y0 - vy * x0
        lines.append((a, b, c))
    
    return lines


def estimate_vp_u(min_valid_lines_to_stop=500, show_video=False):
    """
    Estimates the road direction vanishing point (u) using KLT tracking.
    
    Args:
        video_path: Path to video file
        mask_path: Optional ROI mask path
        min_frame_displacement: Minimum pixel movement per frame for valid tracks
        min_valid_lines_to_stop: Stop after collecting this many valid lines
        show_video: Display tracking visualization
        
    Returns:
        vp_u: (x, y) coordinates of vanishing point u, or None if failed
    """
    logging.info("="*80)
    logging.info("PHASE 1: Estimating VP-u (Road Direction) using KLT Tracking")
    logging.info("="*80)
    
    # Parameters
    min_dist = 5
    min_frame_displacement=1.0
    min_track_len_filter = 10
    min_track_disp_filter = 100
    
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=min_dist,
        blockSize=7
    )
    
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    old_frame = video.get_frame()
    if old_frame is None:
        logging.error("Could not read first frame")
        return None
    
    p0 = cv2.goodFeaturesToTrack(old_frame, **feature_params)
    
    all_valid_lines = []
    if p0 is not None:
        active_tracks = {i: [p0[i].ravel()] for i in range(len(p0))}
        logging.info(f"Found {len(p0)} initial features to track")
    else:
        active_tracks = {}
    
    frame_count = 0
    next_feature_id = len(active_tracks)
    
    while True:
        frame = video.get_frame
        if frame is None:
            break
        
        newly_completed_tracks = []
        
        if not active_tracks:
            p0 = cv2.goodFeaturesToTrack(old_frame, **feature_params)
            if p0 is not None:
                active_tracks = {i + next_feature_id: [p0[i].ravel()] for i in range(len(p0))}
                next_feature_id += len(p0)
            else:
                old_frame = frame.copy()
                continue
        
        p0_list = np.array([track[-1] for track in active_tracks.values()]).astype(np.float32).reshape(-1, 1, 2)
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0_list, None, **lk_params)
        
        new_active_tracks = {}
        track_ids = list(active_tracks.keys())
        
        if p1 is not None:
            for track_id, pt_new, st in zip(track_ids, p1, status):
                track = active_tracks[track_id]
                
                if st == 1:
                    pt_old = track[-1]
                    displacement = np.linalg.norm(pt_new.ravel() - pt_old)
                    
                    if displacement < min_frame_displacement:
                        if len(track) > 1:
                            newly_completed_tracks.append(track)
                        continue
                    
                    track.append(pt_new.ravel())
                    new_active_tracks[track_id] = track
                else:
                    if len(track) > 1:
                        newly_completed_tracks.append(track)
        
        active_tracks = new_active_tracks
        
        if newly_completed_tracks:
            new_lines = _fit_lines_to_tracks(newly_completed_tracks, 
                                           min_track_len=min_track_len_filter,
                                           min_track_displacement=min_track_disp_filter)
            if new_lines:
                all_valid_lines.extend(new_lines)
        
        if len(all_valid_lines) > min_valid_lines_to_stop:
            logging.info(f"Collected {len(all_valid_lines)} valid lines, stopping")
            break
        
        if show_video:
            vis_frame = frame.copy()
            for track in active_tracks.values():
                for k in range(len(track) - 1):
                    x1, y1 = map(int, track[k])
                    x2, y2 = map(int, track[k+1])
                    cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                last_pt = track[-1]
                cv2.circle(vis_frame, (int(last_pt[0]), int(last_pt[1])), 5, (0, 0, 255), -1)
            
            cv2.imshow('KLT Tracking for VP-u', vis_frame)
            cv2.imwrite('test-output/vp_debug/KLT_Tracking_for_VPu.png')
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        
        old_frame = frame.copy()
        
        if frame_count % 10 == 0 and len(active_tracks) < 50:
            p0_new = cv2.goodFeaturesToTrack(old_frame, **feature_params)
            if p0_new is not None:
                for pt in p0_new:
                    if not any(np.linalg.norm(pt.ravel() - track[-1]) < 5 for track in active_tracks.values()):
                        active_tracks[next_feature_id] = [pt.ravel()]
                        next_feature_id += 1
        
        frame_count += 1
    
    # Add remaining active tracks
    final_active_tracks = list(active_tracks.values())
    if final_active_tracks:
        new_lines = _fit_lines_to_tracks(final_active_tracks,
                                       min_track_len=min_track_len_filter,
                                       min_track_displacement=min_track_disp_filter)
        if new_lines:
            all_valid_lines.extend(new_lines)
    
    if show_video:
        cv2.destroyAllWindows()
    
    logging.info(f"Tracking complete: {frame_count} frames, {len(all_valid_lines)} valid lines")
    
    if len(all_valid_lines) < 10:
        logging.error(f"Only {len(all_valid_lines)} valid lines. Cannot estimate VP-u.")
        return None
    
    frame_shape = old_frame.shape[:2]
    
    # Use DiamondSpace with 'params' format (lines are already in (a, b, c) format)
    vp_u = _find_vp_from_lines_diamond_space(all_valid_lines, frame_shape, 'params')
    
    if vp_u is not None:
        logging.info(f"VP-u estimated: {tuple(vp_u)}")
        return tuple(vp_u)
    
    return None


# =============================================================================
# VP-V (Perpendicular Direction) Functions - Plate Detection Based
# =============================================================================

def find_lines_with_hough(gray_frame, final_mask, gradient_threshold=50):
    """Finds straight line segments using masked gradient."""
    grad_x = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    masked_grad = cv2.bitwise_and(grad_mag_norm, grad_mag_norm, mask=final_mask)
    _, wireframe = cv2.threshold(masked_grad, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    lines = cv2.HoughLinesP(wireframe, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=20, maxLineGap=2)
    
    raw_lines = []
    if lines is not None:
        for line in lines:
            raw_lines.append(line[0])
    
    return wireframe, raw_lines


def filter_lines_by_vp(lines, vp_u, angle_threshold_deg=60):
    """Removes lines pointing towards the first vanishing point (u)."""
    filtered = []
    threshold_rad = np.deg2rad(angle_threshold_deg)
    
    for line in lines:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        
        orientation = np.array([dx, dy]) / np.linalg.norm([dx, dy])
        if orientation[0] < 0:
            orientation = -orientation
        
        mid_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        vec_to_vp = vp_u - mid_point
        vec_to_vp_norm = np.linalg.norm(vec_to_vp)
        if vec_to_vp_norm == 0:
            continue
        
        vec_to_vp = vec_to_vp / vec_to_vp_norm
        dot_product = np.abs(np.dot(orientation, vec_to_vp))
        
        if dot_product < np.cos(threshold_rad):
            filtered.append(line)
    
    return filtered


def estimate_plate_angle_from_aspect_ratio(plate_box, known_ratio=5.0):
    """
    Estimates plate orientation from aspect ratio (520mm x 110mm ≈ 5:1).
    Returns (angles_list, bbox_aspect) or (None, None).
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


def get_plate_angle(plate_boxes):
    """Calculates possible angles from detected plate boxes."""
    for box in plate_boxes:
        angles, bbox_aspect = estimate_plate_angle_from_aspect_ratio(box, known_ratio=5.0)
        if angles is not None and len(angles) > 0:
            return angles, bbox_aspect, box
    return None, None, None


def filter_lines_by_plate_angles(lines, target_angles, angle_tolerance_deg=15):
    """Keeps only lines close to target angles from plate detection."""
    filtered = []
    tolerance_rad = np.deg2rad(angle_tolerance_deg)
    
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        
        line_angle = np.arctan2(dy, dx) % np.pi
        
        for target_angle in target_angles:
            angle_diff = line_angle - target_angle
            angle_diff = (angle_diff + np.pi/2) % np.pi - np.pi/2
            
            if np.abs(angle_diff) <= tolerance_rad:
                filtered.append(line)
                break
    
    return filtered


def estimate_vp_v(vp_u, plate_detector, frame_limit=1000, show_video=False):
    """
    Two-phase estimation of perpendicular vanishing point (v):
    Phase 1: Collect plate angles (10 plates)
    Phase 2: Detect and filter lines, calculate VP-v
    
    Args:
        detection_obj: Detection object with video frames
        fg_masks: Foreground masks
        vp_u: Previously computed VP-u
        mask_path: Optional ROI mask path
        frame_limit: Maximum frames to process
        plate_detector: PlateDetector instance
        show_video: Display visualization
        
    Returns:
        vp_v: (x, y) coordinates of vanishing point v, or None if failed
    """
    logging.info("="*80)
    logging.info("PHASE 2: Estimating VP-v (Perpendicular) using Plate Detection")
    logging.info("="*80)
    
    # Phase tracking
    phase = 1
    collected_plate_angles = []
    required_good_plates = 10
    good_plate_count = 0
    accumulated_filtered_lines = []
    
    # Phase 1: Plate Detection
    logging.info(f"\nPhase 1: Collecting angles from {required_good_plates} plates...")
    
    for frame_count, (frame, fg_mask) in enumerate(zip(detection_obj.frames, fg_masks.masks)):
        if frame_count > frame_limit:
            break
        
        if frame is None or fg_mask is None:
            continue
        
        # Resize ROI mask on first frame
        if frame_count == 0 and roi_mask is not None:
            h, w = frame.shape[:2]
            if roi_mask.shape[0] != h or roi_mask.shape[1] != w:
                roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        
        gray_frame = frame
        final_mask = cv2.bitwise_and(fg_mask, roi_mask) if roi_mask is not None else fg_mask
        
        # Phase 1: Detect plates
        if phase == 1 and plate_detector is not None:
            plate_boxes = plate_detector.detect(frame, size=640, save_crops=False)
            
            if len(plate_boxes) > 0:
                plate_angles, bbox_aspect, plate_box = get_plate_angle(plate_boxes)
                
                if plate_angles is not None:
                    collected_plate_angles.extend(plate_angles)
                    good_plate_count += 1
                    
                    if good_plate_count >= required_good_plates:
                        logging.info(f"Phase 1 complete: Collected {good_plate_count} plates")
                        phase = 2
                        break
    
    if good_plate_count < required_good_plates:
        logging.error(f"Phase 1 incomplete: Only {good_plate_count}/{required_good_plates} plates")
        return None
    
    # Phase 2: Line Detection with Filtering
    logging.info(f"\nPhase 2: Line detection with dual filtering...")
    logging.info(f"Using plate angles: {[f'{np.rad2deg(a):.1f}°' for a in collected_plate_angles]}")
    
    for frame_count, (frame, fg_mask) in enumerate(zip(detection_obj.frames, fg_masks.masks)):
        if frame_count > frame_limit:
            break
        
        if frame is None or fg_mask is None:
            continue
        
        gray_frame = frame
        final_mask = cv2.bitwise_and(fg_mask, roi_mask) if roi_mask is not None else fg_mask
        
        # Detect lines
        wireframe, raw_lines = find_lines_with_hough(gray_frame, final_mask, gradient_threshold=50)
        
        # Apply dual filtering
        vp_filtered = filter_lines_by_vp(raw_lines, vp_u, angle_threshold_deg=45)
        dual_filtered = filter_lines_by_plate_angles(vp_filtered, collected_plate_angles, 
                                                      angle_tolerance_deg=15)
        
        accumulated_filtered_lines.extend(dual_filtered)
        
        if show_video and frame_count % 10 == 0:
            vis_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            for (x1, y1, x2, y2) in dual_filtered:
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(vis_frame, f"Phase 2: Lines {len(accumulated_filtered_lines)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Phase 2: Line Detection", vis_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    if show_video:
        cv2.destroyAllWindows()
    
    logging.info(f"Phase 2 complete: {len(accumulated_filtered_lines)} dual-filtered lines")
    
    if not accumulated_filtered_lines:
        logging.error("No valid lines after filtering")
        return None
    
    # Calculate VP-v using 'segment' format (lines are in (x1, y1, x2, y2) format from HoughLinesP)
    canvas_shape = detection_obj.frames[0].shape[:2]
    vp_v = _find_vp_from_lines_diamond_space(accumulated_filtered_lines, canvas_shape, 'segment')
    
    if vp_v is not None:
        logging.info(f"VP-v estimated: {vp_v}")
    
    return vp_v

# =============================================================================
# Main Unified Interface
# =============================================================================

def detect_road_and_cross_vps(video: VideoStream, show_video=False):
    """
    Detect both vanishing points from traffic video.
    
    This is the main entry point that:
    1. Estimates VP-u (road direction) using KLT tracking
    2. Estimates VP-v (perpendicular) using plate detection and line filtering
    
    Args:
        video_path: Path to video file
        mask_path: Optional ROI mask path
        frame_limit: Maximum number of frames to process
        skip_frames: Number of frames to skip at start
        show_video: Display visualization during processing
        
    Returns:
        Tuple of (vpu, vpv) where each is (x, y) coordinates, or (None, None) if failed
    """
    
    total_start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Estimate VP-u (Road Direction)
    # -------------------------------------------------------------------------
    vpu = estimate_vp_u(
        video,
        min_valid_lines_to_stop=500,
        show_video=show_video
    )
    
    if vpu is None:
        logging.error("Failed to estimate VP-u. Cannot proceed to VP-v estimation.")
        return None, None
    
    logging.info(f"\n✓ VP-u (road direction) found: {vpu}\n")
    
    # -------------------------------------------------------------------------
    # Step 2: Prepare for VP-v Estimation
    # -------------------------------------------------------------------------
    logging.info("Loading video frames for VP-v estimation...")
    d = Detection(video_path, max_frames=frame_limit, color=False, 
                  frame_interval=1, start_frame=skip_frames)
    
    logging.info("Generating foreground masks...")
    d.init_background(method='median')
    fg_masks = d.median_subtract(threshold_value=25).morphology.fill_holes()
    
    # Initialize plate detector
    try:
        plate_detector = PlateDetector(conf_threshold=0.5)
    except Exception as e:
        logging.error(f"Failed to initialize PlateDetector: {e}")
        return vpu, None
    
    # -------------------------------------------------------------------------
    # Step 3: Estimate VP-v (Perpendicular Direction)
    # -------------------------------------------------------------------------
    vpv = estimate_vp_v(
        detection_obj=d,
        fg_masks=fg_masks,
        vp_u=vpu,
        mask_path=mask_path,
        frame_limit=frame_limit,
        plate_detector=plate_detector,
        show_video=show_video
    )
    
    if vpv is None:
        logging.error("Failed to estimate VP-v")
        return vpu, None
    
    logging.info(f"\n✓ VP-v (perpendicular) found: {vpv}\n")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - total_start_time
    
    logging.info("="*80)
    logging.info("DETECTION COMPLETE")
    logging.info("="*80)
    logging.info(f"VP-u (road direction):    {vpu}")
    logging.info(f"VP-v (perpendicular):     {vpv}")
    logging.info(f"Total processing time:    {total_time:.2f} seconds")
    logging.info("="*80)
    
    return vpu, vpv


# =============================================================================
# Main Execution (for testing)
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    # Test configuration
    video_path = os.path.join(PROJECT_ROOT, "assets", "video.avi")
    mask_path = os.path.join(PROJECT_ROOT, "assets", "video_mask.png")
    
    # Run detection
    vpu, vpv = detect_road_and_cross_vps(
        video_path=video_path,
        mask_path=mask_path,
        frame_limit=500,
        skip_frames=0,
        show_video=True
    )
    
    if vpu is not None and vpv is not None:
        logging.info("\nBoth vanishing points successfully detected!")
        logging.info(f"VP-u: {vpu}")
        logging.info(f"VP-v: {vpv}")
    else:
        logging.error("Failed to detect one or both vanishing points")
