import cv2
import numpy as np
import logging
import time
import os
from detect_plate import PlateDetector
from diamond_space import DiamondSpace
from video_stream import video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

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


def estimate_vp_u(min_valid_lines_to_stop=200, show_video=False):
    """
    Estimates the road direction vanishing point (u) using KLT tracking.
    
    Uses the global video stream object to process frames and track features
    using the KLT (Kanade-Lucas-Tomasi) algorithm. Tracks are fitted to lines
    and accumulated until sufficient valid lines are collected.
    
    Args:
        min_valid_lines_to_stop: Stop after collecting this many valid lines (default: 500)
        show_video: Display tracking visualization and save debug video (default: False)
        
    Returns:
        vp_u: (x, y) coordinates of vanishing point u, or None if failed
    """
    logging.info("="*80)
    logging.info("PHASE 1: Estimating VP-u (Road Direction) using KLT Tracking")
    logging.info("="*80)
    
    # Create output directory if show_video is enabled
    video_writer = None
    if show_video:
        output_dir = 'test_output/vp_debug'
        os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    min_dist = 5
    min_frame_displacement=1.0
    min_track_len_filter = 10
    min_track_disp_filter = 100
    roi = video.roi_mask
    
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=min_dist,
        blockSize=7,
        mask=roi
    )
    
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    first_frame_count, old_frame = video.get_frame()
    if old_frame is None:
        logging.error("Could not read first frame")
        return None
    
    # Initialize video writer if show_video is enabled
    if show_video:
        frame_h, frame_w = old_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            'test_output/vp_debug/vp_u_tracking.mp4',
            fourcc, 20.0, (frame_w, frame_h)
        )
    
    p0 = cv2.goodFeaturesToTrack(old_frame, **feature_params)
    
    all_valid_lines = []
    if p0 is not None:
        active_tracks = {i: [p0[i].ravel()] for i in range(len(p0))}
        logging.info(f"Found {len(p0)} initial features to track")
    else:
        active_tracks = {}
    
    next_feature_id = len(active_tracks)
    early_exit = False
    
    while True:
        frame_count, frame = video.get_frame()
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
            vis_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
            for track in active_tracks.values():
                for k in range(len(track) - 1):
                    x1, y1 = map(int, track[k])
                    x2, y2 = map(int, track[k+1])
                    cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                last_pt = track[-1]
                cv2.circle(vis_frame, (int(last_pt[0]), int(last_pt[1])), 5, (0, 0, 255), -1)
            
            # Add text overlay
            cv2.putText(vis_frame, f"Frame: {frame_count} | Lines: {len(all_valid_lines)}/{min_valid_lines_to_stop}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Active Tracks: {len(active_tracks)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to video
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            cv2.imshow('KLT Tracking for VP-u', vis_frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                early_exit = True
                break
        
        old_frame = frame.copy()
        
        if frame_count % 10 == 0 and len(active_tracks) < 50:
            p0_new = cv2.goodFeaturesToTrack(old_frame, **feature_params)
            if p0_new is not None:
                for pt in p0_new:
                    if not any(np.linalg.norm(pt.ravel() - track[-1]) < 5 for track in active_tracks.values()):
                        active_tracks[next_feature_id] = [pt.ravel()]
                        next_feature_id += 1
    
    # Add remaining active tracks
    final_active_tracks = list(active_tracks.values())
    if final_active_tracks:
        new_lines = _fit_lines_to_tracks(final_active_tracks,
                                       min_track_len=min_track_len_filter,
                                       min_track_displacement=min_track_disp_filter)
        if new_lines:
            all_valid_lines.extend(new_lines)
    
    if show_video:
        if video_writer is not None:
            video_writer.release()
            logging.info(f"Saved video to: test_output/vp_debug/vp_u_tracking.mp4")
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process window destruction events
    
    if early_exit:
        logging.info("Early exit requested by user")
    
    logging.info(f"Tracking complete: {frame_count - first_frame_count} frames, {len(all_valid_lines)} valid lines")
    
    if len(all_valid_lines) < 10:
        logging.error(f"Only {len(all_valid_lines)} valid lines. Cannot estimate VP-u.")
        return None
    
    frame_shape = old_frame.shape[:2]
    
    # Use DiamondSpace with 'params' format (lines are already in (a, b, c) format)
    vp_u = _find_vp_from_lines_diamond_space(all_valid_lines, frame_shape, 'params')
    
    if vp_u is not None:
        logging.info(f"VP-u estimated: {tuple(vp_u)}")
        
        # Visualize supporting lines if show_video is enabled
        if show_video:
            _visualize_supporting_lines(
                vp=vp_u,
                all_lines=all_valid_lines,
                line_format='params',
                frame_shape=frame_shape,
                vp_name='u',
                output_prefix='vp_u'
            )
        
        return tuple(vp_u)
    
    return None


# =============================================================================
# VP-V (Perpendicular Direction) Functions - Plate Detection Based
# =============================================================================

def _find_lines_with_hough(frame, gradient_threshold=50):
    """Finds straight line segments using masked gradient."""
    grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    masked_grad = cv2.bitwise_and(grad_mag_norm, grad_mag_norm, mask=video.roi_mask)
    
    _, wireframe = cv2.threshold(masked_grad, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    lines = cv2.HoughLinesP(wireframe, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=20, maxLineGap=2)
    
    raw_lines = []
    if lines is not None:
        for line in lines:
            raw_lines.append(line[0])
    
    return wireframe, raw_lines


def _filter_lines_by_vp(lines, vp_u, angle_threshold_deg=60):
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


def _estimate_plate_angle_from_aspect_ratio(plate_box, known_ratio=5.0):
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


def _get_plate_angle(plate_boxes):
    """Calculates possible angles from detected plate boxes."""
    for box in plate_boxes:
        angles, bbox_aspect = _estimate_plate_angle_from_aspect_ratio(box, known_ratio=5.0)
        if angles is not None and len(angles) > 0:
            return angles, bbox_aspect, box
    return None, None, None


def _filter_lines_by_plate_angles(lines, target_angles, angle_tolerance_deg=15):
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


def _find_supporting_lines(lines, vp, support_max_dist_px=5.0, line_format='segment'):
    """
    Filters lines to return only those that "support" the vanishing point.
    
    Args:
        lines: List of lines in one of two formats:
               - 'segment': [(x1, y1, x2, y2), ...] from HoughLinesP
               - 'params': [(a, b, c), ...] from _fit_lines_to_tracks where ax + by + c = 0
        vp: Vanishing point as (x, y) tuple
        support_max_dist_px: Maximum distance in pixels for a line to be considered supporting
        line_format: 'segment' or 'params' to specify input format
        
    Returns:
        List of supporting lines in the same format as input
    """
    logging.info(f"Finding supporting lines for VP {vp} (distance < {support_max_dist_px}px)...")
    supporting = []
    
    if vp is None:
        return supporting
    
    xv, yv = vp
    
    for line in lines:
        if line_format == 'segment':
            x1, y1, x2, y2 = line
            # Convert to (A, B, C) format for distance calculation
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
        elif line_format == 'params':
            A, B, C = line
        else:
            logging.error(f"Unknown line_format: {line_format}")
            continue
        
        # Calculate distance from point (xv, yv) to line Ax + By + C = 0
        denominator = np.sqrt(A**2 + B**2)
        if denominator == 0:
            continue
        
        distance = abs(A * xv + B * yv + C) / denominator
        
        if distance <= support_max_dist_px:
            supporting.append(line)
    
    logging.info(f"Found {len(supporting)} supporting lines out of {len(lines)}.")
    return supporting


def _visualize_supporting_lines(vp, all_lines, line_format, frame_shape, vp_name, output_prefix):
    """
    Visualizes and saves supporting lines for a vanishing point.
    
    Args:
        vp: Vanishing point as (x, y) tuple
        all_lines: List of all lines used to calculate the VP
        line_format: 'segment' or 'params' to specify line format
        frame_shape: (height, width) of the frame
        vp_name: Name of the vanishing point (e.g., 'u' or 'v')
        output_prefix: Prefix for output files (e.g., 'vp_u' or 'vp_v')
    """
    output_dir = 'test_output/vp_debug'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a frame from video for visualization
    frame_count, frame = video.get_frame()
    if frame is None:
        logging.warning("Could not get frame for supporting lines visualization")
        return
    
    # Convert to BGR for visualization
    if len(frame.shape) == 2:
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis_frame = frame.copy()
    
    frame_h, frame_w = frame_shape
    
    # Calculate distance threshold as 3% of VP distance from image center
    center_x, center_y = frame_w / 2, frame_h / 2
    vp_distance_from_center = np.sqrt((vp[0] - center_x)**2 + (vp[1] - center_y)**2)
    support_threshold = 0.03 * vp_distance_from_center
    logging.info(f"Using support threshold: {support_threshold:.2f}px (3% of VP-{vp_name} distance from center: {vp_distance_from_center:.2f}px)")
    
    # Find supporting lines
    supporting_lines = _find_supporting_lines(all_lines, vp, support_max_dist_px=support_threshold, line_format=line_format)
    
    # Draw supporting lines
    for line in supporting_lines:
        if line_format == 'segment':
            x1, y1, x2, y2 = line
        elif line_format == 'params':
            # Convert from (a, b, c) to segment for drawing
            # We need to find two points on the line within the frame
            a, b, c = line
            
            # Find intersections with frame boundaries
            points = []
            
            # Check intersection with top edge (y=0)
            if b != 0:
                x = -c / a if a != 0 else 0
                if 0 <= x <= frame_w:
                    points.append((int(x), 0))
            
            # Check intersection with bottom edge (y=frame_h)
            if b != 0:
                x = -(c + b * frame_h) / a if a != 0 else 0
                if 0 <= x <= frame_w:
                    points.append((int(x), frame_h))
            
            # Check intersection with left edge (x=0)
            if a != 0:
                y = -c / b if b != 0 else 0
                if 0 <= y <= frame_h:
                    points.append((0, int(y)))
            
            # Check intersection with right edge (x=frame_w)
            if a != 0:
                y = -(c + a * frame_w) / b if b != 0 else 0
                if 0 <= y <= frame_h:
                    points.append((frame_w, int(y)))
            
            # Remove duplicates and take first two points
            points = list(set(points))
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:
                continue
        
        # Extend line for better visualization
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
        cv2.line(vis_frame, (x1_ext, y1_ext), (x2_ext, y2_ext), (255, 0, 0), 2)
    
    # Add text overlay
    cv2.putText(vis_frame, f"VP-{vp_name} Supporting Lines: {len(supporting_lines)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"VP-{vp_name}: ({vp[0]:.1f}, {vp[1]:.1f})", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save the visualization
    output_path = os.path.join(output_dir, f'{output_prefix}_supporting_lines.jpg')
    cv2.imwrite(output_path, vis_frame)
    logging.info(f"Saved supporting lines visualization to: {output_path}")
    
    # Display the visualization
    cv2.imshow(f"VP-{vp_name} Supporting Lines", vis_frame)
    logging.info(f"Displaying VP-{vp_name} supporting lines. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def estimate_vp_v(vp_u, plate_detector, show_video=False):
    """
    Two-phase estimation of perpendicular vanishing point (v):
    Phase 1: Collect plate angles from detected license plates (10 plates)
    Phase 2: Detect lines using Hough transform, filter by VP-u and plate angles, calculate VP-v
    
    Uses the global video stream object to process frames. Lines are filtered to exclude
    those pointing toward VP-u, then further filtered to match angles derived from
    license plate aspect ratios.
    
    Args:
        vp_u: Previously computed VP-u as (x, y) tuple
        plate_detector: PlateDetector instance for license plate detection
        show_video: Display visualization and save debug videos (default: False)
        
    Returns:
        vp_v: (x, y) coordinates of vanishing point v, or None if failed
    """
    logging.info("="*80)
    logging.info("PHASE 2: Estimating VP-v (Perpendicular) using Plate Detection")
    logging.info("="*80)
    
    # Create output directory if show_video is enabled
    video_writer_phase1 = None
    video_writer_phase2 = None
    if show_video:
        output_dir = 'test_output/vp_debug'
        os.makedirs(output_dir, exist_ok=True)
    
    # Phase tracking
    collected_plate_angles = []
    required_good_plates = 10
    good_plate_count = 0
    required_filtered_lines = 2000
    accumulated_filtered_lines = []
    
    # Phase 1: Plate Detection
    logging.info(f"\nPhase 1: Collecting angles from {required_good_plates} plates...")
    
    while good_plate_count < required_good_plates:
        frame_count, frame = video.get_frame_background_subtracted()
        cv2.imshow("Background Subtracted Frame", frame)
        cv2.waitKey(1)
        if frame is None:
            continue
        
        # Initialize video writer for phase 1
        if show_video and video_writer_phase1 is None and frame is not None:
            frame_h, frame_w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_phase1 = cv2.VideoWriter(
                'test_output/vp_debug/vp_v_phase1_plates.mp4',
                fourcc, 20.0, (frame_w, frame_h)
            )
        
        plate_boxes = plate_detector.detect(frame, size=640, save_crops=False)
        
        if len(plate_boxes) == 0:
            if show_video:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis_frame, f"Phase 1: Plates {good_plate_count}/{required_good_plates}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if video_writer_phase1 is not None:
                    video_writer_phase1.write(vis_frame)
                cv2.imshow("Phase 1: Plate Detection", vis_frame)
                cv2.waitKey(1)
            continue
        
        plate_angles, bbox_aspect, plate_box = _get_plate_angle(plate_boxes)
        
        if plate_angles is None:
            if show_video:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Draw detected plates
                for box in plate_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Phase 1: Plates {good_plate_count}/{required_good_plates} (Invalid aspect)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if video_writer_phase1 is not None:
                    video_writer_phase1.write(vis_frame)
                cv2.imshow("Phase 1: Plate Detection", vis_frame)
                cv2.waitKey(1)
            continue
        
        collected_plate_angles.extend(plate_angles)
        good_plate_count += 1
        
        if show_video:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Draw all detected plates
            for box in plate_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Highlight the used plate
            if plate_box is not None:
                x1, y1, x2, y2 = map(int, plate_box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(vis_frame, f"Aspect: {bbox_aspect:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Phase 1: Plates {good_plate_count}/{required_good_plates}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if video_writer_phase1 is not None:
                video_writer_phase1.write(vis_frame)
            # Save periodic snapshots
            if good_plate_count % 2 == 0:
                cv2.imwrite(f'test_output/vp_debug/vp_v_phase1_plate_{good_plate_count:02d}.png', vis_frame)
            cv2.imshow("Phase 1: Plate Detection", vis_frame)
            cv2.waitKey(1)
    
    if show_video and video_writer_phase1 is not None:
        video_writer_phase1.release()
        logging.info(f"Saved Phase 1 video to: test_output/vp_debug/vp_v_phase1_plates.mp4")
    
    if good_plate_count < required_good_plates:
        logging.error(f"Phase 1 incomplete: Only {good_plate_count}/{required_good_plates} plates")
        return None
    logging.info(f"Phase 1 complete: Collected {good_plate_count} plates")
    
    # Phase 2: Line Detection with Filtering
    logging.info(f"\nPhase 2: Line detection with dual filtering...")
    logging.info(f"Using plate angles: {[f'{np.rad2deg(a):.1f}°' for a in collected_plate_angles]}")
        
    
    while len(accumulated_filtered_lines) < required_filtered_lines:
        frame_count, frame = video.get_frame()
        if frame is None:
            continue
        
        # Initialize video writer for phase 2
        if show_video and video_writer_phase2 is None and frame is not None:
            frame_h, frame_w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_phase2 = cv2.VideoWriter(
                'test_output/vp_debug/vp_v_phase2_lines.mp4',
                fourcc, 20.0, (frame_w, frame_h)
            )
        
        # Detect lines
        wireframe, raw_lines = _find_lines_with_hough(frame, gradient_threshold=50)
        
        # Apply dual filtering
        vp_filtered = _filter_lines_by_vp(raw_lines, vp_u, angle_threshold_deg=45)
        dual_filtered = _filter_lines_by_plate_angles(vp_filtered, collected_plate_angles, angle_tolerance_deg=15)
        
        accumulated_filtered_lines.extend(dual_filtered)
        
        if show_video:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Draw all accumulated lines (persistent display)
            for (x1, y1, x2, y2) in accumulated_filtered_lines:
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(vis_frame, f"Phase 2: Lines {len(accumulated_filtered_lines)}/{required_filtered_lines}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Frame: {frame_count} | This frame: {len(dual_filtered)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to video
            if video_writer_phase2 is not None:
                video_writer_phase2.write(vis_frame)
            
            cv2.imshow("Phase 2: Line Detection", vis_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                logging.info("Early exit requested - stopping Phase 2")
                break
    
    if show_video:
        if video_writer_phase2 is not None:
            video_writer_phase2.release()
            logging.info(f"Saved Phase 2 video to: test_output/vp_debug/vp_v_phase2_lines.mp4")
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process window destruction events
    
    logging.info(f"Phase 2 complete: {len(accumulated_filtered_lines)} dual-filtered lines")
    
    if not accumulated_filtered_lines:
        logging.error("No valid lines after filtering")
        return None
    
    # Calculate VP-v using 'segment' format (lines are in (x1, y1, x2, y2) format from HoughLinesP)
    canvas_shape = frame.shape[:2]
    vp_v = _find_vp_from_lines_diamond_space(accumulated_filtered_lines, canvas_shape, 'segment')
    
    if vp_v is not None:
        logging.info(f"VP-v estimated: {vp_v}")
        
        # Visualize supporting lines if show_video is enabled
        if show_video:
            _visualize_supporting_lines(
                vp=vp_v,
                all_lines=accumulated_filtered_lines,
                line_format='segment',
                frame_shape=canvas_shape,
                vp_name='v',
                output_prefix='vp_v'
            )
    
    return vp_v

# =============================================================================
# Main Unified Interface
# =============================================================================

def detect_road_and_cross_vps(show_video=False):
    """
    Detect both vanishing points from traffic video.
    
    This is the main entry point that orchestrates the complete vanishing point detection:
    1. Estimates VP-u (road direction) using KLT tracking of features
    2. Estimates VP-v (perpendicular) using license plate detection and line filtering
    
    Uses the global video stream object for frame access. The video stream should be
    initialized before calling this function.
    
    Args:
        show_video: Display visualization during processing and save debug videos (default: False)
        
    Returns:
        Tuple of (vpu, vpv) where each is (x, y) coordinates, or (None, None) if failed
    """
    total_start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Estimate VP-u (Road Direction)
    # -------------------------------------------------------------------------
    video.set_intended_fps(30)
    vpu = estimate_vp_u(
        show_video=show_video
    )
    
    if vpu is None:
        logging.error("Failed to estimate VP-u. Cannot proceed to VP-v estimation.")
        return None, None
    
    logging.info(f"\n✓ VP-u (road direction) found: {vpu}\n")
    
    # -------------------------------------------------------------------------
    # Step 2: Prepare for VP-v Estimation
    # -------------------------------------------------------------------------
    # Initialize plate detector
    try:
        plate_detector = PlateDetector(conf_threshold=0.5)
    except Exception as e:
        logging.error(f"Failed to initialize PlateDetector: {e}")
        return vpu, None
    
    # -------------------------------------------------------------------------
    # Step 3: Estimate VP-v (Perpendicular Direction)
    # -------------------------------------------------------------------------
    video.set_intended_fps(10)
    vpv = estimate_vp_v(
        vp_u=vpu,
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
    total_time = time.perf_counter() - total_start_time
    
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
    detect_road_and_cross_vps(show_video=True)
