import cv2
import numpy as np
import logging
import time
import os
from detect_plate import PlateDetector
from diamond_space import DiamondSpace
from video_stream import video
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def _find_vp_from_lines_diamond_space(lines, frame_shape, line_format, reference_vp=None, save_graph=True, graph_name='diamond_space'):
    """
    Finds vanishing point using DiamondSpace accumulator.
    
    Args:
        lines: List of lines in one of two formats:
               - 'segment': [(x1, y1, x2, y2), ...] from HoughLinesP
               - 'params': [(a, b, c), ...] from _fit_lines_to_tracks where ax + by + c = 0
        frame_shape: (height, width) of the frame
        line_format: 'segment' or 'params' to specify input format
        reference_vp: Optional reference vanishing point (x, y) for horizontal opposition filtering.
                     When provided, ensures returned VP is on opposite horizontal side of image center.
        save_graph: Whether to save the DiamondSpace accumulator visualization (default: True)
        graph_name: Name for the saved graph file (default: 'diamond_space')
        
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
    
    # Visualize and save DiamondSpace accumulator
    if save_graph and plt is not None:
        try:
            logging.info("Generating DiamondSpace accumulator visualization...")
            A_img = DS.attach_spaces()
            extent = ((-DS.size + 0.5) / DS.scale, (DS.size - 0.5) / DS.scale,
                      (DS.size - 0.5) / DS.scale, (-DS.size + 0.5) / DS.scale)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(A_img, cmap="Greys", extent=extent)
            
            # Plot peaks if available
            if p_ds is not None and len(p_ds) > 0:
                ax.plot(p_ds[:, 0] / DS.scale, p_ds[:, 1] / DS.scale, "r+", alpha=0.8, markersize=12, mew=2)
            
            ax.set_title(f"Diamond Space Accumulator - {graph_name}\n(Red+ = Peaks)", fontsize=12, fontweight='bold')
            ax.set_xlabel("X coordinate", fontsize=10)
            ax.set_ylabel("Y coordinate", fontsize=10)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Save the plot
            output_dir = 'test_output/vp_debug'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f'{graph_name}_accumulator.jpg')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved DiamondSpace graph to: {plot_path}")
            plt.close(fig)
        except Exception as e:
            logging.warning(f"Failed to save DiamondSpace graph: {e}")
    
    # If reference_vp provided, filter peaks for horizontal opposition
    if reference_vp is not None:
        center_x = img_w / 2
        ref_x = reference_vp[0]
        
        # Determine which side reference VP is on
        ref_is_left = ref_x < center_x
        
        logging.info(f"Applying horizontal opposition filter:")
        logging.info(f"  Image center x: {center_x:.1f}")
        logging.info(f"  Reference VP x: {ref_x:.1f} ({'left' if ref_is_left else 'right'} of center)")
        logging.info(f"  Looking for VP on {'right' if ref_is_left else 'left'} side")
        
        # Find first peak on opposite side
        for i, (peak, weight) in enumerate(zip(p, w)):
            peak_x = peak[0]
            peak_is_left = peak_x < center_x
            
            # Check if peak is on opposite side from reference
            if ref_is_left and not peak_is_left:  # ref is left, peak is right
                logging.info(f"  Peak {i}: x={peak_x:.1f} (right) - ACCEPTED with weight {weight:.2f}")
                best_peak_xy = peak[:2].astype(np.float32)
                return best_peak_xy
            elif not ref_is_left and peak_is_left:  # ref is right, peak is left
                logging.info(f"  Peak {i}: x={peak_x:.1f} (left) - ACCEPTED with weight {weight:.2f}")
                best_peak_xy = peak[:2].astype(np.float32)
                return best_peak_xy
            else:
                logging.info(f"  Peak {i}: x={peak_x:.1f} ({'left' if peak_is_left else 'right'}) - REJECTED (same side as reference)")
        
        logging.warning("No peaks found on opposite side of reference VP")
        return None
    else:
        # No filtering, return best peak
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
    if old_frame is None or first_frame_count is None:
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
        if frame is None or frame_count is None:
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
                    # Check if the tracked point is within the ROI mask
                    x, y = int(pt_new[0, 0]), int(pt_new[0, 1])
                    
                    # Check if point is within frame bounds
                    if 0 <= y < roi.shape[0] and 0 <= x < roi.shape[1]:
                        # Check if point is within the mask (non-zero value means inside ROI)
                        if roi[y, x] == 0:
                            # Point moved outside the ROI, complete the track
                            if len(track) > 1:
                                newly_completed_tracks.append(track)
                            continue
                    else:
                        # Point moved outside frame bounds, complete the track
                        if len(track) > 1:
                            newly_completed_tracks.append(track)
                        continue
                    
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    vp_u = _find_vp_from_lines_diamond_space(all_valid_lines, frame_shape, 'params', 
                                             save_graph=show_video, graph_name='vp_u')
    
    if vp_u is not None:
        logging.info(f"VP-u estimated: {tuple(vp_u)}")
        
        # Visualize supporting lines if show_video is enabled
        # if show_video:
        #     _visualize_supporting_lines(
        #         vp=vp_u,
        #         all_lines=all_valid_lines,
        #         line_format='params',
        #         frame_shape=frame_shape,
        #         vp_name='u',
        #         output_prefix='vp_u'
            # )
        
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


def _detect_plate_edges_with_hough(frame, plate_box, min_length_ratio=0.90):
    """
    Detects plate edges using HoughLinesP on the gradient within the plate region.
    Computes gradient only for the cropped plate region.
    
    Args:
        frame: Original grayscale frame
        plate_box: Plate bounding box as (x1, y1, x2, y2)
        min_length_ratio: Minimum line length as ratio of plate width (default: 0.95)
        
    Returns:
        detected_lines: List of line segments in frame coordinates [(x1, y1, x2, y2), ...]
        detected_angles: List of angles in radians for each line
        plate_edges: Thresholded gradient (wireframe) of the plate region
        grad_mag_norm_crop: Normalized gradient magnitude of the plate region
    """
    x1, y1, x2, y2 = map(int, plate_box)
    plate_width = x2 - x1
    plate_height = y2 - y1
    
    if plate_width <= 0 or plate_height <= 0:
        return [], [], None, None
    
    # Crop plate region from frame first
    plate_crop = frame[y1:y2, x1:x2]
    
    # Compute gradient on the cropped plate
    grad_x = cv2.Sobel(plate_crop, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(plate_crop, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag_norm_crop = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Threshold the gradient to get edges
    _, plate_edges = cv2.threshold(grad_mag_norm_crop, 50, 255, cv2.THRESH_BINARY)
    
    # Run HoughLinesP to detect line segments directly
    min_line_length = int(min_length_ratio * plate_width)
    lines = cv2.HoughLinesP(plate_edges, rho=1, theta=np.pi/180, threshold=int(plate_width * 0.3),
                            minLineLength=min_line_length, maxLineGap=1)
    
    if lines is None:
        return [], [], plate_edges, grad_mag_norm_crop
    
    detected_lines = []
    detected_angles = []
    
    for line in lines:
        x1_local, y1_local, x2_local, y2_local = line[0]
        
        # Convert back to frame coordinates
        x1_frame = x1_local + x1
        y1_frame = y1_local + y1
        x2_frame = x2_local + x1
        y2_frame = y2_local + y1
        
        detected_lines.append((x1_frame, y1_frame, x2_frame, y2_frame))
        
        # Calculate angle (normalized to [0, pi))
        dx = x2_frame - x1_frame
        dy = y2_frame - y1_frame
        angle = np.arctan2(dy, dx) % np.pi
        logging.info(f"Detected plate angle: {angle}")
        detected_angles.append(angle)
    
    return detected_lines, detected_angles, plate_edges, grad_mag_norm_crop


def _get_plate_angle_from_hough(frame, plate_boxes, min_length_ratio=0.90):
    """
    Extracts plate angles from HoughLinesP detection on plate regions.
    Computes gradient separately for each plate.
    
    Args:
        frame: Original grayscale frame
        plate_boxes: List of detected plate bounding boxes
        min_length_ratio: Minimum line length as ratio of plate width
        
    Returns:
        detected_angles: List of detected angles from plate edges (radians)
        plate_lines: List of detected line segments
        plate_box: The plate box that was analyzed
        all_data: List of (box, wireframe, grad_mag_norm_crop) tuples for all processed plates
    """
    all_data = []
    
    for box in plate_boxes:
        x1, y1, x2, y2 = box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        if bbox_width <= 0 or bbox_height <= 0:
            continue
        
        # Run HoughLinesP on this plate (gradient computed inside)
        plate_lines, detected_angles, wireframe, grad_mag_norm_crop = _detect_plate_edges_with_hough(
            frame, box, min_length_ratio=min_length_ratio
        )
        
        # Store data for all plates
        if wireframe is not None and grad_mag_norm_crop is not None:
            all_data.append((box, wireframe, grad_mag_norm_crop))
        
        if len(detected_angles) > 0:
            logging.info(f"Detected {len(detected_angles)} plate edges with HoughLinesP (>={min_length_ratio*100:.0f}% of plate width)")
            return detected_angles, plate_lines, box, all_data
    
    return None, None, None, all_data


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
    if frame is None or frame_count is None:
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
    support_threshold = 0.01 * vp_distance_from_center
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
    
    Phase 1: Collect plate edges and generate intermediate VP-v (vpv_plate)
    - Detects 20 plate edges with length >= 85% of plate width
    - Uses conf_threshold=0.5 for plate detection
    - Generates vpv_plate from these edges
    
    Phase 2: Filter general edgelets one at a time and refine VP-v
    - Detects general edgelets using HoughLinesP
    - Filters EACH edgelet individually by angle test
    - For each edgelet: finds furthest point from vpv_plate,
      calculates angle between (vpv_plate->furthest_point) and edgelet direction
    - Keeps edgelet if angle < 5 degrees
    - Accumulates until 2000 filtered edgelets
    - Generates final VP-v from filtered edgelets
    
    Args:
        vp_u: Previously computed VP-u as (x, y) tuple
        plate_detector: PlateDetector instance for license plate detection
        show_video: Display visualization and save debug videos (default: False)
        
    Returns:
        vp_v: (x, y) coordinates of vanishing point v, or None if failed
    """
    logging.info("="*80)
    logging.info("PHASE 2: Two-Phase VP-v Estimation (Plate Edges + Filtered Edgelets)")
    logging.info("="*80)
    
    # Create output directory if show_video is enabled
    video_writer_phase1 = None
    if show_video:
        output_dir = 'test_output/vp_debug'
        os.makedirs(output_dir, exist_ok=True)
        # Create gradient subdirectory
        gradient_dir = os.path.join(output_dir, 'gradient')
        os.makedirs(gradient_dir, exist_ok=True)
    
    # Phase 1 parameters
    collected_plate_edges = []
    min_edges_threshold = 100
    min_length_ratio = 0.85  # 85% of plate width
    
    # =========================================================================
    # PHASE 1: Plate Edge Collection and VPV_PLATE Generation
    # =========================================================================
    logging.info(f"\nPhase 1: Collecting {min_edges_threshold} plate edges (min_length: {min_length_ratio*100:.0f}% of plate_width)...")
    
    while len(collected_plate_edges) < min_edges_threshold:
        frame_count, frame = video.get_frame_background_subtracted()
        if frame is None or frame_count is None:
            logging.warning("Reached end of video during plate edge collection")
            break
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
                cv2.putText(vis_frame, f"Phase 1: Plate Edges {len(collected_plate_edges)}/{min_edges_threshold}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if video_writer_phase1 is not None:
                    video_writer_phase1.write(vis_frame)
                cv2.imshow("Phase 1: Plate Edge Detection", vis_frame)
                cv2.waitKey(1)
            continue
        
        # Extract edges using HoughLinesP on plate edges with 85% min_length
        # Gradient is computed separately for each plate inside _get_plate_angle_from_hough
        plate_angles, plate_lines, plate_box, all_data = _get_plate_angle_from_hough(
            frame, plate_boxes, min_length_ratio=min_length_ratio
        )
        
        if plate_lines is None or len(plate_lines) == 0:
            if show_video:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Draw detected plates
                for box in plate_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Phase 1: Plate Edges {len(collected_plate_edges)}/{min_edges_threshold} (No edges)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if video_writer_phase1 is not None:
                    video_writer_phase1.write(vis_frame)
                cv2.imshow("Phase 1: Plate Edge Detection", vis_frame)
                cv2.waitKey(1)
            continue
        
        # Collect the plate edge lines
        collected_plate_edges.extend(plate_lines)
        logging.info(f"Collected {len(plate_lines)} edges from plate, total: {len(collected_plate_edges)}")
        
        if show_video:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Draw all detected plates
            for box in plate_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Highlight the used plate and draw detected edges
            if plate_box is not None:
                x1, y1, x2, y2 = map(int, plate_box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
                # Draw detected plate edge lines (magenta)
                for line in plate_lines:
                    lx1, ly1, lx2, ly2 = map(int, line)
                    cv2.line(vis_frame, (lx1, ly1), (lx2, ly2), (255, 0, 255), 2)
                
                # Display number of detected edges
                cv2.putText(vis_frame, f"Edges: {len(plate_lines)}", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Save individual plate image with detected edges, gradient, and wireframe
                plate_crop = vis_frame[y1:y2, x1:x2].copy()
                if plate_crop.size > 0:
                    # Get the gradient and wireframe for this plate from all_data
                    grad_plate_crop = None
                    wireframe_crop = None
                    for box, wireframe, grad in all_data:
                        if np.array_equal(box, plate_box):
                            wireframe_crop = wireframe
                            grad_plate_crop = grad
                            break
                    
                    if grad_plate_crop is None:
                        continue
                    
                    grad_plate_crop_bgr = cv2.cvtColor(grad_plate_crop, cv2.COLOR_GRAY2BGR)
                    
                    # Add edge information to the crops
                    plate_h, plate_w = plate_crop.shape[:2]
                    info_img = np.zeros((60, plate_w, 3), dtype=np.uint8)
                    
                    # Add edge count text
                    text = f"Edges: {len(plate_lines)} (>={min_length_ratio*100:.0f}% width)"
                    cv2.putText(info_img, text, (5, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    text = f"Plate width: {x2-x1}px, min_length: {int((x2-x1)*min_length_ratio)}px"
                    cv2.putText(info_img, text, (5, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Combine plate crop with info
                    plate_with_info = np.vstack([plate_crop, info_img])
                    
                    # Combine gradient crop with same info
                    grad_with_info = np.vstack([grad_plate_crop_bgr, info_img.copy()])
                    
                    # Combine wireframe with same info if available
                    images_to_combine = [plate_with_info, grad_with_info]
                    if wireframe_crop is not None:
                        wireframe_crop_bgr = cv2.cvtColor(wireframe_crop, cv2.COLOR_GRAY2BGR)
                        wireframe_with_info = np.vstack([wireframe_crop_bgr, info_img.copy()])
                        images_to_combine.append(wireframe_with_info)
                    
                    # Save all images side by side
                    combined = np.hstack(images_to_combine)
                    plate_output_path = f'test_output/vp_debug/plate_{len(collected_plate_edges):03d}_edges.png'
                    cv2.imwrite(plate_output_path, combined)
                    logging.info(f"Saved plate with edges, gradient, and wireframe to: {plate_output_path}")
                    
                    # Also save gradient separately
                    grad_output_path = f'test_output/vp_debug/gradient/gradient_frame_{frame_count:04d}.jpg'
                    cv2.imwrite(grad_output_path, grad_plate_crop)
                    
                    # Display combined visualization
                    cv2.imshow('Plate Analysis (Plate | Gradient | Wireframe)', combined)
                    cv2.waitKey(1)
            
            cv2.putText(vis_frame, f"Phase 1: Plate Edges {len(collected_plate_edges)}/{min_edges_threshold}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if video_writer_phase1 is not None:
                video_writer_phase1.write(vis_frame)
            cv2.imshow("Phase 1: Plate Edge Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Early exit requested - Phase 1")
                break
    
    if show_video:
        if video_writer_phase1 is not None:
            video_writer_phase1.release()
            logging.info(f"Saved Phase 1 video to: test_output/vp_debug/vp_v_phase1_plates.mp4")
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    if len(collected_plate_edges) < min_edges_threshold:
        logging.error(f"Phase 1 failed: Only {len(collected_plate_edges)}/{min_edges_threshold} plate edges collected")
        return None
    
    logging.info(f"✓ Phase 1 complete: {len(collected_plate_edges)} plate edges collected")
    
    # Check if we have a valid frame for canvas shape
    if frame is None:
        logging.error("No valid frame available for canvas shape")
        return None
    
    # Generate vpv_plate from plate edges (no horizontal opposition filter)
    canvas_shape = frame.shape[:2]
    vpv_plate = _find_vp_from_lines_diamond_space(
        collected_plate_edges, 
        canvas_shape, 
        'segment',
        reference_vp=vp_u,  # No opposition filter for intermediate VP
        save_graph=show_video,
        graph_name='vp_v_phase1_plate_edges'
    )
    
    if vpv_plate is None:
        logging.error("Failed to generate vpv_plate from plate edges")
        return None
    
    logging.info(f"✓ Generated vpv_plate (intermediate): {vpv_plate}")
    
    return vpv_plate

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
    # vpu = (np.float32(1996.7653), np.float32(-516.6893))
    
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
    # vpv = np.array([-83322, 864])
    
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
