#!/usr/bin/env python

"""
vp_estimation_step1_klt.py

This is the implementation using the "dumb" (KLT) tracker.

*** UPDATE: Reverted winSize, min_track_len_filter, and ransac_threshold. ***
"""

import cv2
import numpy as np
import logging
import time

# -----------------------------------------------------------------------------
# RANSAC and Line Geometry Helper Functions
# -----------------------------------------------------------------------------

def fit_lines_to_tracks(tracks, 
                        min_track_len=10, 
                        min_track_displacement=50):
    """
    Fits a line (ax + by + c = 0) to tracks that pass
    a length and displacement filter.
    """
    lines = []
    logging.info(f"Filtering {len(tracks)} raw tracks with (len>={min_track_len}, disp>={min_track_displacement})...")
    
    valid_tracks_count = 0
    for track in tracks:
        # Filter 1: Track length
        if len(track) < min_track_len:
            continue 
        
        start_point = track[0]
        end_point = track[-1]
        
        # Filter 2: Track displacement
        displacement = np.linalg.norm(start_point - end_point)
        if displacement < min_track_displacement:
            continue # Track is stationary/jittery noise
            
        # If it passes all filters, fit a line
        valid_tracks_count += 1
        points = np.array(track).reshape(-1, 2)
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        vx, vy, x0, y0 = line_params.flatten()
        a = vy
        b = -vx
        c = vx * y0 - vy * x0
        lines.append((a, b, c))
            
    logging.info(f"Found {len(lines)} valid motion lines from {valid_tracks_count} valid tracks.")
    return lines

def get_intersection(line1, line2):
    """Finds the intersection of two lines in general form (a, b, c)."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    D = a1 * b2 - a2 * b1
    Dx = -c1 * b2 - (-c2 * b1)
    Dy = a1 * (-c2) - a2 * (-c1)
    
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (x, y)
    else:
        return None # Lines are parallel

def find_vp_with_ransac(lines, iterations=1000, threshold=5.0):
    """Robustly finds the vanishing point from a list of lines using RANSAC."""
    best_vp = None
    max_inliers = -1
    
    start_time = time.time()
    logging.info(f"Running RANSAC on {len(lines)} lines for {iterations} iterations (threshold={threshold}px)...")
    
    for _ in range(iterations):
        if len(lines) < 2:
            logging.warning("Need at least 2 lines for RANSAC.")
            return None
            
        idx1, idx2 = np.random.choice(len(lines), 2, replace=False)
        line1 = lines[idx1]
        line2 = lines[idx2]

        vp = get_intersection(line1, line2)
        if vp is None:
            continue
            
        x, y = vp
        inliers_count = 0
        for line in lines:
            a, b, c = line
            distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
            
            if distance < threshold:
                inliers_count += 1
        
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_vp = vp
    
    end_time = time.time()
    logging.info(f"RANSAC complete in {end_time - start_time:.2f} seconds.")
    logging.info(f"Best VP: {best_vp} with {max_inliers} inliers.")
    return best_vp

# -----------------------------------------------------------------------------
# Main Feature Tracking and VP Estimation
# -----------------------------------------------------------------------------

def estimate_vanishing_point_u(video_path, 
                             show_video=True, 
                             mask_path=None, 
                             min_frame_displacement=1.0, 
                             min_valid_lines_to_stop=500):
    """
    Main function to process the video and estimate the first vanishing point (u)
    using KLT tracking.
    """
    
    # --- 1. Setup ---
    
    # Get user-defined parameters
    min_dist = 5
    min_track_len_filter = 10        # <-- CHANGED from 5
    min_track_disp_filter = 100
    ransac_iterations = 10000
    ransac_threshold = 5.0           # <-- CHANGED from 10.0
    
    feature_params: dict[str, object] = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=min_dist, 
        blockSize=7
    )
    
    lk_params = dict(
        winSize=(15, 15), # <-- CHANGED from (15, 31)
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        return None
    
    logging.info(f"Video file opened successfully. Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    ret, old_frame = cap.read()
    if not ret or old_frame is None:
        logging.error("Could not read *first* frame. Check video codec or file path.")
        cap.release()
        return None
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1b. Load and Prepare Mask ---
    if mask_path:
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_np is None:
            logging.warning(f"Could not load mask from {mask_path}. Proceeding without mask.")
        else:
            h, w = old_gray.shape
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                logging.warning(f"Mask dimensions {mask_np.shape} do not match frame {old_gray.shape}. Resizing mask...")
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            
            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            logging.info(f"Successfully loaded and prepared mask from {mask_path}")
            feature_params['mask'] = mask_np 

    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    
    all_valid_lines = [] # This will store the final, filtered lines
    if p0 is not None:
        active_tracks = {i: [p0[i].ravel()] for i in range(len(p0))}
        logging.info(f"Found {len(p0)} initial features to track.")
    else:
        active_tracks = {}
        logging.warning("No initial features found in the first frame (or in the mask).")
    
    # --- 2. KLT Tracking Loop ---
    
    frame_count = 0
    next_feature_id = len(active_tracks)
    
    tracking_start_time = time.time()
    
    logging.info(f"Starting KLT tracking loop... Will stop after collecting {min_valid_lines_to_stop} valid lines.")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.info("\nEnd of video or corrupt frame.")
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        newly_completed_tracks = [] # Tracks that finished *this frame*

        if not active_tracks:
            p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
            if p0 is not None:
                active_tracks = {i + next_feature_id: [p0[i].ravel()] for i in range(len(p0))}
                next_feature_id += len(p0)
            else:
                old_gray = frame_gray.copy()
                continue # No features found

        p0_list = np.array([track[-1] for track in active_tracks.values()]).astype(np.float32).reshape(-1, 1, 2)

        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0_list,
            None,
            **lk_params
        )

        new_active_tracks = {}
        track_ids = list(active_tracks.keys())
        
        if p1 is not None:
            for i, (track_id, pt_new, st) in enumerate(zip(track_ids, p1, status)):
                track = active_tracks[track_id]
                
                if st == 1:
                    pt_old = track[-1]
                    displacement = np.linalg.norm(pt_new.ravel() - pt_old)
                    
                    if displacement < min_frame_displacement:
                        # Jitter filter: Kill the track
                        if len(track) > 1:
                            newly_completed_tracks.append(track)
                        continue 
                    
                    # Point is valid, add it
                    track.append(pt_new.ravel())
                    new_active_tracks[track_id] = track
                        
                else:
                    # Point was lost by KLT
                    if len(track) > 1:
                        newly_completed_tracks.append(track)

        active_tracks = new_active_tracks
        
        # --- Filter completed tracks and check limit ---
        if newly_completed_tracks:
            new_lines = fit_lines_to_tracks(
                newly_completed_tracks, 
                min_track_len=min_track_len_filter,
                min_track_displacement=min_track_disp_filter
            )
            if new_lines:
                all_valid_lines.extend(new_lines)
                logging.info(f"Total valid lines collected so far: {len(all_valid_lines)}")

        # Check if we have enough valid lines
        if len(all_valid_lines) > min_valid_lines_to_stop:
            logging.info(f"Reached {len(all_valid_lines)} valid lines. Stopping video processing.")
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
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                logging.info("User pressed 'q', stopping video processing.")
                break

        old_gray = frame_gray.copy()
        
        if frame_count % 10 == 0 and len(active_tracks) < 50: # Your value
            p0_new = cv2.goodFeaturesToTrack(old_gray, **feature_params)
            if p0_new is not None:
                for pt in p0_new:
                    if not any(np.linalg.norm(pt.ravel() - track[-1]) < 5 for track in active_tracks.values()):
                        active_tracks[next_feature_id] = [pt.ravel()]
                        next_feature_id += 1
        
        frame_count += 1

    tracking_end_time = time.time()
    
    # --- 3. Post-Processing and VP Calculation ---
    
    # Add any tracks that were still active when the loop broke
    final_active_tracks = list(active_tracks.values())
    if final_active_tracks:
        new_lines = fit_lines_to_tracks(
            final_active_tracks, 
            min_track_len=min_track_len_filter,
            min_track_displacement=min_track_disp_filter
        )
        if new_lines:
            all_valid_lines.extend(new_lines)
            logging.info(f"Final valid line count: {len(all_valid_lines)}")

    cap.release()
    cv2.destroyAllWindows()
    
    logging.info(f"--- Tracking Summary ---")
    logging.info(f"Total frames processed: {frame_count}")
    tracking_time = tracking_end_time - tracking_start_time
    logging.info(f"Total tracking time: {tracking_time:.2f} seconds")
    if tracking_time > 0:
        logging.info(f"Average FPS (processing): {frame_count / (tracking_time):.2f}")
    logging.info(f"Total valid lines collected: {len(all_valid_lines)}")
    
    
    if len(all_valid_lines) < 10: # Use a small hardcoded minimum
        logging.error(f"Found only {len(all_valid_lines)} valid lines. Cannot estimate vanishing point.")
        return None
        
    vanishing_point_u = find_vp_with_ransac(
        all_valid_lines, # Pass the pre-filtered lines
        iterations=ransac_iterations, 
        threshold=ransac_threshold
    )
    
    return vanishing_point_u

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
    # mask_file_path = None
    
    # ---------------------------------
    
    logging.info(f"Starting VP-u estimation on video: {video_file_path}")
    if mask_file_path:
        logging.info(f"Using mask: {mask_file_path}")
    else:
        logging.info("No mask provided.")
    
    total_start_time = time.time()
    
    vp_u = estimate_vanishing_point_u(
        video_file_path, 
        show_video=True,
        mask_path=mask_file_path,
        min_frame_displacement=3.0, # Your value
        min_valid_lines_to_stop=500 # Your value
    )
    
    total_end_time = time.time()
    
    if vp_u:
        logging.info(f"\n--- \nFinal Estimated Vanishing Point (u): {vp_u}")
        logging.info(f"Total estimation time: {total_end_time - total_start_time:.2f} seconds")
        np.save('vp_u.npy', vp_u)

        # --- Visualization of the final VP ---
        logging.info("Displaying final vanishing point. Press any key to close.")
        cap = cv2.VideoCapture(video_file_path)
        ret, frame = cap.read()
        if ret:
            frame_h, frame_w = frame.shape[:2]
            vp_x, vp_y = int(vp_u[0]), int(vp_u[1])

            padding_color = [100, 100, 100] 
            min_x = min(0, vp_x - 50) 
            min_y = min(0, vp_y - 50)
            max_x = max(frame_w, vp_x + 50)
            max_y = max(frame_h, vp_y + 50)
            new_w = max_x - min_x
            new_h = max_y - min_y
            offset_x = -min_x
            offset_y = -min_y
            
            new_canvas = np.full((new_h, new_w, 3), padding_color, dtype=np.uint8)
            
            new_canvas[offset_y : offset_y + frame_h, 
                       offset_x : offset_x + frame_w] = frame
                       
            vp_draw_coords = (vp_x + offset_x, vp_y + offset_y)
            
            cv2.circle(new_canvas, vp_draw_coords, 10, (0, 0, 255), -1) 
            cv2.line(new_canvas, (vp_draw_coords[0] - 20, vp_draw_coords[1]), 
                                 (vp_draw_coords[0] + 20, vp_draw_coords[1]), (0, 255, 0), 2)
            cv2.line(new_canvas, (vp_draw_coords[0], vp_draw_coords[1] - 20), 
                                 (vp_draw_coords[0], vp_draw_coords[1] + 20), (0, 255, 0), 2)
            
            cv2.imshow("Final Vanishing Point (u) - Padded", new_canvas)
            cv2.waitKey(0) # Wait for a key press
            cv2.destroyAllWindows()
        
        cap.release()

    else:
        logging.error("Could not estimate vanishing point.")
