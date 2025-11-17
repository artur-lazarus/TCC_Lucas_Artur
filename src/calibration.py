import os
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

import detection
import optical_flow
import vp_detector
from video_stream import video
import background

import homography
import roi_maker
import time

def select_greedy_hue_window(counts, window_size=None, coverage_threshold=None):
    """Select circular window of bins by window size or coverage threshold.
    
    Parameters
    ----------
    counts : array-like
        Histogram weights per bin.
    window_size : int, optional
        Fixed window size (number of bins). Greedily adds highest bins while arc <= window_size.
    coverage_threshold : float, optional
        Fraction of total weight [0,1]. Finds minimal contiguous window meeting threshold.
    
    Returns
    -------
    start_idx, end_idx, chosen_bins : int, int, list
        Start/end indices (inclusive, wraps if start>end) and chosen bin indices.
    
    Notes
    -----
    Exactly one of window_size or coverage_threshold must be provided.
    """
    
    counts = np.asarray(counts, dtype=float)
    N = counts.size
    if N == 0:
        return 0, -1, []
    
    if window_size is not None:
        # Fixed-size mode: greedy selection
        order = np.argsort(counts)[::-1]
        selected = []
        
        def minimal_arc(idxs):
            if not idxs:
                return 0, 0, 0
            arr = np.sort(np.array(idxs, dtype=int))
            diffs = np.diff(np.r_[arr, arr[0] + N])
            max_gap_idx = int(np.argmax(diffs))
            length = N - diffs[max_gap_idx]
            start = (arr[(max_gap_idx + 1) % arr.size]) % N
            end = (start + length - 1) % N
            return int(length), int(start), int(end)
        
        best_len, best_start, best_end = 0, 0, -1
        for idx in order:
            length, start, end = minimal_arc(selected + [int(idx)])
            if length <= int(window_size):
                selected.append(int(idx))
                best_len, best_start, best_end = length, start, end
            else:
                break
        
        def in_window(i, s, e):
            return (s <= e and s <= i <= e) or (s > e and (i >= s or i <= e))
        
        chosen = [i for i in selected if in_window(i, best_start, best_end)]
        return best_start, best_end, chosen
    
    # Coverage mode: minimal contiguous window
    total = float(counts.sum())
    if total <= 0.0:
        return 0, -1, []
    
    target = float(coverage_threshold) * total
    counts2 = np.concatenate([counts, counts])
    best_len, best_start, best_end = N + 1, 0, -1
    cur_sum, j = 0.0, 0
    
    for i in range(N):
        while j < i + N and cur_sum < target:
            cur_sum += counts2[j]
            j += 1
        if cur_sum >= target:
            length = j - i
            if length < best_len:
                best_len, best_start, best_end = length, i % N, (j - 1) % N
        cur_sum -= counts2[i]
    
    if best_len == N + 1:
        return 0, -1, []
    
    if best_start <= best_end:
        chosen = list(range(best_start, best_end + 1))
    else:
        chosen = list(range(best_start, N)) + list(range(0, best_end + 1))
    
    return best_start, best_end, chosen


def get_main_movement_range(n_frames, coverage_threshold=None, window_size=None, magnitude_threshold=2.0):
    """Find dominant flow orientation range via weighted histogram.
    
    Parameters
    ----------
    n_frames : int
        Number of frames for the calculation.
    flows_polar : list of (magnitude, angle) tuples
        Polar coordinates (mag, ang in radians) for each flow frame.
    coverage_threshold : float
        Fraction of total flow weight to capture (default 0.9).
    magnitude_threshold : float
        Minimum magnitude to consider (default 2.0).
    
    Returns
    -------
    start_angle, end_angle : float
        Range in radians [0, π).
    chosen_bins : list of int
        Histogram bins in selected window.
    """

    dis_preset="FAST"

    if (window_size is None) == (coverage_threshold is None):
        raise ValueError("Exactly one of window_size or coverage_threshold must be provided")
    
    angle_bins = np.zeros(90, dtype=np.float64)
    prev = video.get_frame()
    for i in range(n_frames):
        curr = video.get_frame()
        flow_polar_magnitude, flow_polar_angle = optical_flow.flow_to_polar(optical_flow.calculate_optical_flow(prev, curr, dis_preset="FAST"))
        mask = flow_polar_magnitude > magnitude_threshold
        angle_bin = ((flow_polar_angle[mask] % np.pi) * 90 / np.pi).astype(np.int32)
        weight = flow_polar_magnitude[mask].astype(np.float64)
        angle_bins += np.bincount(angle_bin, weights=weight, minlength=90)

    start_idx, end_idx, chosen_bins = select_greedy_hue_window(
        angle_bins, coverage_threshold=coverage_threshold
    )
    
    return start_idx * np.pi / 90, end_idx * np.pi / 90, chosen_bins

def get_lanes_y_pxs(n_frames, background_warped, min_area_for_car_detection):
    background_subtract_threshold = 14

    bottom_edges_y = []
    image_height = 0
    for i in range(n_frames):
        warped_frame = video.get_frame_warped()
        mask = detection.fill_holes(
                background.background_subtract(
                    warped_frame, background_warped, 
                    threshold=background_subtract_threshold, 
                    subtract_percentile=50, normalize=True))
        bbox_image, all_bboxes, bbox_areas = detection.detect_blobs(mask, min_area = min_area_for_car_detection)
        if len(all_bboxes) and len(all_bboxes[0])==4:
            for bbox in all_bboxes:
                bottom_edges_y.append(bbox[1] + bbox[3])
        if image_height==0:
            image_height = warped_frame.shape[0]
        
    bl_y_values = np.asarray(bottom_edges_y, dtype=float)

    # --- Handle empty case early ---
    if bl_y_values.size == 0:
        lanes_y_px = []
    else:
        # --- Histogram of Y values (adaptive bin count) ---
        num_hist_bins = max(32, min(256, max(1, image_height // 10)))
        histogram_counts, bin_edges = np.histogram(
            bl_y_values, bins=num_hist_bins, range=(0, image_height)
        )

        # --- Smooth histogram with a small binomial kernel ---
        smoothing_kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        smoothing_kernel = smoothing_kernel / smoothing_kernel.sum()
        pad = len(smoothing_kernel) // 2
        padded_counts = np.pad(histogram_counts, (pad, pad), mode='edge')
        smoothed_counts = np.convolve(padded_counts, smoothing_kernel, mode='valid')

        # Save hist

        
        plt.figure(figsize=(6,3))
        plt.plot(smoothed_counts)
        plt.tight_layout()
        plt.savefig("final_debug/smoothed_histogram.png", dpi=150)
        plt.close()

        # --- Peak candidates: local maxima in smoothed histogram ---
        n_bins = smoothed_counts.shape[0]
        candidate_idxs = np.where(
            (smoothed_counts[1:-1] > smoothed_counts[:-2]) &
            (smoothed_counts[1:-1] >= smoothed_counts[2:])
        )[0] + 1
        print("Candidate lane peak bin indices:", candidate_idxs.tolist())

        # --- Prominence and separation thresholds scale with data ---
        window_radius = max(3, int(0.01 * n_bins))                      # neighborhood to estimate local minima
        min_separation = max(3, int(0.04 * n_bins))                     # bins between accepted peaks
        dynamic_range = smoothed_counts.max() - smoothed_counts.min()
        min_prominence = max(5.0, 0.01 * dynamic_range)                 # reject tiny ripples

        # --- Score candidates by prominence (and height as tiebreaker) ---
        scored_peaks = []
        for idx in candidate_idxs:
            left_start = max(0, idx - window_radius)
            right_end = min(n_bins, idx + 1 + window_radius)
            left_slice = smoothed_counts[left_start:idx]
            right_slice = smoothed_counts[idx + 1:right_end]
            left_min = left_slice.min() if left_slice.size else smoothed_counts[idx]
            right_min = right_slice.min() if right_slice.size else smoothed_counts[idx]
            local_base = max(left_min, right_min)
            prominence = smoothed_counts[idx] - local_base
            if prominence >= min_prominence:
                scored_peaks.append((idx, prominence, smoothed_counts[idx]))

        scored_peaks.sort(key=lambda t: (t[1], t[2]), reverse=True)

        # --- Greedily keep well-separated strongest peaks ---
        selected_idxs = []
        for idx, _, _ in scored_peaks:
            if all(abs(idx - kept) >= min_separation for kept in selected_idxs):
                selected_idxs.append(idx)
        selected_idxs.sort()

        # --- Convert peak bin indices to Y coordinates (bin centers) ---
        lane_y_centers = 0.5 * (bin_edges[np.array(selected_idxs)] + bin_edges[np.array(selected_idxs) + 1])

        # --- Final integer pixel rows, clamped to [0, H-1] ---
        lanes_y_px = [int(np.clip(y, 0, image_height - 1)) for y in lane_y_centers.tolist()]

    return lanes_y_px

def calculate_roi_polygon(n_frames, car_direction_range):
    """Calculate ROI polygon from video frames."""
    flow_magnitude_threshold = 2.0
    background_subtraction_threshold = 30
    roi_coverage = 0.97
    roi_polygon_sides = 6

    prev=video.get_frame()
    meta_background = background.Background(prev.shape[1], prev.shape[0], size=n_frames)
    for i in range(n_frames):
        curr=video.get_frame()
        optical_flow_polar = optical_flow.flow_to_polar(optical_flow.calculate_optical_flow(prev, curr, dis_preset="FAST"))
        flow_mask = optical_flow.flow_subtract(optical_flow_polar, car_direction_range, flow_magnitude_threshold)
        bg_mask = background.background_subtract(curr, video._background, threshold=background_subtraction_threshold, subtract_percentile=50)
        and_mask = cv2.bitwise_and(flow_mask, bg_mask)
        filled_mask = detection.fill_holes(and_mask)
        meta_background.update(filled_mask)
        prev=curr

    bg = meta_background.get_background_percentile(roi_coverage*100/0.99)
    roi_visual = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    pts_roi, stats_roi, tl_roi, kicks_roi = roi_maker.fit_polygon_to_mask_optimized(bg, roi_polygon_sides, target_coverage=0.99)
    polygon_points = np.array(pts_roi, dtype=np.int32)
    cv2.polylines(roi_visual, [polygon_points], True, (0, 255, 0), 2)
    cv2.imwrite("final_debug/roi_on_mask.png", roi_visual)
    return np.array(pts_roi, dtype=np.int32)

def calibrate():
    """Calibrate camera parameters from video frames."""
    main_movement_range_frame_number = 800
    roi_polygon_frame_number = 800
    warped_bg_window_size = 800
    get_lanes_frame_number = 800

    movement_range_coverage = 0.9
    flow_magnitude_threshold = 2.0
    min_area_for_car_detection = 1600 #TODO: Change based on scaling later

    timec0 = time.perf_counter()
    start_angle, end_angle, chosen_bins = get_main_movement_range(
        main_movement_range_frame_number, 
        coverage_threshold=movement_range_coverage, 
        magnitude_threshold=flow_magnitude_threshold)
    timec1 = time.perf_counter()
    print(f"Main movement range calculation time: {timec1 - timec0:.3f} seconds")
    polygon_pts = calculate_roi_polygon(roi_polygon_frame_number, start_angle, end_angle)
    timec2 = time.perf_counter()
    print(f"ROI polygon calculation time: {timec2 - timec1:.3f} seconds")

    # Basic intrinsics estimation
    frame=video.get_frame()
    w,h = frame[0].shape[1], frame[0].shape[0]
    cx = w // 2
    cy = h // 2
    timec3 = time.perf_counter()
    print(f"Basic intrinsics estimation time: {timec3 - timec2:.3f} seconds")

    # VP calculation

    # Homography calculation
    f = homography.f_from_two_orthogonal_vps(road_vp, perpendicular_vp, cx, cy)
    K_matrix = np.array([
        [     f,   0.0,    cx],
        [   0.0,     f,    cy],
        [   0.0,   0.0,   1.0]
    ], dtype=np.float64)

    r1, r2, r3 = homography.get_rotation_matrix_from_vps(perpendicular_vp, road_vp, K_matrix)
    H_matrix, (W_out, H_out) = homography.build_img_to_bird_homography(
        frame.shape, K_matrix, r1, r2, scale=None, margin=0.01, roi_polygon=polygon_pts, target_width_px=1280.0
    )
    video.set_warping_configs(H_matrix, W_out, H_out)

    # Scale calculation
    background_warped = background.Background(W_out, H_out, warped_bg_window_size)
    for _ in range(warped_bg_window_size):
        frame_warped = video.get_frame_warped()
        background_warped.update(frame_warped)

    lanes_y_pxs = get_lanes_y_pxs(get_lanes_frame_number, background_warped, min_area_for_car_detection)


    with open("final_debug/calibration_info.txt", "w") as fout:
        fout.write("CALIBRATION PARAMETERS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Vanishing Points:\n")
        fout.write(f"  Road VP: ({vp_road[0]:.2f}, {vp_road[1]:.2f})\n")
        fout.write(f"  Parallel VP: ({vp_vertical[0]:.2f}, {vp_vertical[1]:.2f})\n\n")
        fout.write(f"Focal Length: {focal_length:.2f} px\n\n")
        fout.write(f"K Matrix:\n{K_matrix}\n\n")
        fout.write(f"Rotation Vectors:\n")
        fout.write(f"  r1 (road): {r1}\n")
        fout.write(f"  r2 (across): {r2}\n")
        fout.write(f"  r3 (up): {r3}\n\n")
        fout.write(f"Homography Matrix:\n{H_matrix}\n\n")
        fout.write(f"Output size: {W_out} x {H_out}\n")
        fout.write(f"ROI polygon: {polygon_pts.tolist()}\n")
        fout.write(f"Lane Y-pixels: {lanes_y_pxs}\n")
    print("Saved: final_debug/calibration_info.txt")

    roi_area = cv2.contourArea(polygon_pts)
    M = cv2.moments(polygon_pts)
    cx_roi = M['m10'] / M['m00'] if M['m00'] != 0 else 0
    cy_roi = M['m01'] / M['m00'] if M['m00'] != 0 else 0
    with open("final_debug/roi_stats.txt", "w") as fout:
        fout.write("ROI POLYGON STATISTICS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Number of vertices: {len(polygon_pts)}\n")
        fout.write(f"Area: {roi_area:.2f} px²\n")
        fout.write(f"Centroid: ({cx_roi:.2f}, {cy_roi:.2f})\n")
        fout.write(f"Vertices:\n")
        for i, pt in enumerate(polygon_pts):
            fout.write(f"  {i}: ({pt[0]}, {pt[1]})\n")
    print("Saved: final_debug/roi_on_mask.png, final_debug/roi_stats.txt")

    warped_example = video.get_frame_warped()
    cv2.imwrite("final_debug/warped_example.png", warped_example)
    print("Saved: final_debug/warped_example.png")

    return H_matrix, polygon_pts, H_out, W_out, lanes_y_pxs, background_warped._background, final_road_vp1, final_vertical_vp1, K_matrix, r1, r2, r3, f