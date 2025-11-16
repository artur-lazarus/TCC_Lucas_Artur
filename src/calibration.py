import os
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

import detection
import vp_detector

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


def get_main_movement_range(flows_polar, coverage_threshold=None, window_size=None, magnitude_threshold=2.0):
    """Find dominant flow orientation range via weighted histogram.
    
    Parameters
    ----------
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

    if (window_size is None) == (coverage_threshold is None):
        raise ValueError("Exactly one of window_size or coverage_threshold must be provided")
    
    angle_bins = np.zeros(90, dtype=np.float64)
    
    for magnitude, angle in flows_polar:
        mask = magnitude > magnitude_threshold
        if not np.any(mask):
            continue
        
        # Fold to [0, π) and bin to [0, 89]
        bins = ((angle[mask] % np.pi) * 90 / np.pi).astype(np.int32)
        weights = magnitude[mask].astype(np.float64)
        angle_bins += np.bincount(bins, weights=weights, minlength=90)
    
    start_idx, end_idx, chosen_bins = select_greedy_hue_window(
        angle_bins, coverage_threshold=coverage_threshold
    )
    
    return start_idx * np.pi / 90, end_idx * np.pi / 90, chosen_bins

def calculate_roi_polygon(d, frames, roi_polygon_sides, roi_coverage, start_angle, end_angle, flow_magnitude_threshold):
    masks_bg_subtract = d.background_subtract_multiple(frames, threshold=14, normalize=True, percentiles=(10,90))
    masks_flow_subtract = d.flow_subtract_multiple(len(frames)-1, direction_range=(start_angle, end_angle), threshold=flow_magnitude_threshold, save=True)
    masks_and = [cv2.bitwise_and(mb, mf) for mb, mf in zip(masks_bg_subtract, masks_flow_subtract)]
    masks = d.fill_holes_multiple(masks_and)
    writer=cv2.VideoWriter("final_debug/masks.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (masks[0].shape[1], masks[0].shape[0]), False)
    for mask in masks_flow_subtract:
        writer.write(mask.astype(np.uint8))
    writer.release()

    d_bg = detection.Detection()
    d_bg.init_background_populated(masks)
    bg = d_bg._background.get_background_percentile(roi_coverage*100/0.99)
    roi_visual = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    
    
    pts_roi, stats_roi, tl_roi, kicks_roi = roi_maker.fit_polygon_to_mask_optimized(bg, roi_polygon_sides, target_coverage=0.99)
    polygon_points = np.array(pts_roi, dtype=np.int32)
    cv2.polylines(roi_visual, [polygon_points], True, (0, 255, 0), 2)
    cv2.imwrite("final_debug/roi_on_mask.png", roi_visual)
    return np.array(pts_roi, dtype=np.int32)

def get_lanes_y_pxs(d_warped, warped_frames, min_area_for_car_detection):
    masks_bg_filled = d_warped.fill_holes_multiple(d_warped.background_subtract_multiple(warped_frames, threshold=14, normalize=True, percentiles=(10,90)))
    writer=cv2.VideoWriter("final_debug/masks_pre_lanes.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (masks_bg_filled[0].shape[1], masks_bg_filled[0].shape[0]), False)
    for mask in masks_bg_filled:
        writer.write(mask.astype(np.uint8))
    writer.release()


    bbox_images, all_bboxes, bbox_areas = d_warped.detect_blobs_multiple(masks_bg_filled, min_area=min_area_for_car_detection)
    bottom_edges_y = []
    if len(all_bboxes[0]) != 0:
        print("Len of bbox: ", len(all_bboxes[0][0]))
        print("Bbox_0:", all_bboxes[0][0])

    for bbox in all_bboxes:
        if len(bbox) == 0:
            continue
        if len(bbox[0]) < 4:
            continue
        bottom_edges_y.append(bbox[0][1] + bbox[0][3])

    image_height = warped_frames[0].shape[0]
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
        print("Scored lane peak candidates (idx, prominence, height):", scored_peaks)

        # --- Greedily keep well-separated strongest peaks ---
        selected_idxs = []
        for idx, _, _ in scored_peaks:
            if all(abs(idx - kept) >= min_separation for kept in selected_idxs):
                selected_idxs.append(idx)
        selected_idxs.sort()
        print("Selected lane peak bin indices:", selected_idxs)

        # --- Convert peak bin indices to Y coordinates (bin centers) ---
        lane_y_centers = 0.5 * (bin_edges[np.array(selected_idxs)] + bin_edges[np.array(selected_idxs) + 1])

        # --- Final integer pixel rows, clamped to [0, H-1] ---
        lanes_y_px = [int(np.clip(y, 0, image_height - 1)) for y in lane_y_centers.tolist()]

    return lanes_y_px


def calibrate(frames):
    """Calibrate camera parameters from video frames."""
    vp_undersampling = 10
    dis_preset = "FAST"
    vp_direction_range_coverage = 0.9
    flow_magnitude_threshold = 2.0
    roi_coverage = 0.97
    roi_polygon_sides = 6
    min_area_for_car_detection = 1600 #TODO: Change based on scaling later

    timec0= time.perf_counter()
    # Detection initialized and used for refinements (like flow range)
    d = detection.Detection()
    timec1 = time.perf_counter()
    print(f"Detection initialization time: {timec1 - timec0:.3f} seconds")
    d.init_flows(frames, dis_preset=dis_preset)
    timec2 = time.perf_counter()
    print(f"Flow initialization time: {timec2 - timec1:.3f} seconds")
    d.init_background_populated(frames)
    timec3 = time.perf_counter()
    print(f"Background initialization time: {timec3 - timec2:.3f} seconds")

    # ROI extraction
    

    # Basic intrinsics estimation
    w,h = frames[0].shape[1], frames[0].shape[0]
    cx = w // 2
    cy = h // 2
    timec4 = time.perf_counter()
    print(f"Basic intrinsics estimation time: {timec4 - timec3:.3f} seconds")

    #Vp detection and final homography parameters computation
    start_angle, end_angle, chosen_bins = get_main_movement_range(d._flows_polar, coverage_threshold=vp_direction_range_coverage, magnitude_threshold=flow_magnitude_threshold)
    timec5 = time.perf_counter()
    print(f"VP detection preparation time: {timec5 - timec4:.3f} seconds")
    samples_for_vp = frames[::vp_undersampling]
    final_road_vp1, final_vertical_vp1 = vp_detector.detect_road_and_vertical_vps((start_angle, end_angle), samples_for_vp, plot=True)

    timec6 = time.perf_counter()
    print(f"VP detection time: {timec6 - timec5:.3f} seconds")
    f = homography.f_from_two_orthogonal_vps(final_road_vp1, final_vertical_vp1, cx, cy)
    timec7 = time.perf_counter()
    print(f"Focal length computation time: {timec7 - timec6:.3f} seconds")
    K_matrix = np.array([
        [     f,   0.0,    cx],
        [   0.0,     f,    cy],
        [   0.0,   0.0,   1.0]
    ], dtype=np.float64)

    r1, r2, r3 = homography.get_rotation_matrix_from_vps(final_vertical_vp1, final_road_vp1, K_matrix)
    timec8 = time.perf_counter()
    print(f"Rotation matrix computation time: {timec8 - timec7:.3f} seconds")

    polygon_pts = calculate_roi_polygon(d, frames, roi_polygon_sides, roi_coverage, start_angle, end_angle, flow_magnitude_threshold)
    timec9 = time.perf_counter()
    print(f"ROI polygon calculation time: {timec9 - timec8:.3f} seconds")
    H_matrix, (W_out, H_out) = homography.build_img_to_bird_homography(
        frames[0].shape, K_matrix, r1, r2, scale=None, margin=0.01, roi_polygon=polygon_pts, target_width_px=1280.0
    )
    timec10 = time.perf_counter()
    print(f"Homography matrix computation time: {timec10 - timec9:.3f} seconds")

    frames_warped = [cv2.warpPerspective(frame, 
                                         H_matrix, 
                                         (W_out, H_out),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0) for frame in frames]
    timec11 = time.perf_counter()
    print(f"Frame warping time: {timec11 - timec10:.3f} seconds")
    d_warped = detection.Detection()
    d_warped.init_background_populated(frames_warped)
    lanes_y_pxs = get_lanes_y_pxs(d_warped, frames_warped, min_area_for_car_detection)
    timec12 = time.perf_counter()
    print(f"Lane y-pixels extraction time: {timec12 - timec11:.3f} seconds")
    return H_matrix, polygon_pts, H_out, W_out, lanes_y_pxs, d_warped._background, final_road_vp1, final_vertical_vp1, K_matrix, r1, r2, r3, f