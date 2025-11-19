import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import detection
import optical_flow
import vp_detector
import estimate_scale
from video_stream import video
import background
import homography
import roi_maker
import time

def _compute_vp3_from_vp1_vp2(vp1_px, vp2_px, K):
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
    d1 = homography.pixel_vp_to_cam_dir(vp1_px, K)
    d2 = homography.pixel_vp_to_cam_dir(vp2_px, K)
    
    # VP3 direction is perpendicular to both VP1 and VP2 (cross product)
    d3 = np.cross(d1, d2)
    d3 /= np.linalg.norm(d3)
    
    # Project back to image coordinates: vp = K @ d
    vp3_h = K @ d3
    vp3_px = (vp3_h[0] / vp3_h[2], vp3_h[1] / vp3_h[2])
    
    return vp3_px

def select_greedy_hue_window(counts, window_size=None, coverage_threshold=None, min_weight_threshold=10.0):
    """Select circular window of bins by window size or coverage threshold with belly filtering.
    
    Parameters
    ----------
    counts : array-like
        Histogram weights per bin.
    window_size : int, optional
        Fixed window size (number of bins). Greedily adds highest bins while arc <= window_size.
    coverage_threshold : float, optional
        Fraction of total weight [0,1]. Finds minimal contiguous window meeting threshold.
    min_weight_threshold : float, optional
        Minimum normalized weight threshold (0-100 scale). Bins below this trigger belly splitting.
        Default 10.0 (10% on normalized scale).
    
    Returns
    -------
    start_idx, end_idx, chosen_bins, threshold_original : int, int, list, float
        Start/end indices (inclusive, wraps if start>end), chosen bin indices, and threshold in original scale.
    
    Notes
    -----
    Exactly one of window_size or coverage_threshold must be provided.
    With coverage_threshold mode, counts are normalized to 0-100 before threshold is applied.
    If low-weight bins found, window is split into "bellies" and best belly is selected.
    """
    
    counts = np.asarray(counts, dtype=float)
    N = counts.size
    if N == 0:
        return 0, -1, [], None
    
    # Normalize counts to 0-100 scale for consistent threshold application
    max_count = counts.max()
    if max_count > 0:
        normalized_counts = (counts / max_count) * 100.0
    else:
        normalized_counts = counts.copy()
    
    if window_size is not None:
        # Fixed-size mode: greedy selection (use original counts)
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
        return best_start, best_end, chosen, None
    
    # Coverage mode with belly-first approach
    total = float(counts.sum())
    if total <= 0.0:
        return 0, -1, [], None
    
    # Convert threshold back to original scale for display
    threshold_original_scale = (min_weight_threshold / 100.0) * max_count
    
    print(f"[select_greedy_hue_window] Belly-first analysis:")
    print(f"  Max count: {max_count:.2f}")
    print(f"  Normalized threshold: {min_weight_threshold:.2f}/100")
    print(f"  Threshold in original scale: {threshold_original_scale:.2f}")
    
    # STEP 1: Split entire histogram into bellies based on normalized weights
    low_weight_mask = normalized_counts < min_weight_threshold
    
    print(f"  Found {np.sum(low_weight_mask)} low-weight bins in entire histogram")
    
    # Split into bellies (contiguous regions without low-weight bins)
    bellies = []
    current_belly = []
    
    for bin_idx in range(N):
        if low_weight_mask[bin_idx]:
            # Low-weight bin - end current belly
            if current_belly:
                bellies.append(current_belly)
                current_belly = []
        else:
            # Good bin - add to current belly
            current_belly.append(bin_idx)
    
    # Don't forget the last belly
    if current_belly:
        bellies.append(current_belly)
    
    if not bellies:
        print(f"  WARNING: No valid bellies found in histogram")
        return 0, -1, [], threshold_original_scale
    
    print(f"  Found {len(bellies)} bellies in histogram")
    
    # STEP 2: Select belly with highest coverage meeting threshold
    best_belly = None
    best_belly_coverage = 0.0
    target_coverage = float(coverage_threshold)
    
    for belly_idx, belly in enumerate(bellies):
        belly_weight = counts[belly].sum()
        belly_coverage = belly_weight / total
        print(f"  Belly {belly_idx}: {len(belly)} bins, coverage={belly_coverage:.3f}")
        
        if belly_coverage > best_belly_coverage:
            best_belly_coverage = belly_coverage
            best_belly = belly
    
    print(f"  Selected belly with coverage={best_belly_coverage:.3f} (target was {target_coverage:.3f})")
    
    # Convert belly to start/end indices
    if len(best_belly) == 0:
        return 0, -1, [], threshold_original_scale
    
    # STEP 3: Check if belly coverage meets threshold, then apply coverage filter relative to ALL
    if best_belly_coverage < target_coverage:
        print(f"\n  Belly coverage {best_belly_coverage:.3f} < threshold {target_coverage:.3f}")
        print(f"  Returning full belly without coverage filter")
        final_start = best_belly[0]
        final_end = best_belly[-1]
        final_bins = best_belly
    else:
        print(f"\n  Belly coverage {best_belly_coverage:.3f} >= threshold {target_coverage:.3f}")
        print(f"  Applying coverage filter relative to ALL histogram weight:")
        
        # Target is relative to ALL histogram weight, not just belly
        target_weight = target_coverage * total
        
        print(f"  Total histogram weight: {total:.2f}")
        print(f"  Target weight: {target_weight:.2f} ({target_coverage:.3f} of ALL)")
    
        # Create circular array of belly for window search
        best_belly_array = np.array(best_belly, dtype=int)
        belly_counts = counts[best_belly_array]
        belly_N = len(best_belly)
        
        # Find minimal contiguous window within belly achieving target (relative to ALL)
        belly_counts2 = np.concatenate([belly_counts, belly_counts])
        window_len, window_start_rel, window_end_rel = belly_N + 1, 0, -1
        cur_sum, j = 0.0, 0
        
        for i in range(belly_N):
            while j < i + belly_N and cur_sum < target_weight:
                cur_sum += belly_counts2[j]
                j += 1
            if cur_sum >= target_weight:
                length = j - i
                if length < window_len:
                    window_len = length
                    window_start_rel = i % belly_N
                    window_end_rel = (j - 1) % belly_N
            cur_sum -= belly_counts2[i]
        
        # Map relative indices back to original histogram indices
        if window_len == belly_N + 1:
            # Could not achieve coverage, use full belly
            print(f"  Could not achieve coverage within belly, using full belly")
            final_start = best_belly[0]
            final_end = best_belly[-1]
            final_bins = best_belly
        else:
            print(f"  Found minimal window: {window_len} bins achieving coverage")
            if window_start_rel <= window_end_rel:
                final_bins = best_belly_array[window_start_rel:window_end_rel + 1].tolist()
            else:
                final_bins = (best_belly_array[window_start_rel:].tolist() + 
                             best_belly_array[:window_end_rel + 1].tolist())
            
            final_start = final_bins[0]
            final_end = final_bins[-1]
            
            final_weight = counts[final_bins].sum()
            final_coverage = final_weight / total
            print(f"  Final window: bins {final_start}-{final_end}, coverage={final_coverage:.3f}")
    
    # Handle circular wrap for final window
    if len(final_bins) > 1:
        gaps = np.diff(sorted(final_bins))
        if np.any(gaps > 1):
            sorted_final = sorted(final_bins)
            max_gap_idx = np.argmax(gaps)
            if max_gap_idx < len(gaps) - 1:
                final_start = sorted_final[max_gap_idx + 1]
                final_end = sorted_final[max_gap_idx]
    
    return final_start, final_end, final_bins, threshold_original_scale


import matplotlib.pyplot as plt
import numpy as np

def debug_plot_histogram(counts, start_idx=None, end_idx=None, chosen_bins=None, title="Histogram Debug", min_weight_threshold=None):
    counts = np.asarray(counts)

    N = counts.size
    xs = np.arange(N)

    plt.figure(figsize=(10, 4))
    plt.title(title)

    # Plot full histogram
    plt.bar(xs, counts, color="lightgray", label="All bins")

    # Highlight chosen bins
    if chosen_bins is not None and len(chosen_bins) > 0:
        plt.bar(xs[chosen_bins], counts[chosen_bins], color="tab:blue", label="chosen bins")

    # Highlight window arc
    if start_idx is not None and end_idx is not None:
        # circular case
        if start_idx <= end_idx:
            win = np.arange(start_idx, end_idx + 1)
        else:
            win = np.r_[np.arange(start_idx, N), np.arange(0, end_idx + 1)]

        plt.bar(xs[win], counts[win], color="tab:orange", alpha=0.6, label="selected window")

        # draw vertical lines for boundaries
        plt.axvline(start_idx, color="green", linestyle="--", label="start")
        plt.axvline(end_idx, color="red", linestyle="--", label="end")

    # Draw horizontal line for min_weight threshold if provided
    if min_weight_threshold is not None:
        plt.axhline(min_weight_threshold, color="purple", linestyle=":", linewidth=2, label=f"min_weight threshold ({min_weight_threshold:.2f})")
    
    plt.xlabel("Bin index")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_debug/direction_histogram.png", dpi=150)
    plt.show()
    plt.pause(0.1)


def get_main_movement_range(n_frames, coverage_threshold=None, window_size=None, magnitude_threshold=2.0, min_weight_threshold=10.0):
    """Find dominant flow orientation range via weighted histogram.
    
    Parameters
    ----------
    n_frames : int
        Number of frames for the calculation.
    coverage_threshold : float
        Fraction of total flow weight to capture (default 0.9).
    window_size : int, optional
        Fixed window size (number of bins).
    magnitude_threshold : float
        Minimum magnitude to consider (default 2.0).
    min_weight_threshold : float
        Minimum normalized weight threshold (0-100 scale) for belly filtering.
    
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
    
    angle_bins = np.zeros(180, dtype=np.float64)
    prev = video.get_frame()[1]
    for i in range(n_frames):
        if i % 50 == 0:
            print(f"Main movement range calculation frame {i+1}/{n_frames}")
        curr = video.get_frame()[1]
        flow_polar_magnitude, flow_polar_angle = optical_flow.flow_to_polar(optical_flow.calculate_optical_flow(prev, curr, dis_preset="FAST"))
        mask = flow_polar_magnitude > magnitude_threshold
        angle_bin = np.clip(((flow_polar_angle[mask]) * 90 / np.pi).astype(np.int32), 0, 179)
        weight = flow_polar_magnitude[mask].astype(np.float64)
        angle_bins += np.bincount(angle_bin, weights=weight, minlength=180)
        prev = curr

    start_idx, end_idx, chosen_bins, min_weight_thresh = select_greedy_hue_window(
        angle_bins, coverage_threshold=coverage_threshold, min_weight_threshold=min_weight_threshold
    )
    debug_plot_histogram(
        angle_bins,
        start_idx=start_idx,
        end_idx=end_idx,
        chosen_bins=chosen_bins,
        title="Greedy Hue Window Selection",
        min_weight_threshold=min_weight_thresh
    )
    
    print(F"[get_main_movement_range]: {(start_idx * np.pi / 90, end_idx * np.pi / 90)}")
    
    return start_idx * np.pi / 90, end_idx * np.pi / 90, chosen_bins

def get_lanes_y_pxs(n_frames, background_warped: background.Background, min_area_for_car_detection):
    print(f"[get_lanes_y_pxs] Starting lane detection with {n_frames} frames, min_area={min_area_for_car_detection}")
    background_subtract_threshold = 14

    bottom_edges_y = []
    image_height = 0
    total_detections = 0
    frames_with_detections = 0
    
    # warped_frame = video.get_frame_warped()[1]
    # background_warped.update(warped_frame)
    # cv2.imwrite("test_output/calibration_debug/warped_example_getlanes.png", warped_frame)
    # cv2.imwrite("test_output/calibration_debug/background_warped.png", background_warped.get_background_percentile(50))
    # cv2.imwrite("test_output/calibration_debug/background_subtract_no_fill_holes.png", background_warped.background_subtract(warped_frame, threshold=background_subtract_threshold, subtract_percentile=50, normalize=True))
    # cv2.imwrite("test_output/calibration_debug/background_subtract_no_fill_holes_no_normalize.png", background_warped.background_subtract(warped_frame, threshold=background_subtract_threshold, subtract_percentile=50, normalize=False))
    # cv2.imwrite("test_output/calibration_debug/background_subtract_fill_holes_no_normalize.png", detection.fill_holes(background_warped.background_subtract(warped_frame, threshold=background_subtract_threshold, subtract_percentile=50, normalize=False)))
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"[get_lanes_y_pxs] Processing frame {i+1}/{n_frames}")
            
        warped_frame = video.get_frame_warped()[1]
        background_warped.update(warped_frame)
        mask = detection.fill_holes(
                background_warped.background_subtract(
                    warped_frame, 
                    threshold=background_subtract_threshold, 
                    subtract_percentile=50, normalize=False))
        bbox_image, all_bboxes, bbox_areas = detection.detect_blobs(mask, min_area = min_area_for_car_detection)
        
        # cv2.imshow("Mask holes filled", mask)
        # cv2.waitKey(1)
        
        frame_detections = 0
        if len(all_bboxes) and len(all_bboxes[0])==4:
            for bbox in all_bboxes:
                bottom_edges_y.append(bbox[1] + bbox[3])
                frame_detections += 1
            frames_with_detections += 1
            total_detections += frame_detections
            
        if i < 10 or (i + 1) % 200 == 0:  # Log details for first 10 frames and every 200th frame
            print(f"[get_lanes_y_pxs] Frame {i+1}: {frame_detections} detections, areas: {bbox_areas}")
            
        if image_height==0:
            image_height = warped_frame.shape[0]
            print(f"[get_lanes_y_pxs] Image height: {image_height} pixels")
    
    print(f"[get_lanes_y_pxs] Detection summary: {total_detections} total detections across {frames_with_detections}/{n_frames} frames")
    print(f"[get_lanes_y_pxs] Average detections per frame with objects: {total_detections/max(1, frames_with_detections):.2f}")
        
    bl_y_values = np.asarray(bottom_edges_y, dtype=float)
    print(f"[get_lanes_y_pxs] Collected {bl_y_values.size} bottom edge Y values")
    
    if bl_y_values.size > 0:
        print(f"[get_lanes_y_pxs] Y value range: {bl_y_values.min():.1f} to {bl_y_values.max():.1f} pixels")
        print(f"[get_lanes_y_pxs] Y value statistics: mean={bl_y_values.mean():.1f}, std={bl_y_values.std():.1f}")

    # --- Handle empty case early ---
    if bl_y_values.size == 0:
        print("[get_lanes_y_pxs] WARNING: No detections found, returning empty lane list")
        lanes_y_px = []
    else:
        # --- Histogram of Y values (adaptive bin count) ---
        num_hist_bins = max(32, min(256, max(1, image_height // 10)))
        print(f"[get_lanes_y_pxs] Creating histogram with {num_hist_bins} bins for image height {image_height}")
        
        histogram_counts, bin_edges = np.histogram(
            bl_y_values, bins=num_hist_bins, range=(0, image_height)
        )
        print(f"[get_lanes_y_pxs] Histogram max count: {histogram_counts.max()}, non-zero bins: {np.count_nonzero(histogram_counts)}")

        # --- Smooth histogram with a small binomial kernel ---
        smoothing_kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        smoothing_kernel = smoothing_kernel / smoothing_kernel.sum()
        pad = len(smoothing_kernel) // 2
        padded_counts = np.pad(histogram_counts, (pad, pad), mode='edge')
        smoothed_counts = np.convolve(padded_counts, smoothing_kernel, mode='valid')
        print(f"[get_lanes_y_pxs] Smoothed histogram range: {smoothed_counts.min():.2f} to {smoothed_counts.max():.2f}")

        # Save hist
        plt.figure(figsize=(6,3))
        plt.plot(smoothed_counts)
        plt.tight_layout()
        plt.savefig("test_output/calibration_debug/smoothed_histogram.png", dpi=150)
        plt.close()

        # --- Peak candidates: local maxima in smoothed histogram ---
        n_bins = smoothed_counts.shape[0]
        candidate_idxs = np.where(
            (smoothed_counts[1:-1] > smoothed_counts[:-2]) &
            (smoothed_counts[1:-1] >= smoothed_counts[2:])
        )[0] + 1
        print(f"[get_lanes_y_pxs] Found {len(candidate_idxs)} candidate peaks at bin indices: {candidate_idxs.tolist()}")

        # --- Prominence and separation thresholds scale with data ---
        window_radius = max(3, int(0.01 * n_bins))                      # neighborhood to estimate local minima
        min_separation = max(3, int(0.04 * n_bins))                     # bins between accepted peaks
        dynamic_range = smoothed_counts.max() - smoothed_counts.min()
        min_prominence = max(5.0, 0.01 * dynamic_range)                 # reject tiny ripples
        
        print(f"[get_lanes_y_pxs] Peak filtering parameters:")
        print(f"  - window_radius: {window_radius}")
        print(f"  - min_separation: {min_separation}")
        print(f"  - min_prominence: {min_prominence:.2f}")
        print(f"  - dynamic_range: {dynamic_range:.2f}")

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
                print(f"[get_lanes_y_pxs] Peak at bin {idx}: prominence={prominence:.2f}, height={smoothed_counts[idx]:.2f}")

        print(f"[get_lanes_y_pxs] {len(scored_peaks)} peaks passed prominence filter")
        scored_peaks.sort(key=lambda t: (t[1], t[2]), reverse=True)

        # --- Greedily keep well-separated strongest peaks ---
        selected_idxs = []
        for idx, prominence, height in scored_peaks:
            if all(abs(idx - kept) >= min_separation for kept in selected_idxs):
                selected_idxs.append(idx)
                print(f"[get_lanes_y_pxs] Selected peak at bin {idx} (prominence={prominence:.2f}, height={height:.2f})")
            else:
                print(f"[get_lanes_y_pxs] Rejected peak at bin {idx} due to proximity to existing peaks")
                
        selected_idxs.sort()
        print(f"[get_lanes_y_pxs] Final selected peak bins: {selected_idxs}")

        # --- Handle case where no peaks were selected ---
        if len(selected_idxs) == 0:
            print("[get_lanes_y_pxs] WARNING: No peaks found after filtering, returning empty lane list")
            lanes_y_px = []
        else:
            # --- Convert peak bin indices to Y coordinates (bin centers) ---
            selected_array = np.array(selected_idxs, dtype=np.int32)
            lane_y_centers = 0.5 * (bin_edges[selected_array] + bin_edges[selected_array + 1])
            print(f"[get_lanes_y_pxs] Lane Y centers (continuous): {lane_y_centers.tolist()}")

            # --- Final integer pixel rows, clamped to [0, H-1] ---
            lanes_y_px = [int(np.clip(y, 0, image_height - 1)) for y in lane_y_centers.tolist()]
            print(f"[get_lanes_y_pxs] Final lane Y pixels (integer): {lanes_y_px}")

    print(f"[get_lanes_y_pxs] Completed lane detection, found {len(lanes_y_px)} lanes")
    return lanes_y_px

def calculate_roi_polygon(n_frames, car_direction_range):
    """Calculate ROI polygon from video frames."""
    flow_magnitude_threshold = 2.0
    background_subtraction_threshold = 30
    roi_time_coverage = 0.99
    roi_space_coverage = 0.99
    roi_polygon_sides = 6
    prev=video.get_frame()[1]
    meta_background = []
    for i in range(n_frames):
        count, curr=video.get_frame()
        if i % 50 == 0:
            print(f"ROI polygon calculation frame {i+1}/{n_frames}")
        optical_flow_polar = optical_flow.flow_to_polar(optical_flow.calculate_optical_flow(prev, curr, dis_preset="FAST"))
        flow_mask = optical_flow.flow_subtract(optical_flow_polar, car_direction_range, flow_magnitude_threshold)
        bg_mask = video._background.background_subtract(curr, threshold=background_subtraction_threshold, subtract_percentile=50)
        and_mask = cv2.bitwise_and(flow_mask, bg_mask)
        filled_mask = detection.fill_holes(and_mask)
        meta_background.append(filled_mask)
        prev=curr
    bg = np.percentile(np.array(meta_background), roi_time_coverage * 100, axis=0).astype(np.uint8)
    roi_visual = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    pts_roi, stats_roi, tl_roi, kicks_roi = roi_maker.fit_polygon_to_mask_optimized(bg, roi_polygon_sides, target_coverage=roi_space_coverage)
    
    # Debugging images saved
    polygon_points = np.array(pts_roi, dtype=np.int32)
    cv2.polylines(roi_visual, [polygon_points], True, (0, 255, 0), 2)
    cv2.imwrite("final_debug/roi_on_mask.png", roi_visual)
    frame_visual = video.get_frame()[1].copy()
    cv2.polylines(frame_visual, [polygon_points], True, (0, 255, 0), 2)
    cv2.imwrite("final_debug/roi_on_frame.png", frame_visual)

    return np.array(pts_roi, dtype=np.int32)

def calibrate(show_video=False, max_width_meters=None, target_width_px=1280.0):
    """
    Calibrate camera parameters from video frames with optional width constraint.
    
    Args:
        show_video: Whether to display debug visualizations
        max_width_meters: Optional maximum width of warped video in meters. 
                         If provided, the warped image will be cropped to this width.
        target_width_px: Target width in pixels for the initial warping (default: 1280.0)
        
    Returns:
        Tuple of (H_matrix, polygon_pts, H_out, W_out, lanes_y_pxs, scale_lambda, background_warped)
    """
    main_movement_range_frame_number = 800
    roi_polygon_frame_number = 800
    warped_bg_window_size = 400
    get_lanes_frame_number = 800
    
    movement_range_coverage = 0.9
    min_weight_threshold = 3.0  # Normalized threshold (0-100 scale)
    flow_magnitude_threshold = 2.0
    min_car_area_m2 = 13.68

    timec0 = time.perf_counter()
    start_angle, end_angle, chosen_bins = get_main_movement_range(
        main_movement_range_frame_number, 
        coverage_threshold=movement_range_coverage, 
        magnitude_threshold=flow_magnitude_threshold,
        min_weight_threshold=min_weight_threshold)
    # start_angle, end_angle = 4.50294947014537, 5.654866776461628

    timec1 = time.perf_counter()
    print(f"Main movement range calculation time: {timec1 - timec0:.3f} seconds")
    print(f"Main movement range: {start_angle}, {end_angle}")
    polygon_pts = calculate_roi_polygon(roi_polygon_frame_number, (start_angle, end_angle))
    # polygon_pts = np.array([[0, 540], [577, 296], [960, 100], [1241, 882], [1154, 1079], [0, 1079]], dtype=np.int32)/
    timec2 = time.perf_counter()
    print(f"ROI polygon calculation time: {timec2 - timec1:.3f} seconds")
    print(f"ROI polygon: {polygon_pts}")

    # Basic intrinsics estimation
    frame = video.get_frame()[1]
    w, h = frame.shape[1], frame.shape[0]
    cx = w // 2
    cy = h // 2
    timec3 = time.perf_counter()
    print(f"Basic intrinsics estimation time: {timec3 - timec2:.3f} seconds")
    
    # Create binary ROI mask from polygon
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon_pts], 255)
    video.set_roi_mask(roi_mask)
    timec4 = time.perf_counter()
    print(f"ROI mask creation time: {timec4 - timec3:.3f} seconds")

    # VP calculation
    road_vp, perpendicular_vp = vp_detector.detect_road_and_cross_vps(show_video=True)
    # vpu = (np.float32(1018.9128), np.float32(-13.503599))
    # vpv = [     -47815,      933.28]
    # road_vp, perpendicular_vp = vpu, vpv

    # Camera calibration
    f = homography.f_from_two_orthogonal_vps(road_vp, perpendicular_vp, cx, cy)
    K_matrix = np.array([
        [     f,   0.0,    cx],
        [   0.0,     f,    cy],
        [   0.0,   0.0,   1.0]
    ], dtype=np.float64)

    r1, r2, r3 = homography.get_rotation_matrix_from_vps(road_vp, perpendicular_vp, K_matrix)
    
    # FIRST PASS: Build initial homography to estimate scale
    print("\n=== FIRST PASS: Initial homography for scale estimation ===")
    H_img_to_plane, initial_bounds = homography.compute_ground_plane_bounds(
        frame.shape, K_matrix, r1, r2, roi_polygon=polygon_pts
    )
    print(f"Initial bounds: X=[{initial_bounds['Xmin']:.2f}, {initial_bounds['Xmax']:.2f}], "
          f"Y=[{initial_bounds['Ymin']:.2f}, {initial_bounds['Ymax']:.2f}]")
    print(f"Initial H: {H_img_to_plane}")
    
    H_matrix_initial, (W_out_initial, H_out_initial), scale_initial = homography.build_img_to_bird_homography_with_bounds(
            H_img_to_plane, initial_bounds, target_width_px=target_width_px
        )
    
    print(f"Initial warped dimensions: {W_out_initial}x{H_out_initial} pseudo-units")
    print(f"Initial scale: {scale_initial:.6f} pseudo-unit/pixels")
    
    video.set_warping_configs(H_matrix_initial, W_out_initial, H_out_initial)
    
    vertical_vp = _compute_vp3_from_vp1_vp2(road_vp, perpendicular_vp, K_matrix)

    # Scale calculation (meters per pseudo-unit)
    scale_lambda = estimate_scale.estimate_scale(road_vp, perpendicular_vp, vertical_vp, show_video, True)
    # scale_lambda = 0.648167
    print(f"\nEstimated scale: {scale_lambda:.6f} meters/pseudo-unit")
    
    # Calculate initial width in meters
    width_pseudo_units = (initial_bounds["Xmax"] - initial_bounds["Xmin"]) * scale_initial
    width_meters = width_pseudo_units * scale_lambda
    print(f"Initial width: {width_meters:.2f} meters ({width_pseudo_units:.2f} pseudo-units)")
    
    # SECOND PASS: Rebuild homography with max_width constraint if provided
    if max_width_meters is not None and width_meters > max_width_meters:
        print(f"\n=== SECOND PASS: Rebuilding homography with max_width={max_width_meters}m ===")
        
        # Crop bounds to achieve max_width_meters and recalculate Y bounds
        final_bounds, actual_width_meters = homography.recalculate_scale_for_max_width(
            initial_bounds, scale_lambda, scale_initial, max_width_meters,
            H_img_to_plane=H_img_to_plane, roi_polygon=polygon_pts, img_shape=frame.shape
        )
        
        print(f"Cropped bounds: X=[{final_bounds['Xmin']:.2f}, {final_bounds['Xmax']:.2f}], "
              f"Y=[{final_bounds['Ymin']:.2f}, {final_bounds['Ymax']:.2f}]")
        print(f"Final width: {actual_width_meters:.2f} meters")
        
        # Rebuild homography with cropped bounds
        H_matrix_final, (W_out_final, H_out_final), scale_final = homography.build_img_to_bird_homography_with_bounds(
                H_img_to_plane, final_bounds, target_width_px=target_width_px
            )
        scale_lambda = actual_width_meters/W_out_final
        
        print(f"Final warped dimensions: {W_out_final}x{H_out_final} pixels")
        print(f"Final scale: {scale_final:.6f} pseudo-units/pixel")
        print(f"Final lambda: {scale_lambda:.6f} m/pseudo-units")
        
        # Update video warping configs
        video.set_warping_configs(H_matrix_final, W_out_final, H_out_final)
        
        # Use final parameters
        H_matrix, W_out, H_out = H_matrix_final, W_out_final, H_out_final
        used_bounds = final_bounds
    else:
        print(f"\n=== No cropping needed (width {width_meters:.2f}m <= max {max_width_meters}m) ===" 
              if max_width_meters is not None else "\n=== No max_width constraint specified ===")
        H_matrix, W_out, H_out = H_matrix_initial, W_out_initial, H_out_initial
        used_bounds = initial_bounds
    
    if perpendicular_vp[0] < cx:
        H_matrix = np.array([
                            [1, 0, 0],
                            [0, -1, H_out - 1],
                            [0, 0, 1]
                            ], dtype=float) @ H_matrix
    
    # Populate background with final warping
    background_warped = background.Background(W_out, H_out, warped_bg_window_size)
    for _ in range(warped_bg_window_size):
        if _ % 50 == 0:
            print(f"Warped background population frame {_}/{warped_bg_window_size}")
        frame_warped = video.get_frame_warped()[1]
        background_warped.update(frame_warped)
    cv2.imwrite("test_output/calibration_debug/warped_example.png", frame_warped)
    print("Saved: test_output/calibration_debug/warped_example.png")
    
    min_car_area_px = int((min_car_area_m2 / (scale_lambda ** 2)))
    lanes_y_pxs = get_lanes_y_pxs(get_lanes_frame_number, background_warped, min_car_area_px)
    print(f"Lane Y pixels: {lanes_y_pxs}")

    # Save calibration info
    with open("test_output/calibration_debug/calibration_info.txt", "w") as fout:
        fout.write("CALIBRATION PARAMETERS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Vanishing Points:\n")
        fout.write(f"  Road VP: ({road_vp[0]:.2f}, {road_vp[1]:.2f})\n")
        fout.write(f"  Parallel VP: ({perpendicular_vp[0]:.2f}, {perpendicular_vp[1]:.2f})\n\n")
        fout.write(f"Focal Length: {f:.2f} px\n\n")
        fout.write(f"K Matrix:\n{K_matrix}\n\n")
        fout.write(f"Rotation Vectors:\n")
        fout.write(f"  r1 (road): {r1}\n")
        fout.write(f"  r2 (across): {r2}\n")
        fout.write(f"  r3 (up): {r3}\n\n")
        fout.write(f"Homography Matrix:\n{H_matrix}\n\n")
        fout.write(f"Output size: {W_out} x {H_out} pixels\n")
        fout.write(f"Scale lambda: {scale_lambda:.6f} meters/pseudo-unit\n")
        fout.write(f"Ground plane bounds (pseudo-units):\n")
        fout.write(f"  X: [{used_bounds['Xmin']:.2f}, {used_bounds['Xmax']:.2f}]\n")
        fout.write(f"  Y: [{used_bounds['Ymin']:.2f}, {used_bounds['Ymax']:.2f}]\n")
        width_pu = used_bounds['Xmax'] - used_bounds['Xmin']
        height_pu = used_bounds['Ymax'] - used_bounds['Ymin']
        fout.write(f"Warped dimensions in meters:\n")
        fout.write(f"  Width: {width_pu * scale_lambda:.2f} m\n")
        fout.write(f"  Height: {height_pu * scale_lambda:.2f} m\n")
        fout.write(f"ROI polygon: {polygon_pts.tolist()}\n")
        fout.write(f"Lane Y-pixels: {lanes_y_pxs}\n")
        if max_width_meters is not None:
            fout.write(f"\nMax width constraint: {max_width_meters:.2f} meters\n")
    print("Saved: test_output/calibration_debug/calibration_info.txt")

    roi_area = cv2.contourArea(polygon_pts)
    M = cv2.moments(polygon_pts)
    cx_roi = M['m10'] / M['m00'] if M['m00'] != 0 else 0
    cy_roi = M['m01'] / M['m00'] if M['m00'] != 0 else 0
    with open("test_output/calibration_debug/roi_stats.txt", "w") as fout:
        fout.write("ROI POLYGON STATISTICS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Number of vertices: {len(polygon_pts)}\n")
        fout.write(f"Area: {roi_area:.2f} px²\n")
        fout.write(f"Centroid: ({cx_roi:.2f}, {cy_roi:.2f})\n")
        fout.write(f"Vertices:\n")
        for i, pt in enumerate(polygon_pts):
            fout.write(f"  {i}: ({pt[0]}, {pt[1]})\n")
    print("Saved: test_output/calibration_debug/roi_on_mask.png, test_output/calibration_debug/roi_stats.txt")

    return H_matrix, polygon_pts, H_out, W_out, lanes_y_pxs, scale_lambda, background_warped

if __name__ == "__main__":
    time0 = time.perf_counter()
    input_video_path = "assets/video.avi"
    colour = False
    target_fps = 10
    video_background_window_size = 800
    video_resolution = (1920, 1080)  # (W, H)

    time1 = time.perf_counter()
    print(f"Setup time: {time1 - time0:.3f} seconds")

    video.set_config(input_video_path, target_fps, colour=colour, make_background=True)
    video.start_background(window_size=video_background_window_size, W=video_resolution[0], H=video_resolution[1])
    time2 = time.perf_counter()
    print(f"Video instantiation time: {time2 - time1:.3f} seconds")

    # Background population
    last_time = time2
    for _ in range(video_background_window_size):
        if _ % 50 == 0:
            print(f"Background population: {_}/{video_background_window_size} - {time.perf_counter() - last_time:.3f}sec")
            last_time = time.perf_counter()
        video.get_frame()
    time3 = time.perf_counter()
    print(f"Initial background population time: {time3 - time2:.3f} seconds")
    
    lane_y = calibrate(True,40,1280)[4]
    image = video.get_frame_warped()[1]
    
    # Draw black lines at detected lane Y positions
    image_with_lanes = image.copy()
    height, width = image_with_lanes.shape[:2]
    
    print(f"Drawing {len(lane_y)} lane lines on warped frame (width: {width}, height: {height})")
    
    for i, y_pos in enumerate(lane_y):
        # Draw black horizontal line across full width at lane Y position
        cv2.line(image_with_lanes, (0, y_pos), (width - 1, y_pos), (0, 0, 0), 2)
        print(f"Drew lane {i+1} at Y position: {y_pos}")
    
    # Save the warped frame with lane lines drawn
    cv2.imwrite("test_output/calibration_debug/warped_frame_with_lanes.png", image_with_lanes)
