import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Ensure the detection package modules are importable when running this file directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "detection"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from dubska import detect_vanishing_points_debug
from detection_api import Detection



def robust_center(points: np.ndarray, k: float = 3.0, use_geomedian: bool = False, iters: int = 3):
    """Robust center estimation via MAD-based outlier rejection and optional geometric median.

    Returns: center (2,), cov (2x2), inliers (M,2), keep_mask (N,)
    """
    P = np.asarray(points, dtype=float)
    if P.size == 0:
        return np.array([np.nan, np.nan], dtype=float), np.eye(2) * 1e-6, P.copy(), np.zeros((0,), dtype=bool)
    keep = np.ones(P.shape[0], dtype=bool)
    med = None
    for _ in range(max(1, int(iters))):
        Q = P[keep]
        if Q.size == 0:
            break
        med = np.median(Q, axis=0)
        mad = np.median(np.abs(Q - med), axis=0) + 1e-9
        keep = (np.abs(P - med) <= float(k) * mad).all(axis=1)
    inliers = P[keep]

    # center
    if inliers.size == 0:
        center = np.array([np.nan, np.nan], dtype=float)
        cov = np.eye(2) * 1e-6
        return center, cov, inliers, keep

    if use_geomedian:
        # 2D geometric median (Weiszfeld)
        c = (np.median(inliers, axis=0) if med is None else med).astype(float)
        for _ in range(50):
            d = np.linalg.norm(inliers - c, axis=1) + 1e-9
            w = 1.0 / d
            c_new = (inliers * w[:, None]).sum(axis=0) / w.sum()
            if np.linalg.norm(c_new - c) < 1e-6:
                c = c_new
                break
            c = c_new
        center = c
    else:
        center = inliers.mean(axis=0)

    cov = np.cov(inliers.T) if inliers.shape[0] >= 3 else np.eye(2) * 1e-6
    return center, cov, inliers, keep


def detect_vertical_from_image(image) -> List[Tuple[float, float]]:
    vps = detect_vps_in_inclination_range(image, inclination_range=[(np.pi/2-0.1, np.pi/2+0.1)])
    sel = [(float(x), float(y)) for (x, y) in vps if float(y) > 0.0]
    return sel[0] if sel else None

def detect_vps_in_inclination_range(image, inclination_range: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    vps = detect_vanishing_points_debug(
        image,
        tensor_smoothing=True,
        show=False,
        save_dir=None,
        space_size=256,
        # Tuning per user request
        edgelet_min_spacing_px=20,
        edge_sigma_yx=(1.0, 1.0),
        edge_threshold=0.30,
        tensor_alpha=1,
        peak_threshold=0.35,
        peak_prominence=0.9,
        peak_min_dist=8,
        support_max_dist_px=10,
        support_top_k=2500,
        random_sample_if_dense=0,
        select_manhattan_best=True,
        candidate_k=10,
        sensor_width_mm=6.17,
        focal_mm_bounds=(40.0, 40.0),
        color_edgelets_by_inclination=True,
        max_edgelets_color_show=20000,
        max_edgelets_ds_show=4000,
        ds_line_samples=64,
        inclination_ranges=inclination_range,
        # Turn off all visualizations for speed/quiet
        viz_input_image=False,
        viz_edges_and_gradients=False,
        viz_nms_local_maxima=False,
        viz_nms_suppressed_map=False,
        viz_inclination_prefilter=False,
        viz_inclination_lengthfilter=False,
        viz_accumulator_and_peaks=False,
        viz_accumulator_all_candidates_numbered=False,
        viz_diamond_space_edgelets_inclination=False,
        viz_folded_two_spaces=False,
        viz_overlay_vps_on_image=False,
        viz_all_candidates_on_image=False,
        viz_edgelets_inclination_on_image=False,
        viz_supporting_lines_combined=False,
        viz_supporting_lines_panels=False,
        viz_quadrants=False,
        viz_manhattan_check=False,
        # Also disable drawing support lines for speed
        draw_support_lines=False,
    )
    return vps

def plot_direction_histogram(window_size, start_idx, end_idx, chosen_bins, angle_bins_counts):
    # Plot histogram and draw the selected window
    x_centers_deg = 2.0 * (np.arange(90) + 0.5)  # 0..180 deg centers
    width_deg = 2.0

    plt.figure(figsize=(10, 4))
    plt.bar(x_centers_deg, angle_bins_counts, width=width_deg*0.9, align='center', edgecolor='black')

    # Shade the selected window; handle wrap-around
    def span(start, end, color='orange', alpha=0.2, label=None):
        x0 = 2.0 * start
        x1 = 2.0 * (end + 1)
        plt.axvspan(x0, x1, color=color, alpha=alpha, label=label)

    if start_idx <= end_idx:
        span(start_idx, end_idx, label=f'selected window ({window_size} hue = {2*window_size}°)')
    else:
        # Wraps around: [start..89] U [0..end]
        span(start_idx, 89)
        span(0, end_idx)
        plt.text(2.0 * ((end_idx + start_idx) / 2.0), plt.ylim()[1]*0.95, f'{2*window_size}° window', ha='center', color='orange')

    # Mark chosen (largest) bin centers inside the window
    for i in chosen_bins:
        plt.axvline(2.0 * (i + 0.5), color='red', linestyle='--', linewidth=1)

    plt.xlabel('Flow orientation (degrees, folded to [0, 180))')
    plt.ylabel('Weighted frequency (sum of V)')
    plt.title('Folded Hue Orientation Histogram with Selected 10-hue Window')
    plt.legend(loc='upper right')
    print("Selected window (hue bins):", start_idx, "->", end_idx,
          "| size:", window_size, "(=", 2*window_size, "degrees)")
    print("Chosen bins inside window:", chosen_bins)
    print("final time= " + str(time.perf_counter()))
    plt.tight_layout()

def find_vp_from_geomedian_and_plot(vp1_list, plot=True):
    if vp1_list:
        vp1_array = np.array([vp for vp in vp1_list if vp is not None], dtype=float)
        center, cov, inliers, keep_mask = robust_center(vp1_array, k=3.0, use_geomedian=False, iters=3)
        outliers = vp1_array[~keep_mask] if keep_mask.shape[0] == vp1_array.shape[0] else np.empty((0, 2), dtype=float)
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            if outliers.size:
                ax.scatter(outliers[:, 0], outliers[:, 1], s=10, c='tab:red', alpha=0.7, label='outliers', edgecolors='none')
            if inliers.size:
                ax.scatter(inliers[:, 0], inliers[:, 1], s=12, c='tab:green', alpha=0.85, label='inliers', edgecolors='none')
            if np.all(np.isfinite(center)):
                ax.scatter([center[0]], [center[1]], s=60, c='gold', marker='*', label='robust VP', edgecolors='black', linewidths=0.6)
            ax.set_xlabel('VP x (pixels)')
            ax.set_ylabel('VP y (pixels)')
            ax.set_title('VP1 positions (step=10, N≈400) with robust center')
            ax.legend(loc='best')
            fig.tight_layout()
            out_png = os.path.join("test_video", "vp_debug", "vp1_scatter_0_4000_step10_robust.png")
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            fig.savefig(out_png, dpi=150)
            print(f"[saved] scatter plot -> {out_png}")
            print(f"[robust] center ~ (x={center[0]:.2f}, y={center[1]:.2f}), inliers={inliers.shape[0]}/{vp1_array.shape[0]}")
            plt.show()
        return center
    return None

def select_greedy_hue_window(counts, window_size=10, coverage_threshold: float = None):
    """Select a circular window of bins.

    Modes
    -----
    1. Legacy fixed-size mode (coverage_threshold is None):
       Greedily add highest-weight bins while the minimal circular arc covering them
       does not exceed ``window_size``; return that arc.
    2. Coverage mode (coverage_threshold provided):
       Find the *minimal-length contiguous circular window* whose summed weight
       reaches at least ``coverage_threshold`` fraction of the total weight.
       This minimizes number of bins subject to the coverage constraint.

    Parameters
    ----------
    counts : array-like of shape (N,)
        Histogram weights per folded hue/orientation bin.
    window_size : int
        Max arc length for legacy mode; ignored in coverage mode.
    coverage_threshold : float in (0,1], optional
        Fraction of total weight that the chosen contiguous window must cover.

    Returns
    -------
    start_idx, end_idx, chosen_indices
        Inclusive start & end (wrap if start>end); chosen_indices are all bins
        inside the window.
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.size
    if N == 0:
        return (0, -1, [])

    if coverage_threshold is None:
        # --- Legacy greedy non-contiguous selection; kept for backward compatibility ---
        order = np.argsort(counts)[::-1]
        selected = []

        def minimal_cover_arc(idxs):
            if not idxs:
                return 0, 0, 0
            arr = np.sort(np.array(idxs, dtype=int))
            diffs = np.diff(np.r_[arr, arr[0] + N])
            max_gap_idx = int(np.argmax(diffs))
            max_gap = diffs[max_gap_idx]
            length = N - max_gap
            start = (arr[(max_gap_idx + 1) % arr.size]) % N
            end = (start + length - 1) % N
            return int(length), int(start), int(end)

        best_len, best_start, best_end = 0, 0, -1
        for idx in order:
            trial = selected + [int(idx)]
            length, start, end = minimal_cover_arc(trial)
            if length <= int(window_size):
                selected = trial
                best_len, best_start, best_end = length, start, end
            else:
                break

        def within_window(i, s, e):
            return (s <= e and s <= i <= e) or (s > e and (i >= s or i <= e))
        chosen = [i for i in selected if within_window(i, best_start, best_end)]
        return best_start, best_end, chosen

    # --- Coverage mode: minimal-length contiguous circular arc meeting threshold ---
    coverage_threshold = float(max(0.0, min(1.0, coverage_threshold)))
    total = float(counts.sum())
    if coverage_threshold <= 0.0 or total <= 0.0:
        return (0, -1, [])
    target = coverage_threshold * total

    # Duplicate array for circular wrap handling
    counts2 = np.concatenate([counts, counts])
    best_len = N + 1
    best_start = 0
    best_end = -1
    cur_sum = 0.0
    j = 0
    for i in range(N):
        while j < i + N and cur_sum < target:
            cur_sum += counts2[j]
            j += 1
        if cur_sum >= target:
            length = j - i
            if length < best_len:
                best_len = length
                best_start = i % N
                best_end = (j - 1) % N
        # Slide window start
        cur_sum -= counts2[i]
        # Early break: remaining bins cannot form shorter window than current best
        if N - i < best_len:  # remaining start positions insufficient
            break

    if best_len == N + 1:  # not found
        return (0, -1, [])

    # Build chosen indices list
    if best_start <= best_end:
        chosen = list(range(best_start, best_end + 1))
    else:
        chosen = list(range(best_start, N)) + list(range(0, best_end + 1))
    return best_start, best_end, chosen

def detect_road_and_vertical_vps(road_angle_range, sampled_images, plot=False):
    road_vp1_list = []
    vertical_vp1_list = []
    for image in sampled_images:
        vertical_vp1_list.append(detect_vertical_from_image(image))
        road_vps = detect_vps_in_inclination_range(image, [road_angle_range])
        road_vp1 = road_vps[0] if len(road_vps) else None
        road_vp1_list.append(road_vp1)
    final_vertical_vp1 = find_vp_from_geomedian_and_plot(vertical_vp1_list, plot=plot)
    final_road_vp1 = find_vp_from_geomedian_and_plot(road_vp1_list, plot=plot)
    return final_road_vp1, final_vertical_vp1

# =====================
# Calibration-constrained joint VP estimation
# =====================
def _vp_constraint(x: np.ndarray, cx: float, cy: float, f: float) -> float:
    """Orthogonality constraint between two VPs under intrinsics (cx, cy, f).

    Let x = [v1x, v1y, v2x, v2y]. The constraint comes from orthogonality of the
    corresponding 3D directions with intrinsic matrix K = diag(f, f, 1) and principal
    point (cx, cy), yielding:

        (v1x-cx)*(v2x-cx) + (v1y-cy)*(v2y-cy) + f^2 = 0

    Returns g(x); feasible points satisfy g(x) ≈ 0.
    """
    v1x, v1y, v2x, v2y = x
    return (v1x - cx) * (v2x - cx) + (v1y - cy) * (v2y - cy) + (f ** 2)


def _project_to_constraint(x: np.ndarray, cx: float, cy: float, f: float) -> np.ndarray:
    """Project a 4D point x to the manifold g(x)=0 using a first-order correction.

    Uses one Newton-style step along the gradient of g to enforce feasibility.
    """
    g = _vp_constraint(x, cx, cy, f)
    v1x, v1y, v2x, v2y = x
    grad = np.array([v2x - cx, v2y - cy, v1x - cx, v1y - cy], dtype=float)
    denom = float(np.dot(grad, grad)) + 1e-12
    return x - (g / denom) * grad


def _constrained_geomedian(
    X: np.ndarray,
    cx: float,
    cy: float,
    f: float,
    max_iter: int = 50,
    tol: float = 1e-6,
):
    """Weiszfeld-style geometric median in R^4 with per-iteration projection.

    Parameters
    - X: (N,4) with rows [v1x, v1y, v2x, v2y]
    - cx, cy, f: camera intrinsics
    - max_iter, tol: iteration controls

    Returns
    - m: (4,) numpy array on the constraint manifold (approximately)
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return None
    # Initial guess: coordinate-wise median projected to the manifold
    m = np.median(X, axis=0)
    m = _project_to_constraint(m, cx, cy, f)
    for _ in range(max(1, int(max_iter))):
        diffs = X - m
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid zero division; very small distances get large but bounded weights
        w = 1.0 / np.maximum(dists, 1e-6)
        m_new = (X * w[:, None]).sum(axis=0) / w.sum()
        m_new = _project_to_constraint(m_new, cx, cy, f)
        if float(np.linalg.norm(m_new - m)) < float(tol):
            m = m_new
            break
        m = m_new
    return m


def detect_road_and_vertical_vps_with_calibration(
    road_angle_range: Tuple[float, float],
    sampled_images: List[np.ndarray],
    cx: float,
    cy: float,
    f: float,
    max_iter: int = 50,
    tol: float = 1e-6,
):
    """Detect per-frame road and vertical VPs, then jointly robustify under intrinsics.

    - road_angle_range: (min_angle, max_angle) in radians for the road-direction VP search
    - sampled_images: list of grayscale images (H,W) float or uint8 (will be passed through)
    - cx, cy, f: camera intrinsics (principal point in px, focal length in px)
    - returns: (road_vp, vertical_vp) as two (2,) arrays or (None, None) if not enough data

    Convention: v1 = road VP, v2 = vertical VP in the 4D stacking [v1x, v1y, v2x, v2y].
    """
    detections_4d = []

    for image in sampled_images:
        v_vert = detect_vertical_from_image(image)
        road_vps = detect_vps_in_inclination_range(image, [road_angle_range])
        v_road = road_vps[0] if len(road_vps) else None

        if v_vert is None or v_road is None:
            continue

        v1x, v1y = float(v_road[0]), float(v_road[1])
        v2x, v2y = float(v_vert[0]), float(v_vert[1])
        if not (np.isfinite([v1x, v1y, v2x, v2y]).all()):
            continue
        detections_4d.append([v1x, v1y, v2x, v2y])

    if len(detections_4d) == 0:
        return None, None

    X = np.asarray(detections_4d, dtype=float)
    mu = _constrained_geomedian(X, cx=cx, cy=cy, f=f, max_iter=max_iter, tol=tol)
    if mu is None or not np.isfinite(mu).all():
        return None, None

    v1_hat = mu[:2]
    v2_hat = mu[2:]
    return v1_hat, v2_hat

def main() -> None:
    video_path = "dataset/session0_center/video.avi"
    d = Detection(video_path, max_frames=1000)
    d.init_flows(dis_preset="FAST")
    
    step = 10
    sampled_images = []

    for i in range(0, len(d.frames), step):
        frame = d.frames[i]
        if frame is None:
            continue

        # Convert to grayscale float32 in [0,1]; handle both color and already-grayscale frames
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray_src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_src = frame
        gray = gray_src.astype(np.float32) / 255.0
        sampled_images.append(gray)

    print(f"[info] extracted {len(sampled_images)} grayscale frames (step={step})")

    angle_bins_counts = np.zeros(90, dtype=np.float64)  # folded orientation
    intensity_threshold = 5  # Ignore very low intensity (background/noise)
    if d._hsv_flows is not None:
        for hsv in d._hsv_flows:
            hue_h = hsv[:, :, 0].astype(np.int16)   # 0..179
            val_v = hsv[:, :, 2].astype(np.float32)
            mask = val_v > intensity_threshold
            if not np.any(mask):
                continue
            h_sel = (hue_h[mask] % 90).astype(np.int32)  # 0..89 folded
            w_sel = val_v[mask].astype(np.float64)       # weights
            # Accumulate weights into counts
            counts = np.bincount(h_sel, weights=w_sel, minlength=90)
            angle_bins_counts += counts

    # Greedy hue window selection (coverage mode example): capture 80% of total weight
    coverage_threshold = 0.80
    window_size = 15  # still used for legacy mode plotting label if desired
    start_idx, end_idx, chosen_bins = select_greedy_hue_window(
        angle_bins_counts,
        window_size=window_size,
        coverage_threshold=coverage_threshold
    )
    plot_direction_histogram(window_size=window_size, start_idx=start_idx, end_idx=end_idx, chosen_bins=chosen_bins, angle_bins_counts=angle_bins_counts)

    start_angle = start_idx * np.pi / 90
    end_angle = end_idx * np.pi / 90

    # ----------------------------------------------
    # Gather per-frame detections once (for dispersion plotting)
    # ----------------------------------------------
    road_vp_list = []
    vert_vp_list = []
    for image in sampled_images:
        v_vert = detect_vertical_from_image(image)
        vps_road = detect_vps_in_inclination_range(image, [(start_angle, end_angle)])
        v_road = vps_road[0] if len(vps_road) else None
        vert_vp_list.append(v_vert)
        road_vp_list.append(v_road)

    # Unconstrained per-axis robust centers (baseline)
    road_arr = np.array([v for v in road_vp_list if v is not None], dtype=float)
    vert_arr = np.array([v for v in vert_vp_list if v is not None], dtype=float)
    road_uncon, _, road_inliers, _ = robust_center(road_arr, k=3.0, use_geomedian=False, iters=3) if road_arr.size else (np.array([np.nan, np.nan]), None, np.empty((0, 2)), None)
    vert_uncon, _, vert_inliers, _ = robust_center(vert_arr, k=3.0, use_geomedian=False, iters=3) if vert_arr.size else (np.array([np.nan, np.nan]), None, np.empty((0, 2)), None)

    # Calibrated joint estimate
    if len(sampled_images) > 0:
        H, W = sampled_images[0].shape[:2]
        cx = W / 2.0
        cy = H / 2.0
    else:
        cx = cy = 0.0
    f = 2668.0

    road_calib, vert_calib = detect_road_and_vertical_vps_with_calibration(
        (start_angle, end_angle), sampled_images, cx=cx, cy=cy, f=f
    )

    print("[unconstrained] Vertical VP:", vert_uncon)
    print("[unconstrained] Road VP:", road_uncon)
    print("[calibrated]   Vertical VP:", vert_calib)
    print("[calibrated]   Road VP:", road_calib)

    # ----------------------------------------------
    # Comparison plots: dispersion + final estimates
    # ----------------------------------------------
    def _scatter_points(ax, pts, label, color, s=10, alpha=0.7):
        if pts is not None and len(pts):
            A = np.array([p for p in pts if p is not None], dtype=float)
            if A.size:
                ax.scatter(A[:, 0], A[:, 1], s=s, c=color, alpha=alpha, label=label, edgecolors='none')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Road VP subplot
    ax = axes[0]
    _scatter_points(ax, road_vp_list, 'road detections', 'tab:blue', s=12, alpha=0.6)
    if np.all(np.isfinite(road_uncon)):
        ax.scatter([road_uncon[0]], [road_uncon[1]], s=80, c='gold', marker='*', edgecolors='black', linewidths=0.6, label='road unconstrained')
    if road_calib is not None and np.all(np.isfinite(road_calib)):
        ax.scatter([road_calib[0]], [road_calib[1]], s=70, c='tab:red', marker='X', label='road calibrated')
    ax.set_title('Road VP dispersion and estimates')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    # Vertical VP subplot
    ax = axes[1]
    _scatter_points(ax, vert_vp_list, 'vertical detections', 'tab:green', s=12, alpha=0.6)
    if np.all(np.isfinite(vert_uncon)):
        ax.scatter([vert_uncon[0]], [vert_uncon[1]], s=80, c='gold', marker='*', edgecolors='black', linewidths=0.6, label='vertical unconstrained')
    if vert_calib is not None and np.all(np.isfinite(vert_calib)):
        ax.scatter([vert_calib[0]], [vert_calib[1]], s=70, c='tab:red', marker='X', label='vertical calibrated')
    ax.set_title('Vertical VP dispersion and estimates')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_png = os.path.join("test_video", "vp_debug", "vp_comparison.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[saved] comparison plot -> {out_png}")
    plt.show()




"""
def main() -> None:
    # Configuration: sample VP1 every 10 frames up to 4000 (exclusive) -> 400 samples
    folder = os.path.join("dataset", "session0_left")
    video_path = os.path.join(folder, "video.avi")
    indices = list(range(0, 4000, 10))
    

    if not os.path.isfile(video_path):
        print(f"[warn] video.avi not found at: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    images = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Convert to grayscale float32 in [0,1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        images.append(gray)

    cap.release()
    print(f"[info] extracted {len(images)} frames from video for VP detection")


    vp1_list = []
    for image in images:
        vps = detect_vps_in_inclination_range(image, inclination_range=[(8*np.pi/180, 38*np.pi/180)])
        vp1 = vps[0] if len(vps) else None
        print("AAAAAAAAA - VP1 detected at:", vp1)
        vp1_list.append(vp1)

    print(f"[summary] collected VP1 for {len(vp1_list)} / {len(indices)} frames")

    # Scatter plot of VP1 positions with robust inlier/outlier classification and final VP estimate
    if vp1_list:
        vp1_array = np.array([vp for vp in vp1_list if vp is not None], dtype=float)
        center, cov, inliers, keep_mask = robust_center(vp1_array, k=3.0, use_geomedian=False, iters=3)
        outliers = vp1_array[~keep_mask] if keep_mask.shape[0] == vp1_array.shape[0] else np.empty((0, 2), dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if outliers.size:
            ax.scatter(outliers[:, 0], outliers[:, 1], s=10, c='tab:red', alpha=0.7, label='outliers', edgecolors='none')
        if inliers.size:
            ax.scatter(inliers[:, 0], inliers[:, 1], s=12, c='tab:green', alpha=0.85, label='inliers', edgecolors='none')
        if np.all(np.isfinite(center)):
            ax.scatter([center[0]], [center[1]], s=60, c='gold', marker='*', label='robust VP', edgecolors='black', linewidths=0.6)
        ax.set_xlabel('VP x (pixels)')
        ax.set_ylabel('VP y (pixels)')
        ax.set_title('VP1 positions (step=10, N≈400) with robust center')
        ax.legend(loc='best')
        fig.tight_layout()
        out_png = os.path.join("test_video", "vp_debug", "vp1_scatter_0_4000_step10_robust.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"[saved] scatter plot -> {out_png}")
        print(f"[robust] center ~ (x={center[0]:.2f}, y={center[1]:.2f}), inliers={inliers.shape[0]}/{vp1.shape[0]}")
        plt.show()
"""

if __name__ == "__main__":
    main()