import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Import the detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.calibration.dubska import detect_vanishing_points_debug


def top2_positive_y(vps: np.ndarray) -> List[Tuple[float, float]]:
    if vps is None or len(vps) == 0:
        return []
    # vps are already sorted by response in detect_vanishing_points_debug
    sel = [(float(x), float(y)) for (x, y) in vps if float(y) > 0.0]
    return sel[:2]


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


def detect_from_image(path: str) -> List[Tuple[float, float]]:
    image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    vps = detect_vanishing_points_debug(
        image,
        show=True,
        save_dir=None,
        # Tuning per user request
        edgelet_min_spacing_px=20,
        edge_sigma_yx=(1.0, 1.0),
        edge_threshold=0.30,
        peak_threshold=0.35,
        peak_prominence=0.9,
        peak_min_dist=8,
        support_max_dist_px=1000,
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
        inclination_ranges=[(8*np.pi/180, 38*np.pi/180)],
        # Turn off all visualizations for speed/quiet
        viz_input_image=False,
        viz_edges_and_gradients=True,
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
    return top2_positive_y(vps)


def extract_frame_paths(video_path: str, out_dir: str, indices: List[int]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read video frames. Please install opencv-python.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # stop if beyond the end
            break
        # convert to grayscale and save as PNG for the detector to read from path
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, gray)
        saved.append(out_path)
    cap.release()
    return saved


def main() -> None:
    # Configuration: sample VP1 every 10 frames up to 4000 (exclusive) -> 400 samples
    folder = os.path.join("dataset", "session0_left")
    video_path = os.path.join(folder, "video.avi")
    indices = list(range(0, 40, 10))
    tmp_dir = os.path.join("test_video", "vp_debug", "tmpframes_top1")

    if not os.path.isfile(video_path):
        print(f"[warn] video.avi not found at: {video_path}")
        return
    if cv2 is None:
        print("[warn] OpenCV not available; install 'opencv-python' to enable video frame processing.")
        return

    # Extract the requested frames to disk for the detector
    try:
        frame_paths = extract_frame_paths(video_path, tmp_dir, indices)
    except Exception as e:
        print(f"[error] could not extract frames: {e}")
        return

    # Compute VP1 (positive Y) per frame
    vp1_list = []  # (frame_idx, (x,y))
    for fp in frame_paths:
        vps = detect_from_image(fp)
        vp1 = vps[0] if vps else None
        base = os.path.basename(fp)
        try:
            idx = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception:
            idx = -1
        if vp1 is not None:
            vp1_list.append((idx, vp1))

    print(f"[summary] collected VP1 for {len(vp1_list)} / {len(indices)} frames")

    # Scatter plot of VP1 positions with robust inlier/outlier classification and final VP estimate
    if vp1_list:
        pts = np.array([p[1] for p in vp1_list], dtype=float)
        center, cov, inliers, keep_mask = robust_center(pts, k=3.0, use_geomedian=False, iters=3)
        outliers = pts[~keep_mask] if keep_mask.shape[0] == pts.shape[0] else np.empty((0, 2), dtype=float)

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
        print(f"[robust] center ~ (x={center[0]:.2f}, y={center[1]:.2f}), inliers={inliers.shape[0]}/{pts.shape[0]}")
        plt.show()


if __name__ == "__main__":
    main()
