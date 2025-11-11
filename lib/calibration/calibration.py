import os
import numpy as np
import cv2
import sys

import homography
import roi_maker
import vp_detector
import utils

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "detection"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from dubska import detect_vanishing_points_debug
from detection_api import Detection

def save_video_dewarped(video_path,detection_object, H_matrix, roi_polygon, W_out, H_out):
    poly = np.asarray(roi_polygon, dtype=np.int32).reshape(-1, 1, 2)
    d = detection_object

    H_img, W_img = d.frames[0].shape[:2]
    mask_src = np.full((H_img, W_img), 255, dtype=np.uint8)
    cv2.fillPoly(mask_src, [poly], 255)
    bird_frames = []
    for frame in d.frames:
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
        input_for_warp = cv2.bitwise_and(frame, frame, mask=mask_src)
        bird_frame = cv2.warpPerspective(
                            input_for_warp,
                            np.array(H_matrix),
                            (W_out, H_out),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
        bird_frames.append(bird_frame)

    utils.saveGrayscale(bird_frames, video_path+"_birdview.mp4")



def main():
    input_video_path = "dataset/session0_left/video.avi"
    f_calibrated = None

    d = Detection(input_video_path, max_frames=1000, color=False, start_frame=100, frame_interval=5)
    h, w = d.frames[0].shape[0], d.frames[0].shape[1]
    cx = w//2
    cy = h//2
    print(f"Loaded video: {len(d.frames)} frames of size {w}x{h}")

    d.init_flows(dis_preset="FAST")
    d.init_background(method='percentile', percentile=50)

    step = 10
    sampled_images = []

    for i in range(0, len(d.frames), step):
        frame = d.frames[i]
        if frame is None:
            continue
        # Convert to grayscale float32 in [0,1]; handle both color and already-grayscale frames
        if frame is None:
            continue
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray_src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:  # already single channel
            gray_src = frame
        gray = gray_src.astype(np.float32) / 255.0
        sampled_images.append(gray)

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

    # Greedy 15-hue-wide window selection
    window_size = 15
    coverage_threshold = 0.9
    start_idx, end_idx, chosen_bins = vp_detector.select_greedy_hue_window(angle_bins_counts, coverage_threshold=coverage_threshold)
    vp_detector.plot_direction_histogram(window_size=window_size,start_idx=start_idx, end_idx=end_idx, chosen_bins=chosen_bins, angle_bins_counts=angle_bins_counts)

    start_angle = start_idx * np.pi / 90
    end_angle = end_idx * np.pi / 90
    print(f"Selected hue range angles (radians): start={start_angle:.3f}, end={end_angle:.3f}")
    final_road_vp1, final_vertical_vp1 = vp_detector.detect_road_and_vertical_vps((start_angle, end_angle), sampled_images, plot=True)
    print("Final Vertical Vanishing Point:", final_vertical_vp1)
    print("Final Road Vanishing Point:", final_road_vp1)

    f = f_calibrated if f_calibrated is not None else homography.f_from_two_orthogonal_vps(final_road_vp1, final_vertical_vp1, cx, cy)

    K = np.array([
        [     f,   0.0,    cx],
        [   0.0,     f,    cy],
        [   0.0,   0.0,   1.0]
    ], dtype=np.float64)
    r1, r2, r3 = homography.get_rotation_matrix_from_vps(final_vertical_vp1, final_road_vp1, K)
    print("Estimated camera intrinsic matrix K:")
    print(K)
    print("Estimated rotation matrix R:")
    print(np.array_str(np.column_stack((r1, r2, r3)), precision=4, suppress_small=True))

    b = d.flow_subtract(hue_range=(5,40), value_min=6).and_(
        d.median_subtract(threshold_value=14).morphology.fill_holes()
    )

    roi_d = Detection(b.masks, max_frames=500)
    roi_d.init_background(method='percentile', percentile=98)
    bg = roi_d._background.copy()
    cv2.imshow("Background for ROI detection", bg)
    cv2.waitKey(0)

    target = 0.99
    polygon_sides = 6

    pts_corner, stats_corner, tl_corner, kicks_corner = roi_maker.fit_polygon_to_mask(bg, polygon_sides, target_coverage=target)
    print(f"Corner ROI polygon points: {pts_corner}")
    print(f"Corner ROI top-left: {tl_corner}, kicks: {kicks_corner}")
    print(f"Corner ROI stats: {stats_corner}")

    polygon_pts = np.array(pts_corner, dtype=np.int32)
    image = d.frames[0].copy()
    M_img_to_bird, (W_out, H_out) = homography.build_img_to_bird_homography(
        image.shape, K, r1, r2, scale=None, margin=0.01, roi_polygon=polygon_pts, target_width_px=1280.0
    )
    print(f"[DEBUG] bird output size: W_out={W_out}, H_out={H_out}")
    print("Matrix:", M_img_to_bird)

    H_img, W_img = image.shape[:2]
    mask_src = np.full((H_img, W_img), 255, dtype=np.uint8)
    input_for_warp = image
    if polygon_pts is not None:
        poly = np.asarray(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
        mask_src[:] = 0
        cv2.fillPoly(mask_src, [poly], 255)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
        input_for_warp = cv2.bitwise_and(image, image, mask=mask_src)
    
    save_video_dewarped(input_video_path, d, M_img_to_bird, polygon_pts, W_out, H_out)



    

if __name__ == "__main__":
    main()



