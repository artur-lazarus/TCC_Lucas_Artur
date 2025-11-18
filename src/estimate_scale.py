import cv2
import numpy as np
import os
import logging
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from video_stream import video

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def _pixel_to_ground_plane(pixel_points):
    """
    Transform 2D pixel coordinates to ground plane coordinates using homography.
    
    Args:
        pixel_points: Array of shape (N, 2) with (x, y) pixel coordinates
        
    Returns:
        Array of shape (N, 2) with (X, Y) ground plane coordinates in pseudo-units
    """
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    if pixel_points.ndim == 1:
        pixel_points = pixel_points.reshape(1, -1)
    
    N = pixel_points.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pixel_h = np.hstack([pixel_points, ones]).T
    
    ground_h = H @ pixel_h
    
    ground_coords = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        w = ground_h[2, i]
        if abs(w) > 1e-6:
            ground_coords[i, 0] = ground_h[0, i] / w
            ground_coords[i, 1] = ground_h[1, i] / w
        else:
            ground_coords[i] = np.nan
    
    return ground_coords


def _measure_distance_3d(point_A, point_B):
    """
    Measure Euclidean distance between two points in ground plane coordinates.
    
    Args:
        point_A: (x, y) pixel coordinates of point A
        point_B: (x, y) pixel coordinates of point B
        
    Returns:
        Distance in pseudo-units, or None if transformation fails
    """
    points_2d = np.array([point_A, point_B], dtype=np.float64)
    points_3d = _pixel_to_ground_plane(points_2d)
    
    if np.any(np.isnan(points_3d)):
        return None
    
    distance = np.linalg.norm(points_3d[1] - points_3d[0])
    return distance

def _find_kde_mode(data, grid_steps=1000):
    """
    Find the mode (peak) of a distribution using Kernel Density Estimation.
    
    Robust alternative to median for finding the most common measurement value.
    Reference: Dubská et al. 2014 [cite: 816-818], Sochor et al. 2017 [cite: 258]
    
    Args:
        data: 1D array of measurements
        grid_steps: Resolution of the grid for finding the peak (default: 1000)
        
    Returns:
        Mode value, or None if computation fails
    """    
    data = np.asarray(data)
    
    q05 = np.percentile(data, 5)
    q95 = np.percentile(data, 95)
    filtered_data = data[(data >= q05) & (data <= q95)]
    
    if filtered_data.size == 0:
        filtered_data = data
    
    if filtered_data.size == 1:
        return float(filtered_data[0])
    
    try:
        kde = gaussian_kde(filtered_data)
    except (np.linalg.LinAlgError, ValueError):
        return float(np.median(filtered_data))
    
    grid_min = np.min(filtered_data)
    grid_max = np.max(filtered_data)
    
    if grid_min == grid_max:
        return float(grid_min)
    
    data_grid = np.linspace(grid_min, grid_max, grid_steps)
    pdf_values = kde.evaluate(data_grid)
    peak_index = np.argmax(pdf_values)
    
    return float(data_grid[peak_index])


def _compute_scale_from_measurements(all_measurements, real_car_width=1.81):
    """
    Compute scene scale (λ) from car width measurements using KDE mode estimation.
    
    Reference: Dubská et al. 2014 [cite: 816-818]
    
    Args:
        all_measurements: List of width measurements in pseudo-units
        real_car_width: Real-world car width in meters (default: 1.81m)
        
    Returns:
        Scene scale λ (meters per pseudo-unit), or None if computation fails
    """
    if len(all_measurements) == 0:
        logging.error("No valid width measurements")
        return None
    
    logging.info(f"Computing scale from {len(all_measurements)} width measurements")
    
    mode_car_width = _find_kde_mode(np.array(all_measurements))
    if mode_car_width is None:
        logging.error("Failed to compute mode from width measurements")
        return None
    
    lambda_final = real_car_width / mode_car_width
    
    logging.info(f"Car width mode: {mode_car_width:.2f} pseudo-units")
    logging.info(f"Scene scale λ: {lambda_final:.6f} m/unit")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    car_widths_array = np.array(all_measurements)
    ax.hist(car_widths_array, bins=20, density=True, alpha=0.6, color='blue', edgecolor='black')
    
    kde = gaussian_kde(car_widths_array)
    x_range = np.linspace(car_widths_array.min(), car_widths_array.max(), 200)
    kde_values = kde.evaluate(x_range)
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
    ax.axvline(mode_car_width, color='green', linestyle='--', linewidth=2, label=f'Mode: {mode_car_width:.2f}')
    
    ax.set_xlabel('Car Width (pseudo-units)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Car Width Distribution\n{len(all_measurements)} samples, mode={mode_car_width:.2f} units', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_path = os.path.join("test_output", "scale_debug", "measurement_histogram.png")
    os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    logging.info(f"Histogram saved to: {histogram_path}")
    
    plt.show(block=False)
    plt.pause(0.1)
    
    return lambda_final


def _find_tangents_to_hull(vp, hull):
    """
    Find two tangent lines from a vanishing point to a convex hull.
    
    Args:
        vp: (x, y) vanishing point coordinates
        hull: Array of [x, y] points forming the convex hull
        
    Returns:
        Tuple of (line1, line2) where each line is (vp, tangent_point)
        Returns (None, None) if hull is too small
    """
    if len(hull) < 2:
        return None, None
        
    vp = np.array(vp)
    hull_points = np.array(hull).reshape(-1, 2)
    
    angles = [np.arctan2(p[1] - vp[1], p[0] - vp[0]) for p in hull_points]
    
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    
    p_min = tuple(hull_points[min_idx])
    p_max = tuple(hull_points[max_idx])
    
    line1 = (tuple(vp), p_min)
    line2 = (tuple(vp), p_max)
    
    return line1, line2


def _line_line_intersection(line1, line2):
    """
    Find intersection point of two lines defined by two points each.
    
    Args:
        line1: ((x1, y1), (x2, y2)) defining first line
        line2: ((x3, y3), (x4, y4)) defining second line
        
    Returns:
        (x, y) intersection point, or None if lines are parallel
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if den == 0:
        return None
        
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    if 0 <= t <= 1 and u >= 0:
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        return (px, py)
    
    px = int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den)
    py = int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den)
    
    return (px, py)

def _get_projected_corners(hull):
    """
    Compute corners A and B for vehicle width measurement.
    
    Based on Dubská et al. 2014, Fig 3 [cite: 746-762]. Finds intersections of
    tangent lines from three vanishing points to the vehicle's convex hull.
    
    Args:
        hull: Convex hull of the vehicle silhouette
        
    Returns:
        Tuple of (corners, tangent_lines):
        - corners: Dict with {'A': (x,y), 'B': (x,y)}
        - tangent_lines: Dict with tangent lines used for computation
        Returns (None, None) if construction fails
    """
    t_red_l, t_red_u = _find_tangents_to_hull(VP1, hull)
    t_green_l, t_green_u = _find_tangents_to_hull(VP2, hull)
    t_blue_l, t_blue_r = _find_tangents_to_hull(VP3, hull)
    
    tangent_lines = {
        "red_lower": t_red_l, "red_upper": t_red_u,
        "green_lower": t_green_l, "green_upper": t_green_u,
        "blue_left": t_blue_l, "blue_right": t_blue_r
    }
    
    if any(v is None for v in tangent_lines.values()):
        return None, None
    
    corners = {}
    
    def dist(p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    a_candidates = [
        _line_line_intersection(t_red_l, t_green_l),
        _line_line_intersection(t_red_l, t_green_u),
        _line_line_intersection(t_red_u, t_green_l),
        _line_line_intersection(t_red_u, t_green_u)
    ]
    a_distances = [dist(a, VP3) for a in a_candidates]
    corners['A'] = a_candidates[np.argmin(a_distances)]
    
    b_candidates = [
        _line_line_intersection(t_green_l, t_blue_l),
        _line_line_intersection(t_green_l, t_blue_r),
        _line_line_intersection(t_green_u, t_blue_l),
        _line_line_intersection(t_green_u, t_blue_r)
    ]
    b_distances = [dist(b, VP1) for b in b_candidates]
    corners['B'] = b_candidates[np.argmax(b_distances)]
    
    if corners['A'] is None or corners['B'] is None:
        return None, None

    return corners, tangent_lines


def _process_frame(frame, show_video=False):
    """
    Process a single frame to extract vehicle width measurements.
    
    Finds vehicle contours, filters edge-cropped vehicles, computes convex hulls,
    determines corners A and B, and measures width in ground plane coordinates.
    
    Args:
        frame: Binary foreground mask from YOLO segmentation
        show_video: If True, return visualization data
        
    Returns:
        If show_video is False: List of width measurements in pseudo-units
        If show_video is True: Tuple of (widths, vis_data_list)
    """
    widths = []
    vis_data_list = [] if show_video else None
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x1, y1, w, h = cv2.boundingRect(contour)
        x2, y2 = x1 + w, y1 + h
        
        margin = 5
        x1_check = max(0, x1 - margin)
        y1_check = max(0, y1 - margin)
        x2_check = min(video.roi_mask.shape[1], x2 + margin)
        y2_check = min(video.roi_mask.shape[0], y2 + margin)
        
        border_mask = np.zeros_like(video.roi_mask)
        border_mask[y1_check:y2_check, x1_check:x2_check] = 255
        inner_mask = np.zeros_like(video.roi_mask)
        inner_y1 = min(y1 + margin, y2_check)
        inner_x1 = min(x1 + margin, x2_check)
        inner_y2 = max(y2 - margin, y1_check)
        inner_x2 = max(x2 - margin, x1_check)
        if inner_y2 > inner_y1 and inner_x2 > inner_x1:
            inner_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 255
        border_mask = cv2.subtract(border_mask, inner_mask)
        
        border_check = cv2.bitwise_and(video.roi_mask, border_mask)
        border_pixels = np.count_nonzero(border_mask)
        valid_pixels = np.count_nonzero(border_check)
        
        if border_pixels > 0 and valid_pixels < border_pixels * 0.9:
            continue
        
        hull = cv2.convexHull(contour)
        hull_points_list = hull.reshape(-1, 2)
        
        corners, tangent_lines = _get_projected_corners(hull_points_list)
        
        if corners is None:
            continue
        
        width = _measure_distance_3d(corners['A'], corners['B'])
        
        if width is not None:
            widths.append(width)
            if show_video:
                vis_data_list.append({
                    'corners': corners,
                    'tangent_lines': tangent_lines,
                    'hull': hull,
                    'width': width
                })
    
    if show_video:
        return widths, vis_data_list
    return widths


def estimate_scale(vp1, vp2, vp3, show_video=False):
    """
    Process video to collect vehicle width measurements and compute scene scale.
    
    Uses global video stream and homography matrix. Processes frames until
    sufficient measurements are collected, then computes scale using KDE.
    
    Args:
        vp1: Vanishing point u (road direction)
        vp2: Vanishing point v (perpendicular)
        vp3: Vanishing point w (vertical)
        show_video: Display and save visualization video (default: False)
        
    Returns:
        Scene scale λ in meters per pseudo-unit
    """
    global VP1, VP2, VP3, H
    
    VP1, VP2, VP3, H = vp1, vp2, vp3, video.H_matrix
    min_measurements = 100
    
    # logging.info(f"Collecting {min_measurements} width measurements for scale estimation")
    
    # video_writer = None
    # if show_video:
    #     output_dir = 'test_output/scale_debug'
    #     os.makedirs(output_dir, exist_ok=True)
    
    # all_measurements = []
    # while len(all_measurements) < min_measurements:
    #     frame_count, frame = video.get_frame_with_roi()
        
    #     if frame is None:
    #         continue
        
    #     mask = video._background.background_subtract_yolo(frame, conf_threshold=0.8)
        
    #     if show_video:
    #         widths, vis_data_list = _process_frame(mask, show_video=True)
            
    #         if video_writer is None and frame is not None:
    #             frame_h, frame_w = frame.shape[:2]
    #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #             video_writer = cv2.VideoWriter(
    #                 'test_output/scale_debug/scale_estimation.mp4',
    #                 fourcc, 10.0, (frame_w, frame_h)
    #             )
            
    #         vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
    #         for vis_data in vis_data_list:
    #             corners = vis_data['corners']
    #             tangent_lines = vis_data['tangent_lines']
    #             hull = vis_data['hull']
    #             width = vis_data['width']
                
    #             cv2.drawContours(vis_frame, [hull], -1, (255, 255, 0), 2)
                
    #             tangent_colors = {
    #                 "red_lower": (0, 0, 255), "red_upper": (0, 0, 255),
    #                 "green_lower": (0, 255, 0), "green_upper": (0, 255, 0),
    #                 "blue_left": (255, 0, 0), "blue_right": (255, 0, 0)
    #             }
                
    #             for name, line in tangent_lines.items():
    #                 if line is not None:
    #                     p1, p2 = line
    #                     color = tangent_colors.get(name, (128, 128, 128))
    #                     vx, vy = int(p1[0]), int(p1[1])
    #                     px, py = int(p2[0]), int(p2[1])
    #                     p_ext_x = int(1.5*px - 0.5*vx)
    #                     p_ext_y = int(1.5*py - 0.5*vy)
    #                     cv2.line(vis_frame, (vx, vy), (p_ext_x, p_ext_y), color, 2)
                
    #             for point_name in ['A', 'B']:
    #                 if point_name in corners:
    #                     point = corners[point_name]
    #                     cv2.circle(vis_frame, point, 6, (0, 255, 255), -1)
    #                     cv2.putText(vis_frame, point_name, (point[0] + 8, point[1] + 8),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    #             pt1 = corners['A']
    #             pt2 = corners['B']
    #             cv2.line(vis_frame, pt1, pt2, (255, 255, 255), 2)
                
    #             mid_x = (pt1[0] + pt2[0]) // 2
    #             mid_y = (pt1[1] + pt2[1]) // 2
    #             cv2.putText(vis_frame, f"W: {width:.2f}", (mid_x, mid_y - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    #         cv2.putText(vis_frame, f"Frame: {frame_count} | Measurements: {len(all_measurements)}/{min_measurements}",
    #                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    #         if video_writer is not None:
    #             video_writer.write(vis_frame)
            
    #         cv2.imshow('Scale Estimation', vis_frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             logging.info("Early exit requested by user")
    #             break
    #     else:
    #         widths = _process_frame(mask, show_video=False)
        
    #     all_measurements.extend(widths)
    
    # if show_video:
    #     if video_writer is not None:
    #         video_writer.release()
    #         logging.info("Saved video to: test_output/scale_debug/scale_estimation.mp4")
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)
    
    # lambda_final = _compute_scale_from_measurements(
    #     all_measurements,
    #     real_car_width=1.81
    # )
    
    # return lambda_final
    return 0.021356